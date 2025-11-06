from torch.utils.tensorboard import SummaryWriter

from ..utils import *
from .trainer import *


@dataclass
class TrainerGANConfig(TrainerConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: TrainerGAN)
    """target class to instantiate"""


class ManagerGAN(torch.nn.Module):
    def __init__(self, pipeline, is_generator=True):
        super().__init__()
        module_names, _ = pipeline._get_signature_keys(pipeline)
        module_names = sorted(module_names)  # To ensure every rank gets the same module_names
        for module_name in module_names:
            module = getattr(pipeline, module_name, None)
            if module is not None:
                if isinstance(module, torch.nn.Module) or isinstance(module, torch.nn.Parameter):
                    if ("discriminator" in module_name) ^ is_generator:
                        setattr(self, module_name, module)

    def save_ckpts(self, ds_state_dict, path):
        if hasattr(self, "visual_tokenizer"):
            save_model_from_ds("visual_tokenizer", ds_state_dict, path)
        if hasattr(self, "discriminator"):
            save_model_from_ds("discriminator", ds_state_dict, path)


class TrainerGAN(Trainer):
    def __init__(self, config: TrainerConfig):
        if is_amd():
            wrap_conv3d()
        self.config = config

        config.set_timestamp()
        self.exp_root = config.get_base_dir()
        os.makedirs(self.exp_root, exist_ok=True)

        # init deepspeed, state, logger
        self.init_state_and_logger()
        if config.seed is not None:
            set_seed(config.seed, device_specific=True)

        if self.state.is_main_process:
            config.print_to_terminal()

        self.pipeline = config.pipeline.from_pretrained()
        self.netG = ManagerGAN(self.pipeline, is_generator=True)
        self.netD = ManagerGAN(self.pipeline, is_generator=False)

        # Fake a manager obj with eval func
        class dummy_manager:
            def __init__(self, netG, netD):
                self.netG = netG
                self.netD = netD

            def eval(self):
                self.netD.eval()
                self.netG.eval()

        self.manager = dummy_manager(self.netG, self.netD)

        self.pipeline.to(self.state.device)

        # for ema
        if self.config.use_ema:
            self.ema_model = EMAModel(get_trainable_parameters(self.netG), decay=self.config.ema_decay, update_after_step=self.config.ema_start_step)
            self.ema_model.to(self.state.device)

        self.train_dataloader = config.train_data.setup().dataloader
        self.val_dataloader = config.val_data.setup().dataloader

        self.optimizer_G, self.optimizer_D, self.lr_scheduler_G, self.lr_scheduler_D, self.num_steps_per_epoch = self.init_train()

        (
            self.netG,
            self.netD,
            self.optimizer_G,
            self.optimizer_D,
            self.lr_scheduler_G,
            self.lr_scheduler_D,
            self.train_dataloader,
            self.val_dataloader,
        ) = prepare(
            self.netG,
            self.netD,
            self.optimizer_G,
            self.optimizer_D,
            self.lr_scheduler_G,
            self.lr_scheduler_D,
            self.train_dataloader,
            self.val_dataloader,
        )

        self.netG, self.optimizer_G, _, _ = deepspeed.initialize(
            model=self.netG, optimizer=self.optimizer_G, config=self.deepspeed_config, dist_init_required=True
        )
        self.netD, self.optimizer_D, _, _ = deepspeed.initialize(
            model=self.netD, optimizer=self.optimizer_D, config=self.deepspeed_config, dist_init_required=True
        )

        # summary writer tensorboard
        class dummy_logger_tb:
            def __init__(self, summary_writer):
                self.summary_writer = summary_writer

        self.logger_tb = dummy_logger_tb(SummaryWriter(log_dir=os.path.join(self.exp_root, "tensorboard")))

        self.validation_rank_num = min(math.ceil(self.config.val_data.num_samples / self.config.val_data.batch_size), torch.distributed.get_world_size())
        self.validation_process_group = torch.distributed.new_group(ranks=[i for i in range(self.validation_rank_num)])

        if self.state.is_main_process:
            config.save_config()

    def init_train(self):
        # for lr
        if self.config.optimizer.scale_lr:
            self.config.optimizer.lr *= self.state.gradient_accumulation_steps * self.config.train_data.batch_size * self.state.num_processes

        # for requires_grad

        self.netG.requires_grad_(False)
        self.netD.requires_grad_(False)
        self.set_modules_requires_grad(self.config.trainable_modules, target=self.netG)
        self.set_modules_requires_grad(self.config.trainable_modules, target=self.netD)

        # for optimizers
        parameters_to_optmize_G = []
        parameters_to_optmize_D = []
        for name, param in self.netG.named_parameters():
            if param.requires_grad:
                parameters_to_optmize_G.append(param)
        for name, param in self.netD.named_parameters():
            if param.requires_grad:
                parameters_to_optmize_D.append(param)

        num_optimized_params = lambda parameters_to_optmize: sum([param.numel() for param in parameters_to_optmize])

        log_to_rank0("Total of optimized parameters for generator:", num_optimized_params(parameters_to_optmize_G))
        log_to_rank0("Total of optimized parameters for discriminator:", num_optimized_params(parameters_to_optmize_D))

        optimizer_G = self.config.optimizer.setup(parameters_to_optmize_G)
        optimizer_D = self.config.optimizer.setup(parameters_to_optmize_D)

        # for lr_cheduler

        self.global_batch_size = self.state.num_processes * self.state.gradient_accumulation_steps * self.config.train_data.batch_size
        num_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.state.gradient_accumulation_steps / self.state.num_processes)
        self.config.max_steps = max(self.config.num_epochs * num_steps_per_epoch, self.config.max_steps)
        self.config.num_epochs = max(math.ceil(self.config.max_steps / num_steps_per_epoch), self.config.num_epochs)

        lr_scheduler_G = self.config.scheduler.setup(
            optimizer_G, gradient_accumulation_steps=self.state.gradient_accumulation_steps, num_training_steps=self.config.max_steps
        )
        lr_scheduler_D = self.config.scheduler.setup(
            optimizer_D, gradient_accumulation_steps=self.state.gradient_accumulation_steps, num_training_steps=self.config.max_steps
        )

        return optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, num_steps_per_epoch

    def set_modules_requires_grad(self, module_info, requires_grad=True, target=None):
        # support for skipping invalid modules
        if target is None:
            target = self.manager
        for module in module_info:
            if isinstance(module, str):
                if hasattr(target, module):
                    self.logger.info(f"[init train] set {module} requires grad")
                    model = getattr(target, module)
                    model.requires_grad_(requires_grad)
                else:
                    self.logger.info(f"[init train] {module} is not in model")
            else:
                module, suffix = list(module.keys())[0], list(module.values())[0]
                if isinstance(suffix, str):
                    suffix = [suffix]
                if hasattr(target, module):
                    self.logger.info(f"[init train] set {module}: {suffix} requires grad")
                    model = getattr(target, module)
                    for name, module in model.named_modules():
                        if name.endswith(tuple(suffix)):
                            for params in module.parameters():
                                params.requires_grad_(requires_grad)
                else:
                    self.logger.info(f"[init train] {module} is not in model")

    def save_ckpt(self, global_step):
        num_trained_data = global_step * self.global_batch_size
        save_root = os.path.join(self.exp_root, "checkpoints", f"checkpoint-{num_trained_data}")
        ds_save_path = os.path.join(save_root, "deepspeed")

        if self.config.use_ema and self.state.is_main_process:
            ema_path = os.path.join(save_root, "ema")
            os.makedirs(ema_path, exist_ok=True)

            torch.save(self.ema_model.state_dict(), os.path.join(ema_path, "ema.pt"))  # NOTE: we don't resume exp now, so we don't save for saving memory
            with ema_context(self.ema_model, self.netG):
                self.netG.save_ckpts(self.netG.state_dict(), ema_path)  # TODO: fix multi card ema
                torch.save(self.netG.state_dict(), os.path.join(ema_path, "ema.ckpt"))

        save_deepspeed_state(self.netG, self.lr_scheduler_G, os.path.join(ds_save_path, "netG"), self.state.local_process_index)
        save_deepspeed_state(self.netD, self.lr_scheduler_D, os.path.join(ds_save_path, "netD"), self.state.local_process_index)
        self.logger.info(f"Saved state to {ds_save_path}")
        torch.distributed.barrier()
        if self.state.is_main_process:
            self.logger.info("start save torch model weights.")
            ds_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(os.path.join(ds_save_path, "netG"))
            self.netG.save_ckpts(ds_dict, save_root)
            # get deepspeed model ckpt name
            latest_path_tag_or_ref = os.path.join(ds_save_path, "netG", "latest")
            if os.path.isfile(latest_path_tag_or_ref):
                with open(latest_path_tag_or_ref, "r") as fd:
                    tag = fd.read().strip()
                    # get deepspeed model ckpt path
                    latest_path = os.path.join(ds_save_path, "netG", tag)
            # get list of deepspeed model ckpt path
            model_ckpt_files = deepspeed.utils.zero_to_fp32.get_model_state_files(latest_path)
            for file_path in model_ckpt_files:
                o_d = torch.load(file_path)
                # remove deepspeed model weight from ckpt only
                o_d["module"] = {}
                # remove deepspeed model freeze weights if possible
                if "frozen_param_fragments" in o_d:
                    del o_d["frozen_param_fragments"]
                torch.save(o_d, file_path)
            self.logger.info("finish save torch model weights.")

    def load_ckpt(self, ckpt_path=None, no_skip=False):
        if ckpt_path is None:
            dirs = os.listdir(os.path.join(self.exp_root, "checkpoints"))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            ckpt_path = os.path.join(self.exp_root, "checkpoints", dirs[-1])

        if not os.path.exists(ckpt_path):
            self.logger.error(f"Checkpoint '{ckpt_path}' does not exist.")
            raise ValueError()

        self.logger.info(f"Resuming from checkpoint {ckpt_path}")
        load_deepspeed_state(self.netG, self.lr_scheduler_G, os.path.join(ckpt_path, "deepspeed", "netG"), self.state.local_process_index)
        load_deepspeed_state(self.netD, self.lr_scheduler_D, os.path.join(ckpt_path, "deepspeed", "netD"), self.state.local_process_index)
        if self.config.use_ema:
            self.ema_model.load_state_dict(torch.load(os.path.join(ckpt_path, "ema", "ema.pt")))

        num_trained_data = int(ckpt_path.split("-")[-1])
        global_step = num_trained_data // (self.state.gradient_accumulation_steps * self.config.train_data.batch_size * self.state.num_processes)
        first_epoch = global_step // self.num_steps_per_epoch

        if no_skip:
            skipped_dataloader = self.train_dataloader
        else:
            global_accumulation_step = global_step * self.state.gradient_accumulation_steps
            skipped_step = global_accumulation_step % (self.num_steps_per_epoch * self.state.gradient_accumulation_steps)
            skipped_dataloader = skip_first_batches(self.train_dataloader, num_batches=skipped_step)
        return first_epoch, global_step, skipped_dataloader

    def train(self):
        first_epoch, global_step = 0, 0

        if self.config.resume_ckpt_path:
            first_epoch, global_step, skipped_dataloader = self.load_ckpt(self.config.resume_ckpt_path, self.config.resume_no_skip)

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(0, self.config.max_steps), disable=not self.state.is_local_main_process)
        progress_bar.set_description(self.config.experiment_name)

        init_step = global_step * self.global_batch_size
        if self.config.use_ema:
            milestone_ema = init_step + self.config.data_per_ema
        milestone_save, milestone_val = init_step + self.config.data_per_save, init_step + self.config.data_per_val

        if self.config.val_at_begin:
            self.validate(global_step)

        for epoch in range(first_epoch, self.config.num_epochs):
            train_dataloader = self.train_dataloader
            if self.config.resume_ckpt_path and epoch == first_epoch:
                # using skipped_dataloader in the first epoch to skip batchs already used
                train_dataloader = skipped_dataloader
                progress_bar.update(global_step)

            for step, batch in enumerate(train_dataloader):
                with torch.no_grad():
                    for param in self.netD.parameters():
                        param.data = param.data.clone().contiguous()
                    for param in self.netG.parameters():
                        param.data = param.data.clone().contiguous()

                self.netG.train()
                self.netD.train()

                if global_step >= self.config.max_steps:
                    break

                # forward
                batch["data"] = batch["data"].to(self.state.device)
                if self.state.mixed_precision == "fp16":
                    batch = self._cast_inputs_half(batch)

                outputs = self.pipeline.forward_gan(batch)

                # backpropagate
                self.netD.requires_grad_(False)
                self.optimizer_G.zero_grad()
                loss_dict_G = self.pipeline.compute_loss_g(*outputs)
                total_loss_G = loss_dict_G["total_loss"]
                self.netG.backward(total_loss_G)
                self.netG.step()
                self.netD.requires_grad_(True)
                self.optimizer_D.zero_grad()
                loss_dict_D = self.pipeline.compute_loss_d(*outputs)
                total_loss_D = loss_dict_D["total_loss"]
                self.netD.backward(total_loss_D)
                self.netD.step()

                self.lr_scheduler_G.step()
                self.lr_scheduler_D.step()

                if self.netG.is_gradient_accumulation_boundary():
                    global_step += 1
                    num_trained_data = global_step * self.global_batch_size

                    if self.config.use_ema:
                        if num_trained_data >= milestone_ema:
                            milestone_ema += self.config.data_per_ema
                            self.ema_model.step(self.netG.parameters())
                            with ema_context(self.ema_model, self.netG):
                                self.validate(global_step, type_name="ema")

                    logs = {
                        "epoch": epoch,
                        "global_bs": self.global_batch_size,
                        "steps_per_save": self.config.data_per_save // self.global_batch_size,
                        "steps_per_val": self.config.data_per_val // self.global_batch_size,
                        "loss_G": total_loss_G.item(),
                        "loss_D": total_loss_D.item(),
                        "lr": self.lr_scheduler_G.get_last_lr()[0],
                        "grad_norm": float(self.netG.optimizer._global_grad_norm),
                        "memory alloca:": torch.cuda.memory_allocated(),
                        "memory max": torch.cuda.max_memory_allocated(),
                    }

                    progress_bar.set_postfix(**logs)
                    progress_bar.update(1)

                    # val
                    if num_trained_data >= milestone_val:
                        milestone_val += self.config.data_per_val
                        self.validate(global_step)

                    if self.state.is_main_process:
                        for k, v in loss_dict_G.items():
                            self.logger_tb.summary_writer.add_scalar(f"Loss_G/{k}", v.item(), global_step)
                        for k, v in loss_dict_D.items():
                            self.logger_tb.summary_writer.add_scalar(f"Loss_D/{k}", v.item(), global_step)

                    # save ckpt
                    if num_trained_data >= milestone_save:
                        milestone_save += self.config.data_per_save
                        self.save_ckpt(global_step)
