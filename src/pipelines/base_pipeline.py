import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import deepspeed

import torch
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.schedulers import *
from einops import rearrange

from ..configs.base_config import InstantiateConfig, PrintableConfig
from ..configs.config_utils import to_immutable_dict
from ..models import TextEncoderConfig, Transformer2DModelConfig, UNet2DConfig, VisualTokenizer, VisualTokenizerConfig, VoxelTokenizerConfig
from ..utils import pack_data, log_to_rank0, count_numel


@dataclass
class EDMTrainConfig(PrintableConfig):
    sigma_data: float = 1
    P_mean: float = -1.2
    P_std: float = 1
    min_snr_gamma: float = 5
    num_train_timesteps: int = 1000

    c_skip_type: Literal["EDM", "DDPM"] = "EDM"
    c_out_type: Literal["VPred", "EDM", "DDPM"] = "VPred"
    c_in_type: Literal["EDM", "DDPM"] = "EDM"
    c_noise_type: Literal["EDM", "DDPM"] = "EDM"
    loss_weight_type: Literal["EDM", "SoftminSNR", "SNR"] = "EDM"
    noise_dist_type: Literal["EDM", "DDPM"] = "EDM"

    def setup(self, diffusers_scheduler):
        self.num_train_timesteps = diffusers_scheduler.config.num_train_timesteps  # DDPM
        self.alphas_cumprod = diffusers_scheduler.alphas_cumprod  # DDPM

    def c_skip(self, sigma):
        if self.c_skip_type == "EDM":
            return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)
        elif self.c_skip_type == "DDPM":
            return 1
        else:
            raise NotImplementedError

    def c_out(self, sigma):
        if self.c_out_type == "EDM":
            return (self.sigma_data * sigma) / (sigma**2 + self.sigma_data**2) ** 0.5
        elif self.c_out_type == "VPred":
            return -(self.sigma_data * sigma) / (sigma**2 + self.sigma_data**2) ** 0.5
        elif self.c_out_type == "DDPM":
            return -sigma
        else:
            raise NotImplementedError

    def c_in(self, sigma):
        if self.c_in_type == "EDM":
            return 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        elif self.c_in_type == "DDPM":
            return 1 / (sigma**2 + 1) ** 0.5
        else:
            raise NotImplementedError

    def c_noise(self, sigma):
        if self.c_noise_type == "EDM":
            return 0.25 * torch.log(sigma)
        elif self.c_noise_type == "DDPM":
            alpha_cumprod = 1 / (1 + sigma**2)
            self.alphas_cumprod = self.alphas_cumprod.to(sigma.device)
            return torch.argmin(torch.abs(self.alphas_cumprod[None, ...] - alpha_cumprod[..., None]), dim=1)
        else:
            raise NotImplementedError

    def loss_weight(self, sigma):
        """
        Revised version of the following:

            # Old version
            if self.loss_weight_type == "EDM":
                return (sigma**2 + self.sigma_data**2) / (sigma**2 * self.sigma_data**2)
            elif self.loss_weight_type == "MinSNR":
                # Efficient Diffusion Training via Min-SNR Weighting Strategy: https://arxiv.org/pdf/2303.09556.pdf
                # In EDM, SNR = s(t)^2 / (s(t)^2 * sigma(t)^2) = 1 / sigma(t)^2.
                # This loss weight strategy assumes that we are predicting x0, so according to the paper, w_t = min(SNR(t), gamma)
                return torch.min(sigma**-2, self.min_snr_gamma * torch.ones_like(sigma))
            elif self.loss_weight_type == "SoftminSNR":
                # from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
                return 1 / (sigma**2 + self.sigma_data**2)
            elif self.loss_weight_type == "SNR":
                # from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
                return sigma**-2
            else:
                raise NotImplementedError

        and in original pipeline foward:

            # Old version
            pred_original_sample = EDM.c_skip(sigmas_repeat) * noisy_latents + EDM.c_out(sigmas_repeat) * model_output
            loss_weight = EDM.loss_weight(sigmas)
            res = {"pred": pred_original_sample, "target": latents, "loss_weight": loss_weight}

        We modify this to avoid numerical error in EDM.loss_weight(sigmas).
        In old version this function return λ(σ) in EDM paper, now we return λ(σ) * c_out(σ)^2.
        Also we remove MinSNR since it mostly work on x prediction case.
        """
        if self.loss_weight_type == "EDM":
            return torch.ones_like(sigma) if isinstance(sigma, torch.Tensor) else 1
        elif self.loss_weight_type == "SoftminSNR":
            # from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
            return (sigma * self.sigma_data) ** 2 / (sigma**2 + self.sigma_data**2) ** 2
        elif self.loss_weight_type == "SNR":
            # from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
            return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        else:
            raise NotImplementedError

    def sample_sigma(self, batch_size, device):
        if self.noise_dist_type == "EDM":
            return torch.exp(
                self.P_mean
                + self.P_std
                * torch.randn(
                    [
                        batch_size,
                    ],
                    device=device,
                )
            )
        elif self.noise_dist_type == "DDPM":
            idx = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            return ((1 / self.alphas_cumprod[idx]) - 1) ** 0.5
        else:
            raise NotImplementedError


@dataclass
class BasePipelineConfig(InstantiateConfig):
    """Configuration for Pipeline instantiation"""

    _target: Type = field(default_factory=lambda: StableDiffusionPipeline)

    target: Optional[str] = None  # e.g. "StableDiffusionPipeline"
    """specify the target class to instantiate"""

    # EDMTrainConfig(
    #     c_in_type="EDM",
    #     c_noise_type="VPred",
    #     c_skip_type="EDM",
    #     c_out_type="EDM",
    #     loss_weight_type="EDM",
    #     noise_dist_type="EDM",
    # )
    edm_config: Optional[EDMTrainConfig] = None
    """EDM config, default to EDM"""

    ckpt_path: Optional[str] = None
    """Diffusers ckpt path"""
    unet_config: Optional[UNet2DConfig] = None
    """unet config"""
    transformer_config: Optional[Transformer2DModelConfig] = None
    """transformer config"""
    text_encoder_config: Optional[TextEncoderConfig] = None
    """text encoder config"""
    vae_config: Optional[VisualTokenizerConfig] = None
    """vae config"""
    voxel_vae_config: Optional[VoxelTokenizerConfig] = None
    """voxel vae config"""
    diffusers_vae_ckpt_path: Optional[str] = None
    """vae ckpt path"""
    clip_ckpt_path: Optional[dict] = None
    """clip ckpt pathes"""
    distributed_clip: bool = False
    """distributed store clip model"""

    noise_offset: Optional[float] = None
    """The scale of noise offset."""
    endfix_prompt: Optional[str] = None
    """The endfix_prompt prompt."""
    target: Optional[str] = None  # e.g. "StableDiffusionPipeline"
    """specify the target class to instantiate"""
    enable_vae_slicing: bool = True
    """Whether to enable vae slicing"""

    seed: int = 666
    """The random seed."""
    call: Dict[str, Any] = to_immutable_dict({})
    """The inference call arguments for the pipeline."""
    scheduler_kwargs: Optional[Dict[str, Any]] = None
    """kwargs for testing noise scheduler"""
    measure_time: bool = False
    """Whether to measure the time of each step."""
    offload: bool = True
    """Whether to offload pipeline when vae decode."""

    def from_pretrained(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        if self.target is not None:
            self._target = eval(self.target)

        if self.unet_config is not None:
            unet = self.unet_config.from_pretrained(self.ckpt_path)
            if self.unet_config.gradient_checkpointing:
                unet._supports_gradient_checkpointing = True
                unet.enable_gradient_checkpointing()
            kwargs.update({"unet": unet})

        if self.transformer_config is not None:
            transformer = self.transformer_config.from_pretrained(self.ckpt_path)
            kwargs.update({"transformer": transformer})

        if self.vae_config is not None:
            vae = self.vae_config.from_pretrained()
            log_to_rank0(f"decoder numel is {count_numel(vae.decoder)} | encoder numel is {count_numel(vae.encoder)}")
            kwargs.update({"vae": vae})
        elif self.diffusers_vae_ckpt_path is not None:
            vae = AutoencoderKL.from_pretrained(self.diffusers_vae_ckpt_path)
            kwargs.update({"vae": vae})
        
        if self.voxel_vae_config is not None:
            voxel_vae = self.voxel_vae_config.from_pretrained()
            log_to_rank0(f"=============loaded voxel vae=============")
            kwargs.update({"voxel_vae": voxel_vae})

        if self.text_encoder_config is not None:
            tokenizer, text_encoder = self.text_encoder_config.from_pretrained()
            kwargs.update({"tokenizer": tokenizer})
            kwargs.update({"text_encoder": text_encoder})

        if self.scheduler_kwargs is not None:
            class_name = self.scheduler_kwargs.pop("_class_name")
            scheduler_cls = eval(class_name)
            if self.ckpt_path is not None:
                scheduler = scheduler_cls.from_pretrained(self.ckpt_path, subfolder="scheduler", **self.scheduler_kwargs)
            else:
                scheduler = scheduler_cls(**self.scheduler_kwargs)
            self.scheduler_kwargs["_class_name"] = class_name
            kwargs.update({"scheduler": scheduler})

        if self.clip_ckpt_path is not None:
            import open_clip
            from transformers import CLIPTextModel, CLIPTokenizer

            # Load CLIP model and tokenizer
            clip_g_model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=self.clip_ckpt_path["clip_g_path"])
            clip_g_model.dtype = torch.float16  # hack

            clip_g_model = clip_g_model.to(torch.float16).to(torch.cuda.current_device())
            clip_l_model = CLIPTextModel.from_pretrained(self.clip_ckpt_path["clip_l_path"])
            clip_l_model = clip_l_model.to(torch.float16).to(torch.cuda.current_device())
            tokenizer_clip = CLIPTokenizer.from_pretrained(self.clip_ckpt_path["clip_tokenizer_path"])

            if self.distributed_clip:
                before_partition_numel = count_numel(clip_g_model)
                deepspeed.zero.Init(module=clip_g_model, dtype=torch.float16)
                log_to_rank0(f"before partition: {before_partition_numel} | after partition: {count_numel(clip_g_model)}")
                deepspeed.zero.Init(module=clip_l_model, dtype=torch.float16)

            kwargs.update({"clip_g_model": clip_g_model})
            kwargs.update({"clip_l_model": clip_l_model})
            kwargs.update({"tokenizer_clip": tokenizer_clip})

        if self.ckpt_path is not None:
            pipeline = self._target.from_pretrained(
                self.ckpt_path,
                requires_safety_checker=False,
                safety_checker=None,
                **kwargs,
            )
        else:
            pipeline = self._target(**kwargs)

        if self.clip_ckpt_path is not None and self.distributed_clip:
            pipeline.train_status = None

        if self.edm_config is not None:
            self.edm_config.setup(pipeline.scheduler)

        if self.enable_vae_slicing and self.vae_config is not None and not "num_frames" in set(inspect.signature(pipeline.vae.forward).parameters.keys()):
            pipeline.vae.enable_slicing()

        pipeline.pipeline_config = self

        return pipeline


class PipelineMixin:
    @staticmethod
    def prepare_call_kwargs(pipeline, batch, **kwargs):
        call_kwargs = {}
        for k in ["data_paths", "prompts"]:
            if k in batch:
                batch_size = len(batch[k])
                break
        num_frames = pipeline.unet.num_frames

        if "prompt" not in kwargs:
            call_kwargs["prompt"] = [prompt for prompt in batch["prompts"] for _ in range(num_frames)]
        if "generator" not in kwargs:
            call_kwargs["generator"] = torch.Generator(device=pipeline.device).manual_seed(pipeline.pipeline_config.seed)
        if "output_type" not in kwargs:
            call_kwargs["output_type"] = "pt"

        call_kwargs.update(pipeline.pipeline_config.call)
        call_kwargs.update(kwargs)

        if hasattr(pipeline, "unet") and hasattr(pipeline.unet, "set_num_videos"):
            if "guidance_scale" not in call_kwargs or call_kwargs["guidance_scale"] > 1:
                num_videos = batch_size * 2
            else:
                num_videos = batch_size
            pipeline.unet.set_num_videos(num_videos)

        return call_kwargs

    @torch.no_grad()
    def get_latents(self, batch, is_auto_pack=True):
        latents = []
        if "vae_latents" in batch:
            latent_params = batch["vae_latents"]
            if not isinstance(latent_params, list):
                latent_params = [latent_params]
            for latent_param in latent_params:
                batch_size = latent_param.shape[0]
                latent_param = rearrange(latent_param, "b f c h w -> (b f) c h w")
                latent = DiagonalGaussianDistribution(latent_param).sample()
                latent = latent * self.vae.config.scaling_factor
                latent = rearrange(latent, f"(b f) c h w -> b c f h w", b=batch_size)
                latents.append(latent)
        else:
            samples = batch["data"]
            if not isinstance(samples, list):
                samples = [samples]
            for sample in samples:
                if sample.ndim == 4:
                    sample = rearrange(sample, "b c h w -> b 1 c h w")
                latents.append(self.video2latents(sample, out_pattern="b c f h w"))

        if len(latents) > 1 and is_auto_pack:
            latents, masks = pack_data(latents)
        elif len(latents) > 1:
            latents, masks = latents, None
        else:
            latents, masks = latents[0], None
        return latents, masks

    @torch.no_grad()
    def video2latents(self, video, out_pattern="b f c h w", in_pattern="b f c h w", vae_pattern="(b f) c h w"):
        batch_size = video.shape[0]
        if isinstance(self.vae, VisualTokenizer):
            vae_pattern = self.vae.config.vae_pattern
        video = rearrange(video, f"{in_pattern} -> {vae_pattern}", b=batch_size)
        latents = self.vae.encode(video).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = rearrange(latents, f"{vae_pattern} -> {out_pattern}", b=batch_size)
        return latents

    def val_gt(self, batch):
        frames = rearrange(batch["data"], "b f c h w -> (b f) c h w")
        return frames

    @torch.inference_mode()
    def val_vae(self, batch, in_pattern="b f c h w", out_pattern="(b f) c h w"):
        latents = self.get_latents(batch, is_auto_pack=False)[0]
        assert isinstance(latents, torch.Tensor), f"type(latents) is {type(latents)}"
        latents /= self.vae.config.scaling_factor
        batch_size = latents.shape[0]
        if isinstance(self.vae, VisualTokenizer):
            vae_pattern = self.vae.config.vae_pattern
        else:
            vae_pattern = "(b f) c h w"
        latents = rearrange(latents, f"b c f h w -> {vae_pattern}", b=batch_size)
        decoded = self.vae.decode(latents).sample
        decoded = rearrange(decoded, f"{vae_pattern} -> {out_pattern}", b=batch_size)
        return decoded

    def compute_loss(self, result_dict, mask=None):
        # mask shape = b 1 f h w
        pred, target = result_dict["pred"], result_dict["target"]

        loss_weight = 1 if "loss_weight" not in result_dict else result_dict["loss_weight"].float()
        if mask is None:
            loss = ((pred.float() - target.float()) ** 2).reshape(target.shape[0], -1).mean(dim=1)
        else:
            loss = ((pred.float() - target.float()).mul_(mask) ** 2).reshape(target.shape[0], target.shape[1], -1).sum(dim=2)
            loss = (loss / mask.reshape(target.shape[0], 1, -1).sum(dim=2)).mean(dim=1)
        loss_reweight = (loss * loss_weight).mean()  # MSE loss
        loss_orginal = loss.mean()  # MSE loss
        return {"total_loss": loss_reweight, "loss_unweight": loss_orginal}

    def generate_noise(self, latents, noise_offset=None):
        noise = torch.randn_like(latents)
        if noise_offset is not None:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += noise_offset * torch.randn_like(noise, device=latents.device)
        return noise

    def edm_step(self, batch_size, latents):
        # EDM
        EDM = self.pipeline_config.edm_config
        sigmas = EDM.sample_sigma(batch_size, latents.device).to(latents.dtype)
        if latents.ndim == 4:
            sigmas_repeat = sigmas[:, None, None, None].repeat_interleave(self.unet.num_frames, dim=0)
        else:
            sigmas_repeat = sigmas[:, None, None, None, None]
        noise = self.generate_noise(latents, self.pipeline_config.noise_offset)
        noisy_latents = latents + noise * sigmas_repeat
        scaled_input = EDM.c_in(sigmas_repeat) * noisy_latents
        scaled_sigmas = EDM.c_noise(sigmas)
        loss_weight = EDM.loss_weight(sigmas)
        if latents.ndim == 4:
            scaled_sigmas = scaled_sigmas.repeat_interleave(self.unet.num_frames, dim=0)
            loss_weight = loss_weight.repeat_interleave(self.unet.num_frames, dim=0)
        target = (latents - EDM.c_skip(sigmas_repeat) * noisy_latents) / EDM.c_out(sigmas_repeat)
        return scaled_input, scaled_sigmas, target, loss_weight
