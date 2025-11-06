from typing import Dict, Optional, List
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src/submodule/TRELLIS"))

from ...data import DataConfig_3DMaster_Part
from ...engine import AdamWOptimizerConfig, SchedulerConfig, TrainerConfig, StabilityConfig
from ...models import *
from ...pipelines import *


train_dataset_csv_path = "dataset/partversexl/train.csv"
eval_dataset_csv_path = "dataset/partversexl/val.csv"
resume_ckpt_path = "pretrained_model/fullpart/transformer/s1.ckpt"
ss_flow_weights_dir = "pretrained_model/trellis/ckpts"
part_dir="dataset/partversexl/anno_infos"
part_cond_dir="dataset/partversexl/renders_cond"

voxel_vae_config = VoxelTokenizerConfig(
    enc_pretrained="pretrained_model/trellis/ckpts/ss_enc_conv3d_16l8_fp16",
    dec_pretrained="pretrained_model/trellis/ckpts/ss_dec_conv3d_16l8_fp16",
    torch_hub_dir="~/.cache/torch/hub", # for loading dino_v2
)

def config(
    use_ema: bool = True,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_grad_norm: float = 1.0,
    training_precision: str = "bf16",
    num_inference_steps: int = 25,
    guidance_scale: float = 5.0,
    negative_prompt: str = "low-quality, blurry", # NOTE(lihe): 
    transformer_ckpt_path: Optional[str] = None,
    timestep_shift: float = 1.0,
    max_ids_num: int = 3,
    save_ema: bool = False,
    dataloader_num_processes: int = 8,
    step_per_save: int = 1000,
    train_dataset_csv_path: str = "",
    eval_dataset_csv_path: str = "",
    trainable_modules: List = ["transformer"],
    nontrainable_modules: List = [],
    global_part: bool = False,
):
    return TrainerConfig(
        num_epochs=1000,
        training_precision=training_precision,
        step_per_save=step_per_save,
        step_per_val=4000,
        val_at_begin=False,
        val_types=["i2v"],
        use_ema=use_ema,
        save_ema=save_ema,
        step_per_ema=1,
        ema_decay=0.9999,
        ema_start_step=0,
        seed=None,
        delete_deepspeed_weights=True, 
        train_data = DataConfig_3DMaster_Part(
            dataset_csv_path=train_dataset_csv_path,
            part_dir=part_dir,
            part_cond_dir=part_cond_dir,
            id_mapping=False,
            global_part=global_part,
            max_num=30,
            cond_img_num=24, # 24
            part_column="part_id",
            max_ids_num=max_ids_num,
            batch_size=batch_size,
            num_processes=dataloader_num_processes,
            select_prompt_from_list=True,
        ),
        joint_train_prob=0,
        val_data=DataConfig_3DMaster_Part(
            dataset_csv_path=eval_dataset_csv_path,
            part_dir=part_dir,
            part_cond_dir=part_cond_dir,
            id_mapping=False,
            metadata_csv=None,
            global_part=global_part,
            max_num=30,
            cond_img_num=24, # 24
            part_column="part_id",
            max_ids_num=max_ids_num,
            batch_size=1,
            num_samples=2, # number of inference during eval
            use_determinstic_dataset=True,
            shuffle =False,
            select_prompt_from_list = True,
            eval=True,
        ),
        trainable_modules=trainable_modules,
        nontrainable_modules=nontrainable_modules,
        optimizer=AdamWOptimizerConfig(fused=True, lr=lr, max_grad_norm=max_grad_norm),
        scheduler=SchedulerConfig(),
        stability=StabilityConfig(stability_protection=True),
        pipeline=JointDiTSingle3DPipelineConfig(
            ckpt_path=None,
            vae_config=None,
            voxel_vae_config=voxel_vae_config,
            proportion_empty_prompts=0.1,
            transformer_config=TransformerXLModelConfigSingle(
                transformer_ckpt_path=transformer_ckpt_path,
                zero_linear=False,
                abandon_img_cond=False,
                drop_img_conds=0.1,
                cornner_pos_emb=True,
                id_emb=True,
                in_out_emb=True,
                cornner_in_out=False,
                ss_flow_weights_dir=ss_flow_weights_dir,
            ),
            call={
                # "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
            timestep_shift=timestep_shift,
        ),
    )


personal_configs_part: Dict[str, TrainerConfig] = {}


# 3dmaster
personal_configs_part["3dmaster_part"] = config(
    use_ema=True,
    batch_size=1,#2,
    lr=5e-5,#5e-6, 
    max_grad_norm=0.1,
    timestep_shift=3.0,
    transformer_ckpt_path=resume_ckpt_path,
    dataloader_num_processes=2,
    step_per_save = 1000,
    train_dataset_csv_path = train_dataset_csv_path, 
    eval_dataset_csv_path = eval_dataset_csv_path,
    global_part = True, 
)