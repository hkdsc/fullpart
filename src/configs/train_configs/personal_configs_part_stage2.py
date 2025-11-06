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
part_slat_dir="dataset/partversexl/textured_part_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
global_slat_dir="dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"


voxel_vae_config = VoxelTokenizerConfigStage2(
    enc_pretrained="../tmp/interactive_quick_3d/weights/image/ckpts/slat_enc_swin8_B_64l8_fp16",
    dec_pretrained="../tmp/interactive_quick_3d/weights/image/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16",
    dec_gs_pretrained="../tmp/interactive_quick_3d/weights/image/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
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
    negative_prompt: str = "low-quality, blurry",
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
    slat_norm: bool = False,
):
    return TrainerConfig(
        num_epochs=10000,
        training_precision=training_precision,
        step_per_save=step_per_save,
        step_per_val=3, 
        val_at_begin=False,
        val_types=["i2v"],
        # val_types=["i2v2"], # continue eval
        use_ema=use_ema,
        save_ema=save_ema,
        step_per_ema=1,
        ema_decay=0.9999,
        ema_start_step=0,
        seed=None,
        delete_deepspeed_weights=True, # False,
        train_data = DataConfig_3DMaster_Part(
            dataset_csv_path=train_dataset_csv_path,
            part_dir=part_dir,
            part_cond_dir=part_cond_dir,
            part_slat_dir=part_slat_dir,
            global_slat_dir=global_slat_dir,
            stage='2',
            rotate_slat=True,
            id_mapping=False, 
            metadata_csv=None,
            global_part=global_part,
            max_num=30, # 30
            cond_img_num=24, # 24
            part_column="part_id",
            max_ids_num=max_ids_num,
            batch_size=batch_size,
            num_processes=dataloader_num_processes,
            select_prompt_from_list=True,
            slat_norm=slat_norm,
        ),
        joint_train_prob=0,
        val_data=DataConfig_3DMaster_Part(
            dataset_csv_path=eval_dataset_csv_path,
            part_dir=part_dir, # new
            part_cond_dir=part_cond_dir,
            part_slat_dir=part_slat_dir,
            global_slat_dir=global_slat_dir,
            id_mapping=False,
            stage='2',
            rotate_slat=True, # no use
            global_part=global_part,
            max_num=30, # 30
            cond_img_num=24, # 24
            part_column="part_id",
            max_ids_num=max_ids_num,
            batch_size=1,
            num_samples = 2, # number of inference during eval
            use_determinstic_dataset = True,
            shuffle = False,
            select_prompt_from_list = True,
            eval = True, #NOTE
            slat_norm=slat_norm,
        ),
        trainable_modules=trainable_modules,
        nontrainable_modules=nontrainable_modules,
        optimizer=AdamWOptimizerConfig(fused=True, lr=lr, max_grad_norm=max_grad_norm),
        scheduler=SchedulerConfig(),
        stability=StabilityConfig(stability_protection=True),
        pipeline=JointDiTSingle3DPipelineConfigStage2(
            ckpt_path=None,
            vae_config=None,
            voxel_vae_config=voxel_vae_config, # NOTE(lihe): new added
            s1_save_dir='sample_results/sample_results_vox',
            proportion_empty_prompts=0.1,
            transformer_config=TransformerXLModelConfigSingleStage2(
                transformer_ckpt_path=transformer_ckpt_path,
                # abandon_img_cond=False,
                drop_img_conds=0.1,
                cornner_pos_emb=True,
                id_emb=True,#False,
                rotate_slat=True,
                drop_global=0.0, #0.1, # NOTE
                slat_norm=slat_norm,
                ss_flow_weights_dir=ss_flow_weights_dir,
            ),
            call={
                # "negative_prompt": negative_prompt,
                "save_dir": "./sample_results/sample_results_ref",
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
            timestep_shift=timestep_shift, 
        ),
    )


personal_configs_part_s2: Dict[str, TrainerConfig] = {}

# 3dmaster
personal_configs_part_s2["3dmaster_part_s2"] = config(
    use_ema=True,
    batch_size=1,
    lr=5e-5,#5e-6, 
    max_grad_norm=0.1,
    timestep_shift=3.0,
    transformer_ckpt_path=resume_ckpt_path,
    dataloader_num_processes=2, # Noted
    step_per_save = 1000,
    train_dataset_csv_path = train_dataset_csv_path,
    eval_dataset_csv_path = eval_dataset_csv_path,
    global_part = True, 
    slat_norm = True, 
)