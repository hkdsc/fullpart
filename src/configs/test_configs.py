import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from ..utils import NO_UPDATE, Updatable, eval_setup, parse_config_string
from .base_config import PrintableConfig, dataclass
from .train_configs import train_configs


@dataclass
class TestConfig(PrintableConfig):
    """Configuration for Test instantiation
    """

    train_configs = train_configs

    train_name: Updatable[str] = NO_UPDATE
    yml_path: Updatable[str] = NO_UPDATE

    output_dir: Updatable[str] = "test"
    experiment_name: Updatable[str] = NO_UPDATE

    num_frames: Updatable[int] = NO_UPDATE
    sample_fps: Updatable[int] = NO_UPDATE
    train_len: Updatable[int] = NO_UPDATE
    height: Updatable[int] = NO_UPDATE
    width: Updatable[int] = NO_UPDATE
    crop_type: Updatable[str] = NO_UPDATE

    ckpt_path: Updatable[str] = NO_UPDATE
    unet_ckpt_path: Updatable[str] = NO_UPDATE
    transformer_ckpt_path: Updatable[str] = NO_UPDATE
    use_flash_attn: Updatable[bool] = NO_UPDATE
    diffusers_vae_ckpt_path: Updatable[str] = NO_UPDATE
    temporal_vae_ckpt_path: Updatable[str] = NO_UPDATE
    mm_ckpt_path: Updatable[str] = NO_UPDATE

    target: Updatable[str] = NO_UPDATE
    endfix_prompt: Updatable[str] = NO_UPDATE
    negative_prompt: Updatable[str] = NO_UPDATE
    seed: Updatable[int] = NO_UPDATE
    call: Updatable[str] = NO_UPDATE  # "key1=value1,key2=value2"

    data_mode: Literal["video", "image"] = "video"
    csv_path: Updatable[str] = NO_UPDATE
    index_column: Updatable[str] = NO_UPDATE
    video_path_column: Updatable[str] = NO_UPDATE
    image_path_column: Updatable[str] = NO_UPDATE
    vae_latent_column: Updatable[str] = NO_UPDATE
    caption_column: Updatable[str] = NO_UPDATE
    sample_type: Updatable[Literal["fix_fps", "fix_stride", "full_video", "random"]] = NO_UPDATE
    batch_size: Updatable[int] = NO_UPDATE
    num_samples: Updatable[int] = -1
    control_columns: Updatable[List[str]] = NO_UPDATE
    training_precision: str = NO_UPDATE

    motion_strength: Optional[float] = None
    """a user level motion_strength config, only for I2V"""

    def update_config(self):
        assert self.train_name != NO_UPDATE or self.yml_path != NO_UPDATE
        if self.yml_path != NO_UPDATE and os.path.exists(self.yml_path):
            config = eval_setup(self.yml_path)
        else:
            config = train_configs[self.train_name]

        config.pipeline.update(
            target=self.target,
            endfix_prompt=self.endfix_prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
        )  # TODO: update call, some settings are not in pipeline.call

        call_dict = parse_config_string(self.call)
        if self.height is not NO_UPDATE:
            call_dict['height'] = self.height
        if self.width is not NO_UPDATE:
            call_dict['width'] = self.width

        if self.motion_strength is not None:
            assert self.motion_strength >= 0.0 and self.motion_strength <= 1.0, "motion_strength should be in [0.0, 1.0]"
            STRENGTH_MIN = 0.6
            STRENGTH_MAX = 1.0
            DOWNGRADE_SCALE = 0.0
            if self.motion_strength == 1.0:
                call_dict.pop("strength", None)
                call_dict.pop("downgrade_scale", None)
            else:
                call_dict["strength"] = STRENGTH_MIN + (STRENGTH_MAX - STRENGTH_MIN) * self.motion_strength
                call_dict["strength"] = round(call_dict["strength"], 1)
                call_dict["downgrade_scale"] = DOWNGRADE_SCALE
        config.pipeline.call.update(**call_dict)

        config.update(
            output_dir=self.output_dir,
            experiment_name=self.experiment_name,
            training_precision=self.training_precision,
        )
        config.pipeline.update(
            ckpt_path=self.ckpt_path,
            diffusers_vae_ckpt_path=self.diffusers_vae_ckpt_path,
            temporal_vae_ckpt_path=self.temporal_vae_ckpt_path,
        )
        if config.pipeline.unet_config is not None:
            config.pipeline.unet_config.update(
                num_frames=self.num_frames,
                train_len=self.train_len,
                unet_ckpt_path=self.unet_ckpt_path,
                mm_ckpt_path=self.mm_ckpt_path,
                zero_init=False,
            )
        if config.pipeline.transformer_config is not None:
            config.pipeline.transformer_config.update(
                num_frames=self.num_frames,
                transformer_ckpt_path=self.transformer_ckpt_path,
                use_flash_attn=self.use_flash_attn,
            )

        config.val_data.update(
            mode=self.data_mode,
            path=self.csv_path,
            video_path_column=self.video_path_column,
            image_path_column=self.image_path_column,
            caption_column=self.caption_column,
            vae_latent_column=self.vae_latent_column,
            index_column=self.index_column,
            num_frames=self.num_frames,
            sample_type=self.sample_type,
            sample_fps=self.sample_fps,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            height=self.height,
            width=self.width,
            crop_type=self.crop_type,
            control_columns=self.control_columns,
        )
        return config


#### base ####
test_configs = {"test": TestConfig()}