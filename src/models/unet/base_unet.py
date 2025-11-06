import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models import UNet2DConditionModel

from ...configs.base_config import InstantiateConfig
from ...utils import get_substatedict, load_model, load_state_dict, log_to_rank0, save_model_from_ds


@dataclass
class UNet2DConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: UNet2D)

    gradient_checkpointing: bool = True
    """Whether to enable gradient checkpointing."""

    unet_ckpt_path: Optional[str] = None
    """ckpt path for unet"""

    num_frames: int = 16
    """Number of frames"""

    down_block_configs: Optional[List[Any]] = None
    """Customized Configs for down blocks"""
    mid_block_config: Optional[Any] = None
    """Customized Configs for mid block"""
    up_block_configs: Optional[List[Any]] = None
    """Customized Configs for up blocks"""

    def from_pretrained(self, ckpt_path, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        if hasattr(self, "down_block_types"):
            kwargs["down_block_types"] = self.down_block_types
        if hasattr(self, "mid_block_type"):
            kwargs["mid_block_type"] = self.mid_block_type
        if hasattr(self, "up_block_types"):
            kwargs["up_block_types"] = self.up_block_types
        unet = self._target.from_pretrained(ckpt_path, subfolder="unet", low_cpu_mem_usage=False, device_map=None, unet_config=self, **kwargs)
        unet.load_ckpts()
        return unet

    def setup(self, **kwargs) -> Any:
        raise NotImplementedError


class UNet2DMixin:
    def load_ckpts(self):
        if self.unet_config.unet_ckpt_path is not None:
            self.load_ckpt(self.unet_config.unet_ckpt_path)

    def set_num_frames(self, num_frames):
        self.num_frames = num_frames

        self.mid_block.num_frames = num_frames
        for block in self.down_blocks:
            block.num_frames = num_frames
        for block in self.up_blocks:
            block.num_frames = num_frames

    def set_num_videos(self, num_videos):
        self.num_videos = num_videos

        self.mid_block.num_videos = num_videos
        for block in self.down_blocks:
            block.num_videos = num_videos
        for block in self.up_blocks:
            block.num_videos = num_videos

    def save_ckpt(self, ds_state_dict, ckpt_path):
        save_model_from_ds("unet", ds_state_dict, ckpt_path, trainable_only=True, rename_func=self.rename_from_m2v)

    def load_ckpt(self, ckpt_path):
        load_model(self, ckpt_path, rename_func=self.rename_to_m2v)

    @staticmethod
    def rename_to_m2v(state_dict):
        return state_dict

    @staticmethod
    def rename_from_m2v(state_dict):
        return state_dict

    def setup(self):
        self.set_num_frames(self.unet_config.num_frames)
        self.replace_attn_processor()

    def replace_attn_processor(self):
        pass


class UNet2D(UNet2DConditionModel, UNet2DMixin):
    unet_config: UNet2DConfig

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
        unet_config=None,
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )

        # Prepare the mid block parameters, copied from `__init__` in `UNet2DConditionModel`

        num_attention_heads = num_attention_heads or attention_head_dim
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)
        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        time_embed_dim = self.time_embedding.linear_1.out_features
        if class_embeddings_concat:
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, _ in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            # NOTE: here is different from the official implementation
            if unet_config.down_block_configs is not None and unet_config.down_block_configs[i] is not None:
                down_block = unet_config.down_block_configs[i].setup(
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    downsample_padding=downsample_padding,
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_type=attention_type,
                    resnet_skip_time_act=resnet_skip_time_act,
                    resnet_out_scale_factor=resnet_out_scale_factor,
                    cross_attention_norm=cross_attention_norm,
                    attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                    dropout=dropout,
                )
            else:
                down_block = self.down_blocks[i]
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # Mid block initialization, use our get function instead of the official one
        if unet_config.mid_block_config is not None:
            self.mid_block = unet_config.mid_block_config.setup(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                attention_type=attention_type,
            )

        # up, mostly copied from `__init__` in `UNet2DConditionModel`
        up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block)) if reverse_transformer_layers_per_block is None else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, _ in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            # NOTE: here is different from the official implementation
            if unet_config.up_block_configs is not None and unet_config.up_block_configs[i] is not None:
                up_block = unet_config.up_block_configs[i].setup(
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resolution_idx=i,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_type=attention_type,
                    resnet_skip_time_act=resnet_skip_time_act,
                    resnet_out_scale_factor=resnet_out_scale_factor,
                    cross_attention_norm=cross_attention_norm,
                    attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                    dropout=dropout,
                )
            else:
                up_block = self.up_blocks[i]
            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = up_blocks

        self.unet_config = unet_config
        self.setup()
