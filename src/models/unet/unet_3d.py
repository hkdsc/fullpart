from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import torch.nn as nn
from diffusers.configuration_utils import register_to_config
try:
    from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel
except:
    from diffusers.models.transformer_temporal import TransformerTemporalModel

from ..blocks import *
from .base_unet import UNet2D, UNet2DConfig


@dataclass
class UNet3DConfig(UNet2DConfig):
    _target: Type = field(default_factory=lambda: UNet3D)

    unet_ckpt_path: Optional[str] = None
    """ckpt path for unet"""

    down_block_configs: Optional[List[Any]] = field(
        default_factory=lambda: [
            VideoFusionCrossAttnDownBlock3DConfig(),
            VideoFusionCrossAttnDownBlock3DConfig(),
            VideoFusionCrossAttnDownBlock3DConfig(),
            VideoFusionDownBlock3DConfig(),
        ]
    )
    mid_block_config: Optional[Any] = VideoFusionUNetMidBlock3DCrossAttnConfig()
    up_block_configs: Optional[List[Any]] = field(
        default_factory=lambda: [
            VideoFusionUpBlock3DConfig(),
            VideoFusionCrossAttnUpBlock3DConfig(),
            VideoFusionCrossAttnUpBlock3DConfig(),
            VideoFusionCrossAttnUpBlock3DConfig(),
        ]
    )

    num_frames: int = 16
    """Number of frames for training"""


class ConvTransformerIn(nn.Module):
    def __init__(self, conv_in, attention_head_dim, in_channels, num_frames):
        super().__init__()
        self.conv_in = conv_in
        self.attention_head_dim = attention_head_dim
        self.transformer_in = TransformerTemporalModel(
            num_attention_heads=8,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            num_layers=1,
        )
        self.num_frames = num_frames

    def forward(self, x):
        x = self.conv_in(x)
        x = self.transformer_in(x, num_frames=self.num_frames, return_dict=False)[0]
        return x


class UNet3D(UNet2D):
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
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
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
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
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
            unet_config=unet_config,
        )

        self.conv_in = ConvTransformerIn(
            self.conv_in,
            attention_head_dim=attention_head_dim,
            in_channels=block_out_channels[0],
            num_frames=self.num_frames,
        )

    @staticmethod
    def rename_to_m2v(state_dict):
        rules = {
            "temp_attentions.": "motion_modules.",
            "conv_in.": "conv_in.conv_in.",
            "transformer_in.": "conv_in.transformer_in.",
        }
        new_state_dict = {}
        for key, value in state_dict.items():
            for name in rules:
                if name in key:
                    key = key.replace(name, rules[name])
            new_state_dict[key] = value
        return new_state_dict

    @staticmethod
    def rename_from_m2v(state_dict):
        rules = {
            "motion_modules.": "temp_attentions.",
            "conv_in.conv_in.": "conv_in.",
            "conv_in.transformer_in.": "transformer_in.",
        }
        new_state_dict = {}
        for key, value in state_dict.items():
            for name in rules:
                if name in key:
                    key = key.replace(name, rules[name])
            new_state_dict[key] = value
        return new_state_dict
