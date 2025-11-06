import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention import FeedForward, GatedSelfAttentionDense, _chunked_feed_forward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
try:
    from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel
except:
    from diffusers.models.transformer_temporal import TransformerTemporalModel
try:
    from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D
except:
    from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D
from diffusers.utils import is_torch_version
from diffusers.utils.torch_utils import apply_freeu, maybe_allow_in_graph
from einops import rearrange

from ..configs.base_config import InstantiateConfig
from ..configs.config_utils import to_immutable_dict
from .resnet import Downsample1D, Upsample1D, VideoFusionTemporalConvLayer


def prepare_forward_kwargs(block_cls, hidden_states, *tensors):
    get_num_frames = lambda x: x.shape[0] // block_cls.num_videos if block_cls.num_videos > 0 else block_cls.num_frames
    indexed_by_current_num_frames = lambda t, x: t.view(-1, block_cls.num_frames, *t.shape[1:])[:, :x, ...].reshape(-1, *t.shape[1:])
    result = ()
    for i in range(len(tensors)):
        result += (indexed_by_current_num_frames(tensors[i], get_num_frames(hidden_states)) if tensors[i] is not None else None,)
    return result


class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, hidden_states, *args, **kwargs) -> torch.Tensor:
        return (hidden_states,)


@dataclass
class Block3DConfig(InstantiateConfig):
    temp_conv_cls: Type = Identity
    motion_module_cls: Type = Identity
    num_frames: int = 16
    num_videos: int = -1  # Occasionally, the num_frames will change between blocks, so we need num_videos to know dynamic num_frames.
    num_temporal_layers: int = 1
    temporal_num_attention_heads: Optional[int] = 8
    temporal_cross_attention_dim: Optional[int] = None

    block_config: Dict[str, Any] = to_immutable_dict({})
    """Config for the diffusers block class"""

    def setup(self, **kwargs):
        kwargs.update(self.block_config)
        return self._target(**kwargs, config=self)


@dataclass
class CrossAttnDownBlock3DConfig(Block3DConfig):
    _target: Type = field(default_factory=lambda: TemplateCrossAttnDownBlock3D)

    add_downsample_1d: bool = False

    def setup(self, **kwargs):
        kwargs.update(self.block_config)
        return self._target(
            in_channels=kwargs.pop("in_channels"), out_channels=kwargs.pop("out_channels"), temb_channels=kwargs.pop("temb_channels"), config=self, **kwargs
        )


class TemplateCrossAttnDownBlock3D(CrossAttnDownBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        config: CrossAttnDownBlock3DConfig,
        **kwargs,
    ):
        init_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(CrossAttnDownBlock2D.__init__).parameters}
        super().__init__(in_channels, out_channels, temb_channels, **init_kwargs)

        self.config = config
        self.temp_conv_cls = config.temp_conv_cls
        self.motion_module_cls = config.motion_module_cls
        self.num_frames = config.num_frames
        self.num_videos = config.num_videos

        resnet_groups = kwargs.get("resnet_groups", 32)
        num_layers = kwargs.get("num_layers", 1)

        temp_convs = []
        motion_modules = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            kwargs["num_layers"] = 1
            temp_convs.append(self.temp_conv_cls(out_channels, out_channels, **kwargs))
            motion_modules.append(
                self.motion_module_cls(
                    config.temporal_num_attention_heads,
                    out_channels // config.temporal_num_attention_heads,
                    in_channels=out_channels,
                    num_layers=config.num_temporal_layers,
                    cross_attention_dim=config.temporal_cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
        self.temp_convs = nn.ModuleList(temp_convs)
        self.motion_modules = nn.ModuleList(motion_modules)
        if config.add_downsample_1d and self.downsamplers is not None:
            self.downsamplers1d = nn.ModuleList([Downsample1D(out_channels, use_conv=True, out_channels=out_channels, name="op")])
        else:
            self.downsamplers1d = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        additional_residuals: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()
        get_num_frames = lambda x: x.shape[0] // self.num_videos if self.num_videos > 0 else self.num_frames

        # NOTE: we assume no temporal down and up samplings in resnet, temp_conv, attn, temp_attn
        (
            temb,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            additional_residuals,
        ) = prepare_forward_kwargs(
            self,
            hidden_states,
            temb,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            additional_residuals,
        )

        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        blocks = list(zip(self.resnets, self.temp_convs, self.attentions, self.motion_modules))

        for i, (resnet, temp_conv, attn, temp_attn) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    lora_scale,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv),
                    hidden_states,
                    temb,
                    encoder_hidden_states,
                    attention_mask,
                    cross_attention_kwargs,
                    encoder_attention_mask,
                    get_num_frames(hidden_states),  # current_num_frames
                    **ckpt_kwargs,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_attn, return_dict=False),
                    hidden_states,
                    None,  # encoder_hidden_states
                    None,  # timestep
                    None,  # class_labels
                    get_num_frames(hidden_states),  # current_num_frames
                    cross_attention_kwargs,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = temp_conv(
                    hidden_states,
                    temb,
                    encoder_hidden_states,
                    attention_mask,
                    cross_attention_kwargs,
                    encoder_attention_mask,
                    get_num_frames(hidden_states),  # current_num_frames
                )[0]
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = temp_attn(
                    hidden_states,
                    num_frames=get_num_frames(hidden_states),  # current_num_frames
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=lora_scale)

        if self.downsamplers1d is not None:
            for downsampler1d in self.downsamplers1d:
                hidden_states = downsampler1d(hidden_states, num_frames=get_num_frames(hidden_states))

        if self.downsamplers is not None or self.downsamplers1d is not None:
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


@dataclass
class DownBlock3DConfig(CrossAttnDownBlock3DConfig):
    _target: Type = field(default_factory=lambda: TemplateDownBlock3D)


class TemplateDownBlock3D(DownBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        config: DownBlock3DConfig,
        **kwargs,
    ):
        init_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(DownBlock2D.__init__).parameters}
        super().__init__(in_channels, out_channels, temb_channels, **init_kwargs)

        self.config = config
        self.temp_conv_cls = config.temp_conv_cls
        self.motion_module_cls = config.motion_module_cls
        self.num_frames = config.num_frames
        self.num_videos = config.num_videos

        resnet_groups = kwargs.get("resnet_groups", 32)
        num_layers = kwargs.get("num_layers", 1)

        temp_convs = []
        motion_modules = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            kwargs["num_layers"] = 1
            temp_convs.append(self.temp_conv_cls(out_channels, out_channels, **kwargs))
            motion_modules.append(
                self.motion_module_cls(
                    config.temporal_num_attention_heads,
                    out_channels // config.temporal_num_attention_heads,
                    in_channels=out_channels,
                    num_layers=config.num_temporal_layers,
                    cross_attention_dim=config.temporal_cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
        self.temp_convs = nn.ModuleList(temp_convs)
        self.motion_modules = nn.ModuleList(motion_modules)
        if config.add_downsample_1d and self.downsamplers is not None:
            self.downsamplers1d = nn.ModuleList([Downsample1D(out_channels, use_conv=True, out_channels=out_channels, name="op")])
        else:
            self.downsamplers1d = None

    def forward(
        self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None, scale: float = 1.0
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()
        get_num_frames = lambda x: x.shape[0] // self.num_videos if self.num_videos > 0 else self.num_frames
        # NOTE: we assume no temporal down and up samplings in resnet, temp_conv, attn, temp_attn
        (temb,) = prepare_forward_kwargs(self, hidden_states, temb)
        for resnet, temp_conv, temp_attn in zip(self.resnets, self.temp_convs, self.motion_modules):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, scale, use_reentrant=False)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv),
                        hidden_states,
                        temb,
                        None,  # encoder_hidden_states
                        None,  # attention_mask
                        None,  # cross_attention_kwargs
                        None,  # encoder_attention_mask
                        get_num_frames(hidden_states),  # current_num_frames
                        use_reentrant=False,
                    )[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False),
                        hidden_states,
                        None,  # encoder_hidden_states
                        None,  # timestep
                        None,  # class_labels
                        get_num_frames(hidden_states),  # current_num_frames
                        None,  # cross_attention_kwargs
                        use_reentrant=False,
                    )[0]
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, scale)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv),
                        hidden_states,
                        temb,
                        None,  # encoder_hidden_states
                        None,  # attention_mask
                        None,  # cross_attention_kwargs
                        None,  # encoder_attention_mask
                        get_num_frames(hidden_states),  # current_num_frames
                    )[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False),
                        hidden_states,
                        None,  # encoder_hidden_states
                        None,  # timestep
                        None,  # class_labels
                        get_num_frames(hidden_states),  # current_num_frames
                        None,  # cross_attention_kwargs
                    )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)
                hidden_states = temp_conv(hidden_states, temb, num_frames=get_num_frames(hidden_states))[0]
                hidden_states = temp_attn(hidden_states, num_frames=get_num_frames(hidden_states), return_dict=False)[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=scale)

        if self.downsamplers1d is not None:
            for downsampler1d in self.downsamplers1d:
                hidden_states = downsampler1d(hidden_states, num_frames=get_num_frames(hidden_states))

        if self.downsamplers is not None or self.downsamplers1d is not None:
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


@dataclass
class UNetMidBlock3DCrossAttnConfig(Block3DConfig):
    _target: Type = field(default_factory=lambda: TemplateUNetMidBlock3DCrossAttn)

    def setup(self, **kwargs):
        kwargs.update(self.block_config)
        return self._target(in_channels=kwargs.pop("in_channels"), temb_channels=kwargs.pop("temb_channels"), config=self, **kwargs)


class TemplateUNetMidBlock3DCrossAttn(UNetMidBlock2DCrossAttn):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        config: UNetMidBlock3DCrossAttnConfig,
        **kwargs,
    ):
        init_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(UNetMidBlock2DCrossAttn.__init__).parameters}
        super().__init__(in_channels, temb_channels, **init_kwargs)

        self.config = config
        self.temp_conv_cls = config.temp_conv_cls
        self.motion_module_cls = config.motion_module_cls
        self.num_frames = config.num_frames
        self.num_videos = config.num_videos

        out_channels = in_channels
        resnet_groups = kwargs.get("resnet_groups", 32)
        num_layers = kwargs.get("num_layers", 1)

        temp_convs = [self.temp_conv_cls(in_channels, in_channels, **kwargs)]
        motion_modules = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            kwargs["num_layers"] = 1
            temp_convs.append(self.temp_conv_cls(in_channels, in_channels, **kwargs))
            motion_modules.append(
                self.motion_module_cls(
                    config.temporal_num_attention_heads,
                    out_channels // config.temporal_num_attention_heads,
                    in_channels=out_channels,
                    num_layers=config.num_temporal_layers,
                    cross_attention_dim=config.temporal_cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
        self.temp_convs = nn.ModuleList(temp_convs)
        self.motion_modules = nn.ModuleList(motion_modules)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        get_num_frames = lambda x: x.shape[0] // self.num_videos if self.num_videos > 0 else self.num_frames
        # NOTE: we assume no temporal down and up samplings in resnet, temp_conv, attn, temp_attn
        (
            temb,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        ) = prepare_forward_kwargs(
            self,
            hidden_states,
            temb,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.resnets[0]),
                hidden_states,
                temb,
                lora_scale,
                **ckpt_kwargs,
            )
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.temp_convs[0]),
                hidden_states,
                temb,
                encoder_hidden_states,
                attention_mask,
                cross_attention_kwargs,
                encoder_attention_mask,
                get_num_frames(hidden_states),  # current_num_frames
                **ckpt_kwargs,
            )[0]
        else:
            hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
            hidden_states = self.temp_convs[0](
                hidden_states,
                temb,
                encoder_hidden_states,
                attention_mask,
                cross_attention_kwargs,
                encoder_attention_mask,
                get_num_frames(hidden_states),  # current_num_frames
            )[0]

        for attn, temp_attn, resnet, temp_conv in zip(self.attentions, self.motion_modules, self.resnets[1:], self.temp_convs[1:]):
            if self.training and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_attn, return_dict=False),
                    hidden_states,
                    None,  # encoder_hidden_states
                    None,  # timestep
                    None,  # class_labels
                    get_num_frames(hidden_states),  # current_num_frames
                    cross_attention_kwargs,
                    **ckpt_kwargs,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    lora_scale,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv),
                    hidden_states,
                    temb,
                    encoder_hidden_states,
                    attention_mask,
                    cross_attention_kwargs,
                    encoder_attention_mask,
                    get_num_frames(hidden_states),  # current_num_frames
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = temp_attn(
                    hidden_states,
                    num_frames=get_num_frames(hidden_states),  # current_num_frames
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = temp_conv(
                    hidden_states,
                    temb,
                    encoder_hidden_states,
                    attention_mask,
                    cross_attention_kwargs,
                    encoder_attention_mask,
                    get_num_frames(hidden_states),  # current_num_frames
                )[0]

        return hidden_states


@dataclass
class CrossAttnUpBlock3DConfig(Block3DConfig):
    _target: Type = field(default_factory=lambda: TemplateCrossAttnUpBlock3D)

    add_upsample_1d: bool = False
    num_temporal_layers: int = 1

    def setup(self, **kwargs):
        kwargs.update(self.block_config)
        return self._target(
            in_channels=kwargs.pop("in_channels"),
            out_channels=kwargs.pop("out_channels"),
            prev_output_channel=kwargs.pop("prev_output_channel"),
            temb_channels=kwargs.pop("temb_channels"),
            config=self,
            **kwargs,
        )


class TemplateCrossAttnUpBlock3D(CrossAttnUpBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        config: CrossAttnUpBlock3DConfig,
        **kwargs,
    ):
        init_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(CrossAttnUpBlock2D.__init__).parameters}
        super().__init__(in_channels, out_channels, prev_output_channel, temb_channels, **init_kwargs)

        self.config = config
        self.temp_conv_cls = config.temp_conv_cls
        self.motion_module_cls = config.motion_module_cls
        self.num_frames = config.num_frames
        self.num_videos = config.num_videos

        resnet_groups = kwargs.get("resnet_groups", 32)
        num_layers = kwargs.get("num_layers", 1)

        temp_convs = []
        motion_modules = []
        for i in range(num_layers):
            kwargs["num_layers"] = 1
            temp_convs.append(self.temp_conv_cls(out_channels, out_channels, **kwargs))
            motion_modules.append(
                self.motion_module_cls(
                    config.temporal_num_attention_heads,
                    out_channels // config.temporal_num_attention_heads,
                    in_channels=out_channels,
                    num_layers=config.num_temporal_layers,
                    cross_attention_dim=config.temporal_cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
        self.temp_convs = nn.ModuleList(temp_convs)
        self.motion_modules = nn.ModuleList(motion_modules)
        if config.add_upsample_1d and self.upsamplers is not None:
            self.upsamplers1d = nn.ModuleList([Upsample1D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers1d = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        is_freeu_enabled = getattr(self, "s1", None) and getattr(self, "s2", None) and getattr(self, "b1", None) and getattr(self, "b2", None)
        get_num_frames = lambda x: x.shape[0] // self.num_videos if self.num_videos > 0 else self.num_frames

        # NOTE: we assume no temporal down and up samplings in resnet, temp_conv, attn, temp_attn
        (
            temb,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        ) = prepare_forward_kwargs(
            self,
            hidden_states,
            temb,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )
        for resnet, temp_conv, attn, temp_attn in zip(self.resnets, self.temp_convs, self.attentions, self.motion_modules):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    lora_scale,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv),
                    hidden_states,
                    temb,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    get_num_frames(hidden_states),  # current_num_frames
                    **ckpt_kwargs,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_attn, return_dict=False),
                    hidden_states,
                    None,  # encoder_hidden_states
                    None,  # timestep
                    None,  # class_labels
                    get_num_frames(hidden_states),  # current_num_frames
                    cross_attention_kwargs,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = temp_conv(
                    hidden_states,
                    temb,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    get_num_frames(hidden_states),  # current_num_frames
                )[0]
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = temp_attn(
                    hidden_states,
                    num_frames=get_num_frames(hidden_states),  # current_num_frames
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, scale=lora_scale)
        if self.upsamplers1d is not None:
            for upsampler1d in self.upsamplers1d:
                hidden_states = upsampler1d(hidden_states, num_frames=get_num_frames(hidden_states))
        return hidden_states


@dataclass
class UpBlock3DConfig(CrossAttnUpBlock3DConfig):
    _target: Type = field(default_factory=lambda: TemplateUpBlock3D)


class TemplateUpBlock3D(UpBlock2D):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        config: UpBlock3DConfig,
        **kwargs,
    ):
        init_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(UpBlock2D.__init__).parameters}
        super().__init__(in_channels, prev_output_channel, out_channels, temb_channels, **init_kwargs)

        self.config = config
        self.temp_conv_cls = config.temp_conv_cls
        self.motion_module_cls = config.motion_module_cls
        self.num_frames = config.num_frames
        self.num_videos = config.num_videos

        resnet_groups = kwargs.get("resnet_groups", 32)
        num_layers = kwargs.get("num_layers", 1)

        temp_convs = []
        motion_modules = []
        for _ in range(num_layers):
            kwargs["num_layers"] = 1
            temp_convs.append(self.temp_conv_cls(out_channels, out_channels, **kwargs))
            motion_modules.append(
                self.motion_module_cls(
                    config.temporal_num_attention_heads,
                    out_channels // config.temporal_num_attention_heads,
                    in_channels=out_channels,
                    num_layers=config.num_temporal_layers,
                    cross_attention_dim=config.temporal_cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
        self.temp_convs = nn.ModuleList(temp_convs)
        self.motion_modules = nn.ModuleList(motion_modules)
        if config.add_upsample_1d and self.upsamplers is not None:
            self.upsamplers1d = nn.ModuleList([Upsample1D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers1d = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        is_freeu_enabled = getattr(self, "s1", None) and getattr(self, "s2", None) and getattr(self, "b1", None) and getattr(self, "b2", None)
        get_num_frames = lambda x: x.shape[0] // self.num_videos if self.num_videos > 0 else self.num_frames

        (temb,) = prepare_forward_kwargs(self, hidden_states, temb)
        for resnet, temp_conv, temp_attn in zip(self.resnets, self.temp_convs, self.motion_modules):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, scale, use_reentrant=False)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv),
                        hidden_states,
                        temb,
                        None,  # encoder_hidden_states
                        None,  # cross_attention_kwargs
                        None,  # attention_mask
                        None,  # encoder_attention_mask
                        get_num_frames(hidden_states),  # current_num_frames
                        use_reentrant=False,
                    )[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False),
                        hidden_states,
                        None,  # encoder_hidden_states
                        None,  # timestep
                        None,  # class_labels
                        get_num_frames(hidden_states),  # current_num_frames
                        None,  # cross_attention_kwargs
                        use_reentrant=False,
                    )[0]
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, scale)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv),
                        hidden_states,
                        temb,
                        None,  # encoder_hidden_states
                        None,  # cross_attention_kwargs
                        None,  # attention_mask
                        None,  # encoder_attention_mask
                        get_num_frames(hidden_states),  # current_num_frames
                    )[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False),
                        hidden_states,
                        None,  # encoder_hidden_states
                        None,  # timestep
                        None,  # class_labels
                        get_num_frames(hidden_states),  # current_num_frames
                        None,  # cross_attention_kwargs
                    )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)
                hidden_states = temp_conv(hidden_states, temb, num_frames=get_num_frames(hidden_states))[0]
                hidden_states = temp_attn(hidden_states, num_frames=get_num_frames(hidden_states), return_dict=False)[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, scale=scale)
        if self.upsamplers1d is not None:
            for upsampler1d in self.upsamplers1d:
                hidden_states = upsampler1d(hidden_states, num_frames=get_num_frames(hidden_states))
        return hidden_states


# VideoFusion
class VideoFusionCrossAttnDownBlock3DConfig(CrossAttnDownBlock3DConfig):
    temp_conv_cls = VideoFusionTemporalConvLayer
    motion_module_cls = TransformerTemporalModel

    def setup(self, **kwargs):
        kwargs["num_attention_heads"] = (
            kwargs["out_channels"] // kwargs["num_attention_heads"]
        )  # NOTE: VideoFusion misused num_attention_heads here. We have to adjust.
        kwargs["use_linear_projection"] = True  # NOTE: VideoFusion set this to True and not included in config.json. Hard coding here.
        return super().setup(**kwargs)


class VideoFusionDownBlock3DConfig(DownBlock3DConfig):
    temp_conv_cls = VideoFusionTemporalConvLayer
    motion_module_cls = Identity

    def setup(self, **kwargs):
        kwargs["num_attention_heads"] = kwargs["out_channels"] // kwargs["num_attention_heads"]
        kwargs["use_linear_projection"] = True
        return super().setup(**kwargs)


class VideoFusionUNetMidBlock3DCrossAttnConfig(UNetMidBlock3DCrossAttnConfig):
    temp_conv_cls = VideoFusionTemporalConvLayer
    motion_module_cls = TransformerTemporalModel

    def setup(self, **kwargs):
        kwargs["num_attention_heads"] = kwargs["in_channels"] // kwargs["num_attention_heads"]
        kwargs["use_linear_projection"] = True
        return super().setup(**kwargs)


class VideoFusionCrossAttnUpBlock3DConfig(CrossAttnUpBlock3DConfig):
    temp_conv_cls = VideoFusionTemporalConvLayer
    motion_module_cls = TransformerTemporalModel

    def setup(self, **kwargs):
        kwargs["num_attention_heads"] = kwargs["out_channels"] // kwargs["num_attention_heads"]
        kwargs["use_linear_projection"] = True
        return super().setup(**kwargs)


class VideoFusionUpBlock3DConfig(UpBlock3DConfig):
    temp_conv_cls = VideoFusionTemporalConvLayer
    motion_module_cls = Identity

    def setup(self, **kwargs):
        kwargs["num_attention_heads"] = kwargs["out_channels"] // kwargs["num_attention_heads"]
        kwargs["use_linear_projection"] = True
        return super().setup(**kwargs)


class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features * self.t * self.h * self.w, out_features, bias=False)

    def forward(self, x):
        x = rearrange(x, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw e)", nt=self.t, nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features, out_features * self.t * self.h * self.w, bias=False)
        # self.proj.weight.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e", nt=self.t, nh=self.h, nw=self.w)


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features, out_features * self.t * self.h * self.w, bias=False)
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e", nt=self.t, nh=self.h, nw=self.w)
        return torch.lerp(skip, x, self.fac.to(x.dtype))
