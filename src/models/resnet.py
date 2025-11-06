from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import TemporalConvLayer
from einops import rearrange


class Downsample1D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = 1
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv1d(self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)
            # Nearest sampling
            with torch.no_grad():
                conv.weight.fill_(0)
                for i in range(self.out_channels):
                    conv.weight[i, i, 1] = 1
                conv.bias.fill_(0)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool1d(kernel_size=stride, stride=stride)
        self.conv1d = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int = 1,
    ) -> torch.FloatTensor:
        # (B F) C H W
        assert hidden_states.shape[1] == self.channels
        _, _, h, w = hidden_states.shape
        hidden_states = self.conv1d(rearrange(hidden_states, "(b f) c h w -> (b h w) c f", f=num_frames))
        hidden_states = rearrange(hidden_states, "(b h w) c f -> (b f) c h w", h=h, w=w)
        return hidden_states


class Upsample1D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose1d(channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv1d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
            # Nearest sampling
            with torch.no_grad():
                conv.weight.fill_(0)
                for i in range(self.out_channels):
                    conv.weight[i, i, 1] = 1
                conv.bias.fill_(0)

        self.conv1d = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        num_frames: int = 1,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        _, _, h, w = hidden_states.shape

        if self.use_conv_transpose:
            hidden_states = self.conv1d(rearrange(hidden_states, "(b f) c h w -> (b h w) c f", f=num_frames))
            hidden_states = rearrange(hidden_states, "(b h w) c f -> (b f) c h w", h=h, w=w)
            return hidden_states

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        hidden_states = rearrange(hidden_states, "(b f) c h w-> (b h w) c f", f=num_frames)
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if self.use_conv:
            hidden_states = self.conv1d(hidden_states)
            hidden_states = rearrange(hidden_states, "(b h w) c f -> (b f) c h w", h=h, w=w)

        return hidden_states


class VideoFusionTemporalConvLayer(TemporalConvLayer):
    def __init__(self, in_dim: int, out_dim: Optional[int] = None, dropout: float = 0.0, resnet_groups: int = 32, **kwargs):
        super().__init__(in_dim, out_dim, dropout=0.1, norm_num_groups=resnet_groups)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        num_frames: int = 16,
    ):
        return (super().forward(hidden_states, num_frames=num_frames),)


class LumiereTemporalConvLayer(nn.Module):
    # This is a dummy class to fit the diffusers unet blocks
    def __init__(self, in_dim: int, out_dim: Optional[int] = None, dropout: float = 0.0, resnet_groups: int = 32, **kwargs):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = out_dim
        self.out_dim = out_dim

        # conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(resnet_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(resnet_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_dim, out_dim, 1),
        )

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv3[-1].weight)
        nn.init.zeros_(self.conv3[-1].bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        num_frames: int = 16,
    ):
        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(rearrange(hidden_states, "(b f) c h w -> b c f h w", f=num_frames))
        hidden_states = self.conv3(rearrange(hidden_states, "b c f h w -> (b f) c h w"))

        hidden_states = identity + hidden_states
        return (hidden_states,)


class LumiereTemporalConvLayer_Warped(LumiereTemporalConvLayer):
    # This is a dummy class to fit the diffusers unet transformer blocks
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(in_dim=in_channels, out_dim=out_channels, dropout=dropout, resnet_groups=norm_num_groups, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.LongTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: torch.LongTensor = None,
        num_frames: int = 16,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        return super().forward(hidden_states, num_frames=num_frames)
