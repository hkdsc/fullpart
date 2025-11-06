import sys
import copy
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import torch.utils.checkpoint as checkpoint
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput, DecoderOutput
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.nn import Sequential
import cv2
import numpy as np

from ...configs.base_config import InstantiateConfig
from .components import *
from ...utils import load_model, log_to_rank0

import torch.nn.functional as F
import kornia as K

SPLIT_SIZE = 16
torch_conv3d = F.conv3d


def split_conv(input, weight, *args, **kwargs):
    out_channels, in_channels_over_groups, kT, kH, kW = weight.shape
    element_num = in_channels_over_groups * input.shape[2] * input.shape[3] * input.shape[4]
    if element_num < (1 << 31) and out_channels != 1024:
        return torch_conv3d(input, weight, *args, **kwargs)
    else:
        output = None
        if out_channels != 1024 and out_channels != 3 and in_channels_over_groups != 256:
            split_inputs = torch.chunk(input, 32, dim=1)
            split_conv_weight = torch.chunk(weight, 32, dim=1)
        elif in_channels_over_groups == 256:
            split_inputs = torch.chunk(input, 64, dim=1)
            split_conv_weight = torch.chunk(weight, 64, dim=1)
        else:
            split_inputs = torch.chunk(input, SPLIT_SIZE, dim=1)
            split_conv_weight = torch.chunk(weight, SPLIT_SIZE, dim=1)
        for i in range(len(split_inputs)):
            if i == 0:
                output = torch_conv3d(split_inputs[i], split_conv_weight[i], *args, **kwargs)
                #  since bias only needs to added once, we set it to None after i==0
                args = list(args)
                args[0] = None
            else:
                output += torch_conv3d(split_inputs[i], split_conv_weight[i], *args, **kwargs)
        return output


F.conv3d = split_conv


class Encoder(nn.Module):
    def __init__(self, channels, init_dim, num_groups, num_channels_out, output_conv_kernel_size, stride, pad_mode="constant"):
        super().__init__()
        # encoder layers
        self.conv_in = CausalConv3d(channels, init_dim, output_conv_kernel_size, pad_mode=pad_mode)
        self.encoder_block1 = Sequential(*[ResBlockX(init_dim, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(4)])

        self.downblock_space = CausalConv3d(init_dim, init_dim, output_conv_kernel_size, stride=(1, stride, stride), pad_mode=pad_mode)
        self.encoder_block2 = ResBlockX2Y(init_dim, init_dim * 2, num_groups, output_conv_kernel_size, pad_mode=pad_mode)
        self.encoder_block3 = Sequential(*[ResBlockX(init_dim * 2, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(3)])

        self.downblock_spacetime_1 = CausalConv3d(init_dim * 2, init_dim * 2, output_conv_kernel_size, stride=(stride, stride, stride), pad_mode=pad_mode)
        self.encoder_block4 = Sequential(*[ResBlockX(init_dim * 2, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(4)])

        self.downblock_spacetime_2 = CausalConv3d(init_dim * 2, init_dim * 2, output_conv_kernel_size, stride=(stride, stride, stride), pad_mode=pad_mode)
        self.encoder_block5 = ResBlockX2Y(init_dim * 2, init_dim * 4, num_groups, output_conv_kernel_size, pad_mode=pad_mode)
        self.encoder_block6 = Sequential(*[ResBlockX(init_dim * 4, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(3)])
        self.encoder_block7 = Sequential(*[ResBlockX(init_dim * 4, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(4)])

        self.conv_in_last = GroupNormSwishConv3d_in(num_groups, init_dim * 4, num_channels_out)
        self.use_grad_checkpointing = False

    def forward(self, video: Tensor):
        # initial conv
        video = self.conv_in(video)

        # encoder
        if self.use_grad_checkpointing:
            for each_block in self.encoder_block1:
                video = checkpoint.checkpoint(each_block, video, use_reentrant=False)
            video = self.downblock_space(video)
            video = checkpoint.checkpoint(self.encoder_block2, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.encoder_block3, video, use_reentrant=False)
            video = self.downblock_spacetime_1(video)
            video = checkpoint.checkpoint(self.encoder_block4, video, use_reentrant=False)
            video = self.downblock_spacetime_2(video)
            video = checkpoint.checkpoint(self.encoder_block5, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.encoder_block6, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.encoder_block7, video, use_reentrant=False)
        else:
            video = self.encoder_block1(video)
            video = self.downblock_space(video)
            video = self.encoder_block2(video)
            video = self.encoder_block3(video)
            video = self.downblock_spacetime_1(video)
            video = self.encoder_block4(video)
            video = self.downblock_spacetime_2(video)
            video = self.encoder_block5(video)
            video = self.encoder_block6(video)
            video = self.encoder_block7(video)
        video = self.conv_in_last(video)

        return video


class SeparableEncoder(nn.Module):
    def __init__(self, channels, init_dim, num_groups, num_channels_out, output_conv_kernel_size, stride, pad_mode="constant"):
        super().__init__()
        # encoder layers
        self.conv_in = SeparableCausalConv3d(channels, init_dim, output_conv_kernel_size, pad_mode=pad_mode)
        self.encoder_block1 = Sequential(*[SeparableResBlockX(init_dim, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(4)])

        self.downblock_space = SeparableCausalConv3d(init_dim, init_dim, output_conv_kernel_size, stride=(1, stride, stride), pad_mode=pad_mode)
        self.encoder_block2 = SeparableResBlockX2Y(init_dim, init_dim * 2, num_groups, output_conv_kernel_size, pad_mode=pad_mode)
        self.encoder_block3 = Sequential(*[SeparableResBlockX(init_dim * 2, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(3)])

        self.downblock_spacetime_1 = SeparableCausalConv3d(
            init_dim * 2, init_dim * 2, output_conv_kernel_size, stride=(stride, stride, stride), pad_mode=pad_mode
        )
        self.encoder_block4 = Sequential(*[SeparableResBlockX(init_dim * 2, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(4)])

        self.downblock_spacetime_2 = SeparableCausalConv3d(
            init_dim * 2, init_dim * 2, output_conv_kernel_size, stride=(stride, stride, stride), pad_mode=pad_mode
        )
        self.encoder_block5 = SeparableResBlockX2Y(init_dim * 2, init_dim * 4, num_groups, output_conv_kernel_size, pad_mode=pad_mode)
        self.encoder_block6 = Sequential(*[SeparableResBlockX(init_dim * 4, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(3)])
        self.encoder_block7 = Sequential(*[SeparableResBlockX(init_dim * 4, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(4)])

        self.conv_in_last = GroupNormSwishConv3d_in(num_groups, init_dim * 4, num_channels_out)
        self.use_grad_checkpointing = False

    def forward(self, video: Tensor):
        # initial conv
        video = self.conv_in(video)

        # encoder
        if self.use_grad_checkpointing:
            for each_block in self.encoder_block1:
                video = checkpoint.checkpoint(each_block, video, use_reentrant=False)
            video = self.downblock_space(video)
            video = checkpoint.checkpoint(self.encoder_block2, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.encoder_block3, video, use_reentrant=False)
            video = self.downblock_spacetime_1(video)
            video = checkpoint.checkpoint(self.encoder_block4, video, use_reentrant=False)
            video = self.downblock_spacetime_2(video)
            video = checkpoint.checkpoint(self.encoder_block5, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.encoder_block6, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.encoder_block7, video, use_reentrant=False)
        else:
            video = self.encoder_block1(video)
            video = self.downblock_space(video)
            video = self.encoder_block2(video)
            video = self.encoder_block3(video)
            video = self.downblock_spacetime_1(video)
            video = self.encoder_block4(video)
            video = self.downblock_spacetime_2(video)
            video = self.encoder_block5(video)
            video = self.encoder_block6(video)
            video = self.encoder_block7(video)
        video = self.conv_in_last(video)

        return video


class Decoder(nn.Module):
    def __init__(self, channels, init_dim, num_groups, num_channels_out, output_conv_kernel_size, stride, pad_mode="constant"):
        super().__init__()
        # decoder layers
        self.conv_out = CausalConv3d(num_channels_out, init_dim * 4, output_conv_kernel_size, pad_mode=pad_mode)

        self.docoder_block1 = Sequential(*[ResBlockX(init_dim * 4, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(4)])
        self.groupnorm_1 = AdaptiveGroupNorm(num_groups, init_dim * 4, num_channels_out)
        self.docoder_block2 = Sequential(*[ResBlockX(init_dim * 4, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(4)])

        self.upblock_spacetime_1 = CausalConv3dTrans(
            init_dim * 4, depth_to_space_block=(stride, stride, stride), kernel_size=output_conv_kernel_size, pad_mode=pad_mode
        )
        self.groupnorm_2 = AdaptiveGroupNorm(num_groups, init_dim * 4, num_channels_out)
        self.docoder_block3 = ResBlockX2Y(init_dim * 4, init_dim * 2, num_groups, output_conv_kernel_size, pad_mode=pad_mode)
        self.docoder_block4 = Sequential(*[ResBlockX(init_dim * 2, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(3)])

        self.upblock_spacetime_2 = CausalConv3dTrans(
            init_dim * 2, depth_to_space_block=(stride, stride, stride), kernel_size=output_conv_kernel_size, pad_mode=pad_mode
        )
        self.groupnorm_3 = AdaptiveGroupNorm(num_groups, init_dim * 2, num_channels_out)
        self.docoder_block5 = Sequential(*[ResBlockX(init_dim * 2, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(4)])

        self.upblock_space = CausalConv3dTrans(init_dim * 2, depth_to_space_block=(1, stride, stride), kernel_size=output_conv_kernel_size, pad_mode=pad_mode)
        self.groupnorm_4 = AdaptiveGroupNorm(num_groups, init_dim * 2, num_channels_out)
        self.docoder_block6 = ResBlockX2Y(init_dim * 2, init_dim, num_groups, output_conv_kernel_size, pad_mode=pad_mode)
        self.docoder_block7 = Sequential(*[ResBlockX(init_dim, num_groups, output_conv_kernel_size, pad_mode=pad_mode) for _ in range(3)])

        self.conv_out_last = GroupNormSwishConv3d_out(num_groups, init_dim, channels, output_conv_kernel_size, pad_mode=pad_mode)
        self.use_grad_checkpointing = False

    def forward(self, x: Tensor):
        c = x
        video = self.conv_out(x)

        if self.use_grad_checkpointing:
            video = checkpoint.checkpoint(self.docoder_block1, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.groupnorm_1, video, c, use_reentrant=False)
            video = checkpoint.checkpoint(self.docoder_block2, video, use_reentrant=False)
            video = self.upblock_spacetime_1(video)
            video = checkpoint.checkpoint(self.groupnorm_2, video, c, use_reentrant=False)
            video = checkpoint.checkpoint(self.docoder_block3, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.docoder_block4, video, use_reentrant=False)
            video = self.upblock_spacetime_2(video)
            video = checkpoint.checkpoint(self.groupnorm_3, video, c, use_reentrant=False)
            video = checkpoint.checkpoint(self.docoder_block5, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.upblock_space, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.groupnorm_4, video, c, use_reentrant=False)
            video = checkpoint.checkpoint(self.docoder_block6, video, use_reentrant=False)
            for each_block in self.docoder_block7:
                video = checkpoint.checkpoint(each_block, video, use_reentrant=False)
            video = checkpoint.checkpoint(self.conv_out_last, video, use_reentrant=False)
        else:
            video = self.docoder_block1(video)
            video = self.groupnorm_1(video, c)
            video = self.docoder_block2(video)
            video = self.upblock_spacetime_1(video)
            video = self.groupnorm_2(video, c)
            video = self.docoder_block3(video)
            video = self.docoder_block4(video)
            video = self.upblock_spacetime_2(video)
            video = self.groupnorm_3(video, c)
            video = self.docoder_block5(video)
            video = self.upblock_space(video)
            video = self.groupnorm_4(video, c)
            video = self.docoder_block6(video)
            video = self.docoder_block7(video)
            video = self.conv_out_last(video)

        return video


def modify_and_expand_conv3d(state_dict):
    log_to_rank0("Modifying and expanding Conv3d weights in state_dict.")
    modified_state_dict = {}
    for key, weight in state_dict.items():
        if weight.dim() == 5 and weight.size(2) == 1 and weight.size(3) == 3 and weight.size(3) == 3:  # Conv3d权重
            out_channels, in_channels, _, H, W = weight.shape
            expanded_weights = torch.zeros(out_channels, in_channels, 3, H, W)
            expanded_weights[:, :, -1:, :, :] = weight
            modified_state_dict[key] = expanded_weights
        else:
            if key == "decoder.conv_out_last.conv.weight":
                log_to_rank0(f"Renaming {key} in state_dict.")
                key = "decoder.conv_out_last.conv.conv.weight"
            if key == "decoder.conv_out_last.conv.bias":
                log_to_rank0(f"Renaming {key} in state_dict.")
                key = "decoder.conv_out_last.conv.conv.bias"
            modified_state_dict[key] = weight
    return modified_state_dict


@dataclass
class VisualTokenizerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: VisualTokenizer)
    """Target class to instantiate."""
    channels: int = 3
    """Number of input channels."""
    encoder_init_dim: int = 128
    """Initial number of channels of the encoder"""
    decoder_init_dim: int = 128
    """Initial number of channels of the decoder"""
    num_groups: int = 32
    """Number of groups for group normalization."""
    num_channels_out: int = 8
    """Number of output channels."""
    output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3)
    """Output convolution kernel size. (1,3,3) for image and (3,3,3) for video."""
    block_out_channels: Tuple[int, int, int, int] = (128, 256, 512, 512)
    """It just for diffusers to calculate the vae scale factor. No actual use in the model."""
    scaling_factor: float = 1.0
    temporal_scale_factor: int = 4
    """Temporal scale factor for the diffusion model."""
    vae_pattern: str = "b c f h w"
    """VAE pattern for the diffusers."""
    stride: int = 2
    """Stride for the downsample layers."""
    pad_mode: Literal["constant", "reflect", "replicate"] = "constant"
    """Padding mode for the convolutions."""
    use_2plus1d: bool = False
    """Use 2+1d convolutions instead of 3d convolutions."""
    vae_ckpt_path: Optional[str] = None
    """Path to the VAE checkpoint."""
    vae_ckpt_path: Optional[str] = None
    """Path to the VAE checkpoint."""
    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing."""
    encoder_type: Literal["Standard Convolution", "Depthwise Separable Convolution"] = "Standard Convolution"
    """Type of the convolution, only supports for Standard Convolution and Depthwise Separable Convolution"""
    split_conv_3d: bool = False
    """Whether to split conv 3d for low GPU memory"""
    segment_size: int = sys.maxsize
    """Number of frames processed at one time. Long sequences are split into segments."""
    post_process: bool = True
    """Use post processing to eliminate grid issues in the video. From TaoXin Big Lao"""

    def from_pretrained(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        vae = VisualTokenizer(self)
        if self.vae_ckpt_path is not None:
            if self.use_2plus1d or self.output_conv_kernel_size == (1, 3, 3):
                rename_func = None
            else:
                rename_func = modify_and_expand_conv3d
            vae = load_model(vae, self.vae_ckpt_path, rename_func=rename_func)

        return vae


class VisualTokenizer(ModelMixin):
    def __init__(
        self,
        config: VisualTokenizerConfig,
    ):
        super(VisualTokenizer, self).__init__()
        self.config = config
        self.channels = config.channels
        self.encoder_init_dim = config.encoder_init_dim
        self.decoder_init_dim = config.decoder_init_dim
        self.num_groups = config.num_groups
        self.num_channels_out = config.num_channels_out
        self.output_conv_kernel_size = config.output_conv_kernel_size
        self.stride = config.stride
        self.pad_mode = config.pad_mode
        self.encoder_type = config.encoder_type
        self.split_conv_3d = config.split_conv_3d
        self.segment_size = config.segment_size
        self.post_process = config.post_process

        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)

        self.quant_conv = torch.nn.Conv3d(self.num_channels_out, 2 * self.num_channels_out, 1)
        self.post_quant_conv = torch.nn.Conv3d(self.num_channels_out, self.num_channels_out, 1)

        assert self.encoder_type in ["Standard Convolution", "Depthwise Separable Convolution"]

        if self.encoder_type == "Standard Convolution":
            self.encoder = Encoder(
                self.channels, self.encoder_init_dim, self.num_groups, self.num_channels_out, self.output_conv_kernel_size, self.stride, self.pad_mode
            )
        elif self.encoder_type == "Depthwise Separable Convolution":
            self.encoder = SeparableEncoder(
                self.channels, self.encoder_init_dim, self.num_groups, self.num_channels_out, self.output_conv_kernel_size, self.stride, self.pad_mode
            )

        self.decoder = Decoder(
            self.channels, self.decoder_init_dim, self.num_groups, self.num_channels_out, self.output_conv_kernel_size, self.stride, self.pad_mode
        )

        self.use_slicing = False
        if config.gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self):
        self.encoder.use_grad_checkpointing = True
        self.decoder.use_grad_checkpointing = True

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def encode(self, x, return_dict=True):
        x = x.to(self.logvar.dtype)

        batch_size = 1 if self.use_slicing else sys.maxsize
        encoded_slices = [
            torch.cat([self.encoder(segment_slice) for segment_slice in batch_slice.split(self.segment_size, dim=2)], dim=2)
            for batch_slice in x.split(batch_size)
        ]
        h = torch.cat(encoded_slices)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.split_conv_3d:
            F.conv3d = split_conv

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if self.split_conv_3d:
            F.conv3d = torch_conv3d

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(self, z, return_dict=True):
        z = z.to(self.logvar.dtype)
        batch_size = 1 if self.use_slicing else sys.maxsize
        latent_segment_size = (self.segment_size - 1) // self.config.temporal_scale_factor + 1
        decoded_slices = [
            torch.cat([self._decode(segment_slice).sample for segment_slice in batch_slice.split(latent_segment_size, dim=2)], dim=2)
            for batch_slice in z.split(batch_size)
        ]
        decoded = torch.cat(decoded_slices)

        if self.post_process:
            b, c, f, h, w = decoded.shape
            output_tensors = []
            decoded = decoded.mul(0.5).add(0.5).clamp(0, 1)
                
            for i in range(b):
                slice_tensor = decoded[i].permute(1, 0, 2, 3)
                processed_tensor = self.adaptive_mean_tensor(slice_tensor)
                output_tensors.append(processed_tensor.permute(1, 0, 2, 3))
            output_tensor = torch.stack(output_tensors, dim=0)
            decoded = output_tensor
            decoded = decoded.sub(0.5).div(0.5)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(self, input, sample_posterior=False, return_dict=True, generator=None):
        posterior = self.encode(input).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def adaptive_mean_tensor(self, input_tensor):
        T, C, H, W = input_tensor.shape
        yuv = K.color.rgb_to_yuv(input_tensor)
        y = yuv[:, 0:1, :, :]

        s = 4
        y_sub = F.pixel_unshuffle(y, s).reshape([T*s*s, 1, H//s, W//s])
        # rgb_sub = F.pixel_unshuffle(input_tensor.reshape([T*3, 1, H, W]), s).reshape([T, 3, s*s, H//s, W//s]).permute([0,2,1,3,4]).reshape([T*s*s, 3, H//s, W//s])
        y_bf = K.filters.joint_bilateral_blur(y, input_tensor, 5, 0.2, (15, 15), border_type='replicate')

        # y_sub_mean = K.filters.bilateral_blur(y_sub, 5, 0.3, (5, 5), border_type='replicate')
        y_sub_mean = K.filters.joint_bilateral_blur(y_sub, y_sub, 5, 0.3, (5, 5), border_type='replicate')

        y_base_mean = F.pixel_unshuffle(y_bf, s).reshape([T*s*s, 1, H//s, W//s])
        y_base_mean = K.filters.joint_bilateral_blur(y_base_mean, y_sub, 5, 0.3, (5, 5), border_type='replicate')
        y_sub_ratio = y_base_mean / (y_sub_mean + 0.0001)
        y_sub_ratio = K.filters.joint_bilateral_blur(y_sub_ratio, y_sub, 5, 0.3, (5, 5), border_type='replicate')
        
        y_sub_ada = y_sub * y_sub_ratio

        y_ada = F.pixel_shuffle(y_sub_ada.reshape([T, s*s, H//s, W//s]), s)
        yuv[:, :1, :, :] = y_ada

        output_tensor = K.color.yuv_to_rgb(yuv)


        return output_tensor
