import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed_from_grid
from einops import rearrange, repeat

from ..utils import less_than_or_equal_to
from typing import Any, Tuple, Union

class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class RandomSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        _, seq_length, _ = x.shape
        if self.training:
            idx = torch.randperm(self.max_seq_length).to(device=x.device)
            idx = idx[:seq_length].sort().values
        else:
            stride = self.max_seq_length // seq_length
            idx = torch.arange(0, self.max_seq_length, stride)[:seq_length].to(device=x.device)
        x = x + self.pe[:, idx]
        return x


class RotaryEmbeddingFastFlexi(nn.Module):
    def __init__(self, embed_dim, patch_resolution, theta=10000.0):
        super(RotaryEmbeddingFastFlexi, self).__init__()

        if isinstance(patch_resolution, int):
            self.num_axis = 1
        elif isinstance(patch_resolution, tuple):
            if len(patch_resolution) == 2:
                self.num_axis = 2
            elif len(patch_resolution) == 3:
                self.num_axis = 3
            else:
                raise ValueError("patch_resolution tuple must be 2-dimensional or 3-dimensional.")
        else:
            raise TypeError("patch_resolution must be an integer or a tuple.")

        if embed_dim % self.num_axis != 0:
            assert embed_dim == 64, 'debugging mode only support 64 head dim'
            self.axis_embed_dim = [24, 24, 16]
            # raise ValueError("embed_dim must be divisible by self.num_axis for an integer input.")
        else:
            self.axis_embed_dim = embed_dim // self.num_axis # 72 // 3 = 24
        self.patch_resolution = patch_resolution
        self.theta = theta

        freqs_cos, freqs_sin = self.compute_position_embedding()
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def compute_position_embedding(self):
        if isinstance(self.axis_embed_dim, list):
            assert len(self.axis_embed_dim) == 3
            frequency_0 = self.theta ** (torch.arange(0, self.axis_embed_dim[0], 2).float() / self.axis_embed_dim[0])
            frequency_0 = 1.0 / frequency_0 # 12

            frequency_1 = self.theta ** (torch.arange(0, self.axis_embed_dim[1], 2).float() / self.axis_embed_dim[1])
            frequency_1 = 1.0 / frequency_1 # 12

            frequency_2 = self.theta ** (torch.arange(0, self.axis_embed_dim[2], 2).float() / self.axis_embed_dim[2])
            frequency_2 = 1.0 / frequency_2 # 12

            t, h, w = self.patch_resolution
            position_t = (torch.arange(t)[:, None].float() @ frequency_2[None, :]).repeat(1, 2)
            position_h = (torch.arange(h)[:, None].float() @ frequency_0[None, :]).repeat(1, 2)
            position_w = (torch.arange(w)[:, None].float() @ frequency_1[None, :]).repeat(1, 2)
            temperal = position_t[:, None, None, :].expand(t, h, w, self.axis_embed_dim[2])
            height = position_h[None, :, None, :].expand(t, h, w, self.axis_embed_dim[0])
            width = position_w[None, None, :, :].expand(t, h, w, self.axis_embed_dim[1])
            position = torch.cat((temperal, height, width), dim=-1) # [T, H, W, 72]
            freqs_cos = position.cos()
            freqs_sin = position.sin()
            return freqs_cos, freqs_sin
        else:
            frequency = self.theta ** (torch.arange(0, self.axis_embed_dim, 2).float() / self.axis_embed_dim)
            frequency = 1.0 / frequency # 12

            if self.num_axis == 1:
                t = self.patch_resolution
                position_t = (torch.arange(t)[:, None].float() @ frequency[None, :]).repeat(1, 2)
                temperal = position_t[:, :].expand(t, self.axis_embed_dim)
                position = temperal
            elif self.num_axis == 2:
                h, w = self.patch_resolution
                position_h = (torch.arange(h)[:, None].float() @ frequency[None, :]).repeat(1, 2)
                position_w = (torch.arange(w)[:, None].float() @ frequency[None, :]).repeat(1, 2)
                height = position_h[:, None, :].expand(h, w, self.axis_embed_dim)
                width = position_w[None, :, :].expand(h, w, self.axis_embed_dim)
                position = torch.cat((height, width), dim=-1)
            elif self.num_axis == 3:
                t, h, w = self.patch_resolution
                position_t = (torch.arange(t)[:, None].float() @ frequency[None, :]).repeat(1, 2)
                position_h = (torch.arange(h)[:, None].float() @ frequency[None, :]).repeat(1, 2)
                position_w = (torch.arange(w)[:, None].float() @ frequency[None, :]).repeat(1, 2)
                temperal = position_t[:, None, None, :].expand(t, h, w, self.axis_embed_dim)
                height = position_h[None, :, None, :].expand(t, h, w, self.axis_embed_dim)
                width = position_w[None, None, :, :].expand(t, h, w, self.axis_embed_dim)
                position = torch.cat((temperal, height, width), dim=-1) # [T, H, W, 72]

            freqs_cos = position.cos()
            freqs_sin = position.sin()
            return freqs_cos, freqs_sin
    
    @torch.no_grad()
    def compute_sparse_position_embedding(self, pos):
        if isinstance(self.axis_embed_dim, list):
            assert len(self.axis_embed_dim) == 3
            frequency_0 = self.theta ** (torch.arange(0, self.axis_embed_dim[0], 2).float() / self.axis_embed_dim[0])
            frequency_0 = 1.0 / frequency_0 # 12

            frequency_1 = self.theta ** (torch.arange(0, self.axis_embed_dim[1], 2).float() / self.axis_embed_dim[1])
            frequency_1 = 1.0 / frequency_1 # 12

            frequency_2 = self.theta ** (torch.arange(0, self.axis_embed_dim[2], 2).float() / self.axis_embed_dim[2])
            frequency_2 = 1.0 / frequency_2 # 12

            frequency_0 = frequency_0.to(pos.device)
            frequency_1 = frequency_1.to(pos.device)
            frequency_2 = frequency_2.to(pos.device)
            position_t = (pos[:, 0:1].float() @ frequency_2[None, :]).repeat(1, 2)
            position_h = (pos[:, 1:2].float() @ frequency_0[None, :]).repeat(1, 2)
            position_w = (pos[:, 2:3].float() @ frequency_1[None, :]).repeat(1, 2)
            temperal = position_t[:, :].expand(pos.shape[0], self.axis_embed_dim[2])
            height = position_h[:, :].expand(pos.shape[0], self.axis_embed_dim[0])
            width = position_w[:, :].expand(pos.shape[0], self.axis_embed_dim[1])
            position = torch.cat((temperal, height, width), dim=-1) # [N, 72]
            freqs_cos = position.cos()
            freqs_sin = position.sin()
            return freqs_cos, freqs_sin
        else:
            frequency = self.theta ** (torch.arange(0, self.axis_embed_dim, 2).float() / self.axis_embed_dim)
            frequency = 1.0 / frequency # 12

            frequency = frequency.to(pos.device)

            if self.num_axis == 1:
                position_t = (pos.float() @ frequency[None, :]).repeat(1, 2)
                temperal = position_t[:, :].expand(pos.shape[0], self.axis_embed_dim)
                position = temperal
            elif self.num_axis == 2:
                position_h = (pos[:, 0:1].float() @ frequency[None, :]).repeat(1, 2)
                position_w = (pos[:, 1:2].float() @ frequency[None, :]).repeat(1, 2)
                height = position_h[:, :].expand(pos.shape[0], self.axis_embed_dim)
                width = position_w[:, :].expand(pos.shape[0], self.axis_embed_dim)
                position = torch.cat((height, width), dim=-1)
            elif self.num_axis == 3:
                position_t = (pos[:, 0:1].float() @ frequency[None, :]).repeat(1, 2)
                position_h = (pos[:, 1:2].float() @ frequency[None, :]).repeat(1, 2)
                position_w = (pos[:, 2:3].float() @ frequency[None, :]).repeat(1, 2)
                temperal = position_t[:, :].expand(pos.shape[0], self.axis_embed_dim)
                height = position_h[:, :].expand(pos.shape[0], self.axis_embed_dim)
                width = position_w[:, :].expand(pos.shape[0], self.axis_embed_dim)
                position = torch.cat((temperal, height, width), dim=-1) # [N, 72]

            freqs_cos = position.cos()
            freqs_sin = position.sin()
            return freqs_cos, freqs_sin
    
    def get_sparse_rope(self, x, pos):
        freqs_cos, freqs_sin = self.compute_sparse_position_embedding(pos)
        freqs_cos, freqs_sin = freqs_cos.to(x.dtype), freqs_sin.to(x.dtype)
        batch, num_heads, num_patches, embed_dim = x.shape
        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, embed_dim)

        freqs_cos = freqs_cos.reshape(-1, embed_dim)
        freqs_sin = freqs_sin.reshape(-1, embed_dim)

        return inputs * freqs_cos + x * freqs_sin
    

    # NOTE: 逐渐增加resolution会导致compile不停报警，需要跟AIP对一下这个问题
    @torch.compile
    def get_rope(self, x, freqs_cos, freqs_sin):
        batch, num_heads, num_patches, embed_dim = x.shape
        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, embed_dim)

        freqs_cos = freqs_cos.reshape(-1, embed_dim)
        freqs_sin = freqs_sin.reshape(-1, embed_dim)

        return inputs * freqs_cos + x * freqs_sin

    def forward(self, x, patch_resolution=None):
        # Check whether the patch resolution is the predefined size
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
        if patch_resolution is not None:
            if not less_than_or_equal_to(patch_resolution, self.patch_resolution):
                assert (isinstance(self.patch_resolution, int) and isinstance(patch_resolution, int)) or len(self.patch_resolution) == len(patch_resolution)
                if self.num_axis == 1:
                    self.patch_resolution = max(self.patch_resolution, patch_resolution)
                elif self.num_axis == 2:
                    self.patch_resolution = (max(self.patch_resolution[0], patch_resolution[0]), max(self.patch_resolution[1], patch_resolution[1]))
                elif self.num_axis == 3:
                    self.patch_resolution = (
                        max(self.patch_resolution[0], patch_resolution[0]),
                        max(self.patch_resolution[1], patch_resolution[1]),
                        max(self.patch_resolution[2], patch_resolution[2]),
                    )
                else:
                    raise NotImplementedError
                freqs_cos, freqs_sin = self.compute_position_embedding()
                self.register_buffer("freqs_cos", freqs_cos.to(device=x.device, dtype=x.dtype))
                self.register_buffer("freqs_sin", freqs_sin.to(device=x.device, dtype=x.dtype))
                freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
            if isinstance(patch_resolution, int):
                patch_resolution = [patch_resolution]
            for dim_idx, length in enumerate(patch_resolution):
                freqs_cos = torch.narrow(freqs_cos, dim_idx, 0, length)
                freqs_sin = torch.narrow(freqs_sin, dim_idx, 0, length)

        return self.get_rope(x, freqs_cos, freqs_sin)


class RotaryEmbeddingFast(nn.Module):
    def __init__(self, embed_dim, patch_resolution, theta=10000.0):
        super(RotaryEmbeddingFast, self).__init__()

        if isinstance(patch_resolution, int):
            self.num_axis = 1
        elif isinstance(patch_resolution, tuple):
            if len(patch_resolution) == 2:
                self.num_axis = 2
            elif len(patch_resolution) == 3:
                self.num_axis = 3
            else:
                raise ValueError("patch_resolution tuple must be 2-dimensional or 3-dimensional.")
        else:
            raise TypeError("patch_resolution must be an integer or a tuple.")

        if embed_dim % self.num_axis != 0:
            raise ValueError("embed_dim must be divisible by self.num_axis for an integer input.")
        self.axis_embed_dim = embed_dim // self.num_axis # 72 // 3 = 24
        self.patch_resolution = patch_resolution
        self.theta = theta

        freqs_cos, freqs_sin = self.compute_position_embedding()
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def compute_position_embedding(self):
        frequency = self.theta ** (torch.arange(0, self.axis_embed_dim, 2).float() / self.axis_embed_dim)
        frequency = 1.0 / frequency # 12

        if self.num_axis == 1:
            t = self.patch_resolution
            position_t = (torch.arange(t)[:, None].float() @ frequency[None, :]).repeat(1, 2)
            temperal = position_t[:, :].expand(t, self.axis_embed_dim)
            position = temperal
        elif self.num_axis == 2:
            h, w = self.patch_resolution
            position_h = (torch.arange(h)[:, None].float() @ frequency[None, :]).repeat(1, 2)
            position_w = (torch.arange(w)[:, None].float() @ frequency[None, :]).repeat(1, 2)
            height = position_h[:, None, :].expand(h, w, self.axis_embed_dim)
            width = position_w[None, :, :].expand(h, w, self.axis_embed_dim)
            position = torch.cat((height, width), dim=-1)
        elif self.num_axis == 3:
            t, h, w = self.patch_resolution
            position_t = (torch.arange(t)[:, None].float() @ frequency[None, :]).repeat(1, 2)
            position_h = (torch.arange(h)[:, None].float() @ frequency[None, :]).repeat(1, 2)
            position_w = (torch.arange(w)[:, None].float() @ frequency[None, :]).repeat(1, 2)
            temperal = position_t[:, None, None, :].expand(t, h, w, self.axis_embed_dim)
            height = position_h[None, :, None, :].expand(t, h, w, self.axis_embed_dim)
            width = position_w[None, None, :, :].expand(t, h, w, self.axis_embed_dim)
            position = torch.cat((temperal, height, width), dim=-1) # [T, H, W, 72]

        freqs_cos = position.cos()
        freqs_sin = position.sin()
        return freqs_cos, freqs_sin
    
    @torch.no_grad()
    def compute_sparse_position_embedding(self, pos):
        frequency = self.theta ** (torch.arange(0, self.axis_embed_dim, 2).float() / self.axis_embed_dim)
        frequency = 1.0 / frequency # 12

        frequency = frequency.to(pos.device)

        if self.num_axis == 1:
            position_t = (pos.float() @ frequency[None, :]).repeat(1, 2)
            temperal = position_t[:, :].expand(pos.shape[0], self.axis_embed_dim)
            position = temperal
        elif self.num_axis == 2:
            position_h = (pos[:, 0:1].float() @ frequency[None, :]).repeat(1, 2)
            position_w = (pos[:, 1:2].float() @ frequency[None, :]).repeat(1, 2)
            height = position_h[:, :].expand(pos.shape[0], self.axis_embed_dim)
            width = position_w[:, :].expand(pos.shape[0], self.axis_embed_dim)
            position = torch.cat((height, width), dim=-1)
        elif self.num_axis == 3:
            position_t = (pos[:, 0:1].float() @ frequency[None, :]).repeat(1, 2)
            position_h = (pos[:, 1:2].float() @ frequency[None, :]).repeat(1, 2)
            position_w = (pos[:, 2:3].float() @ frequency[None, :]).repeat(1, 2)
            temperal = position_t[:, :].expand(pos.shape[0], self.axis_embed_dim)
            height = position_h[:, :].expand(pos.shape[0], self.axis_embed_dim)
            width = position_w[:, :].expand(pos.shape[0], self.axis_embed_dim)
            position = torch.cat((temperal, height, width), dim=-1) # [N, 72]

        freqs_cos = position.cos()
        freqs_sin = position.sin()
        return freqs_cos, freqs_sin
    
    def get_sparse_rope(self, x, pos):
        freqs_cos, freqs_sin = self.compute_sparse_position_embedding(pos)
        freqs_cos, freqs_sin = freqs_cos.to(x.dtype), freqs_sin.to(x.dtype)
        batch, num_heads, num_patches, embed_dim = x.shape
        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, embed_dim)

        freqs_cos = freqs_cos.reshape(-1, embed_dim)
        freqs_sin = freqs_sin.reshape(-1, embed_dim)

        return inputs * freqs_cos + x * freqs_sin
    

    # NOTE: 逐渐增加resolution会导致compile不停报警，需要跟AIP对一下这个问题
    @torch.compile
    def get_rope(self, x, freqs_cos, freqs_sin):
        batch, num_heads, num_patches, embed_dim = x.shape
        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, embed_dim)

        freqs_cos = freqs_cos.reshape(-1, embed_dim)
        freqs_sin = freqs_sin.reshape(-1, embed_dim)

        return inputs * freqs_cos + x * freqs_sin

    def forward(self, x, patch_resolution=None):
        # Check whether the patch resolution is the predefined size
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
        if patch_resolution is not None:
            if not less_than_or_equal_to(patch_resolution, self.patch_resolution):
                assert (isinstance(self.patch_resolution, int) and isinstance(patch_resolution, int)) or len(self.patch_resolution) == len(patch_resolution)
                if self.num_axis == 1:
                    self.patch_resolution = max(self.patch_resolution, patch_resolution)
                elif self.num_axis == 2:
                    self.patch_resolution = (max(self.patch_resolution[0], patch_resolution[0]), max(self.patch_resolution[1], patch_resolution[1]))
                elif self.num_axis == 3:
                    self.patch_resolution = (
                        max(self.patch_resolution[0], patch_resolution[0]),
                        max(self.patch_resolution[1], patch_resolution[1]),
                        max(self.patch_resolution[2], patch_resolution[2]),
                    )
                else:
                    raise NotImplementedError
                freqs_cos, freqs_sin = self.compute_position_embedding()
                self.register_buffer("freqs_cos", freqs_cos.to(device=x.device, dtype=x.dtype))
                self.register_buffer("freqs_sin", freqs_sin.to(device=x.device, dtype=x.dtype))
                freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
            if isinstance(patch_resolution, int):
                patch_resolution = [patch_resolution]
            for dim_idx, length in enumerate(patch_resolution):
                freqs_cos = torch.narrow(freqs_cos, dim_idx, 0, length)
                freqs_sin = torch.narrow(freqs_sin, dim_idx, 0, length)

        return self.get_rope(x, freqs_cos, freqs_sin)


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16, base_num_frames=16):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert not isinstance(grid_size, int)
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size, grid_size)

    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[2], dtype=np.float32) / (grid_size[2] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[2]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # temporal
    grid_t = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_num_frames)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # concat
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, grid_size[1] * grid_size[2], axis=1)  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, grid_size[0], axis=0)  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed3D(nn.Module):
    """3D Video to Patch Embedding"""

    def __init__(
        self,
        num_frames=16,
        height=224,
        width=224,
        frame_patch_size=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        base_num_frames=1,
        interpolation_scale=1,
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=(frame_patch_size, patch_size, patch_size), stride=(frame_patch_size, patch_size, patch_size), bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.frame_patch_size = frame_patch_size
        self.patch_size = patch_size
        # See:RotaryEmbeddingFast2d
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.num_frames = num_frames // frame_patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_num_frames = base_num_frames // frame_patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim,
            [self.num_frames, self.height, self.width],
            base_size=self.base_size,
            base_num_frames=self.base_num_frames,
            interpolation_scale=self.interpolation_scale,
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

    def forward(self, latent):
        num_frames = latent.shape[-3] // self.frame_patch_size
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCTHW -> B(THW)C
        if self.layer_norm:
            latent = self.norm(latent)

        # Interpolate positional embeddings if needed.
        # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
        if self.num_frames != num_frames or self.height != height or self.width != width:
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                grid_size=(num_frames, height, width),
                base_size=self.base_size,
                base_num_frames=self.base_num_frames,
                interpolation_scale=self.interpolation_scale,
            )
            pos_embed = torch.from_numpy(pos_embed)
            pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
        else:
            pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)

class NoPEPatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size

    def forward(self, latent):
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        return latent.to(latent.dtype)

class NoPEPatchEmbedVoxelPlucker(nn.Module):
    """3D Voxel Plucker to Patch Embedding"""

    def __init__(
        self,
        in_channels=8,
        embed_dim=768,
        temporal_patch_size=4,
        spatial_patch_size=(2, 2, 1),
        voxel_pos_emb=None,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
    ):
        super().__init__()
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.temporal_patch_size = temporal_patch_size

        self.spatial_proj = nn.Conv3d(in_channels, embed_dim, kernel_size=spatial_patch_size, stride=spatial_patch_size, bias=bias)
        self.voxel_pos_emb = voxel_pos_emb

        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

    def forward(self, latent):
        # latent b, 8, 77, 16, 16, 16
        bs, ch, h, w, d = latent.shape
        latent = self.spatial_proj(latent)

        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        return latent.to(latent.dtype)


class NoPEPatchEmbedVoxel(nn.Module):
    """3D Voxel to Patch Embedding"""

    def __init__(
        self,
        in_channels=8,
        embed_dim=768,
        temporal_patch_size=4,
        spatial_patch_size=(2, 2, 1),
        voxel_pos_emb=None,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
    ):
        super().__init__()
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.temporal_patch_size = temporal_patch_size

        self.temporal_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, temporal_patch_size), stride=(1, temporal_patch_size), bias=bias)
        self.spatial_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=spatial_patch_size, stride=spatial_patch_size, bias=bias)
        self.voxel_pos_emb = voxel_pos_emb

        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

    def forward(self, latent):
        # latent b, 8, 77, 16, 16, 16
        bs, ch, num_frames, h, w, d = latent.shape
        # compress teomporal -> b 8 20 16 16 16
        latent = rearrange(latent, "b c f h w d -> (b h w d) c 1 f")
        # print("===before temporal padding===", latent.shape)
        pad_values = latent[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, 3)
        latent = torch.cat([pad_values, latent], dim=-1)
        # latent = F.pad(latent, temporal_padding, mode='replicate')
        # print("===after temporal padding===", latent.shape)
        latent = self.temporal_proj(latent)
        # print("===after temporal proj===", latent.shape)

        # compress spatial -> b 8 20 8 8 16
        num_frames_down = latent.shape[-1]
        latent = rearrange(latent, "(b h w d) c 1 f -> (b f) c h w d", b=bs, h=h, w=w, d=d)
        # print("===before spatial proj===", latent.shape)
        latent = self.spatial_proj(latent)
        # print("===after spatial proj===", latent.shape)
        latent = rearrange(latent, "(b f) c h w d -> b c f h w d", f=num_frames_down)
        _, _, _, h_down, w_down, d_down = latent.shape
        # print("===latent after compressing===", latent.shape)
        # add positional encoding
        latent = latent + rearrange(self.voxel_pos_emb.to(latent.device).to(latent.dtype), "(h w d) c -> 1 c 1 h w d", h=h_down, w=w_down, d=d_down)
        # print("===latent after adding postional encoding===", latent.shape)

        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        return latent.to(latent.dtype)



class CombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim,
        use_text_condition: bool = False,
        use_resolution_condition: bool = False,
        use_aspect_ratio_condition: bool = False,
        use_frames_condition: bool = False,
        use_fps_condition: bool = False,
        split_conditions: bool = False,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_resolution_condition or use_aspect_ratio_condition or use_frames_condition or use_fps_condition
        self.use_text_condition = use_text_condition
        self.use_resolution_condition = use_resolution_condition
        self.use_aspect_ratio_condition = use_aspect_ratio_condition
        self.use_frames_condition = use_frames_condition
        self.use_fps_condition = use_fps_condition
        self.split_conditions = split_conditions

        if self.use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        if use_text_condition:
            self.text_embedder = TimestepEmbedding(in_channels=2048, time_embed_dim=embedding_dim)
        if use_resolution_condition:
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim // 2)
        if use_aspect_ratio_condition:
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        if use_frames_condition:
            self.frames_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        if use_fps_condition:
            self.fps_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        # zero init
        for name, module in self.named_children():
            if isinstance(module, TimestepEmbedding) and not name.endswith("timestep_embedder"):
                module.linear_2.weight.data.zero_()
                module.linear_2.bias.data.zero_()

    def forward(self, timestep, batch_size, hidden_dtype, fps=None, frames=None, resolution=None, aspect_ratio=None, prompt_embeds_pooled=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        conditioning = timesteps_emb
        spatial_conditioning = 0
        temporal_conditioning = 0

        if self.use_text_condition and prompt_embeds_pooled is not None:
            text_emb = self.text_embedder(prompt_embeds_pooled)
            conditioning = conditioning + text_emb
        if self.use_resolution_condition and resolution is not None:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            spatial_conditioning = spatial_conditioning + resolution_emb
        if self.use_aspect_ratio_condition and aspect_ratio is not None:
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            spatial_conditioning = spatial_conditioning + aspect_ratio_emb
        if self.use_frames_condition and frames is not None:
            frames_emb = self.additional_condition_proj(frames.flatten()).to(hidden_dtype)
            frames_emb = self.frames_embedder(frames_emb).reshape(batch_size, -1)
            temporal_conditioning = temporal_conditioning + frames_emb
        if self.use_fps_condition and fps is not None:
            fps_emb = self.additional_condition_proj(fps.flatten()).to(hidden_dtype)
            fps_emb = self.fps_embedder(fps_emb).reshape(batch_size, -1)
            temporal_conditioning = temporal_conditioning + fps_emb

        if self.split_conditions:
            return conditioning, spatial_conditioning, temporal_conditioning
        else:
            return conditioning + spatial_conditioning + temporal_conditioning


def unpadding_mask_args(attention_mask):
    assert attention_mask.ndim == 2
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    unpadding_args_dict = {}
    unpadding_args_dict["seqlens_in_batch"] = seqlens_in_batch
    unpadding_args_dict["indices"] = indices
    unpadding_args_dict["cu_seqlens"] = cu_seqlens
    unpadding_args_dict["max_seqlen_in_batch"] = max_seqlen_in_batch
    return unpadding_args_dict


@torch.no_grad()
def prepare_mask(mask, num_frames, patch_size, token_merge_size=None, mask_type=None):
    if mask is None:
        return None, None

    if mask_type != "cross":
        mask = F.avg_pool3d(mask, kernel_size=patch_size, stride=patch_size)
        assert torch.all((mask == 0) | (mask == 1)), "mask is not binary"

    if token_merge_size is not None:
        mask = F.avg_pool3d(mask, kernel_size=token_merge_size, stride=token_merge_size)
        assert torch.all((mask == 0) | (mask == 1)), "mask is not binary"

    if mask_type == "cross":
        mask = repeat(mask, "b l -> (b f) 1 l", f=num_frames).contiguous()  # num_frames
    elif mask_type == "spatial":
        mask = rearrange(mask, "b c f h w -> (b f) c (h w)")
    elif mask_type == "temporal":
        mask = rearrange(mask, "b c f h w -> (b h w) c f")
    elif mask_type == "3d":
        mask = rearrange(mask, "b c f h w -> b c (f h w)")
    else:
        raise ValueError(f"mask_type={mask_type}, not in (cross, spatial, temporal, 3d)")

    assert mask.ndim == 3 and mask.shape[1] == 1, f"mask.shape = {mask.shape}"
    unpadding_args = unpadding_mask_args(mask[:, 0])
    mask = (1 - mask) * -10000.0
    return mask, unpadding_args
