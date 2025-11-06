from typing import *
from cv2 import repeat
import einops
from einops import rearrange
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet

from torch.nn import init

class PixArtAlphaCondProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            raise NotImplementedError
            # self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        global_block: bool = False,
        mm_block: bool = False,
        mm_dim: Optional[int] = None,
        flat_block: bool = False,
        use_zero_linear: bool = False,
        abandon_img_cond: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )

        # abandon img cond
        self.abandon_img_cond = abandon_img_cond
        if self.abandon_img_cond:
            assert mm_block, "abandon img cond must use mm cond instead"
            self.mm_proj = nn.Linear(mm_dim, ctx_channels)
            self.mm_proj_norm = LayerNorm32(ctx_channels, elementwise_affine=True, eps=1e-6)
        
        self.global_block = global_block

        # multi-modality cross attn
        self.mm_block = mm_block
        self.use_zero_linear = use_zero_linear
        self.flat_block = flat_block
        if mm_block:
            assert mm_dim is not None
            self.norm_mm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
            # we need to norm the mm_context too
            self.norm_mm_cond = LayerNorm32(mm_dim, elementwise_affine=True, eps=1e-6)
            self.cross_attn_mm = MultiHeadAttention(
                channels,
                ctx_channels=mm_dim,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            # plucker proj
            self.camera_plucker_proj = PixArtAlphaCondProjection(mm_dim, mm_dim)
            self.voxel_plucker_proj = PixArtAlphaCondProjection(channels, channels)
            # zero linear
            if self.use_zero_linear:
                self.zero_linear = nn.Linear(channels, channels) 
                init.zeros_(self.zero_linear.weight)
                init.zeros_(self.zero_linear.bias)
        else:
            self.norm_mm = None
            self.cross_attn_mm = None

        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, mm_context: Optional[torch.Tensor]=None,
                 camera_plucker_embed: Optional[torch.Tensor]=None, voxel_plucker_embed: Optional[torch.Tensor]=None,
                 num_parts: Optional[int] = None):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        # flat block
        if self.global_block:
            # [b n c]
            real_batch_size = context.shape[0] // num_parts
            if self.training:
                assert num_parts == h.shape[0]
                assert real_batch_size == 1
            else:
                assert num_parts == h.shape[0] // 2
                assert real_batch_size == 2
            num_tokens = h.shape[1]
            h = rearrange(h, "(b np) n c -> b (np n) c", b=real_batch_size)
            
        h = self.self_attn(h)
        if self.global_block:
            h = rearrange(h, "b (np n) c -> (b np) n c", n=num_tokens)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        # replace img cond
        if self.abandon_img_cond:
            context = self.mm_proj(mm_context)
            context = self.mm_proj_norm(context)
        h = self.cross_attn(h, context)
        x = x + h

        # multi-modality cross attn
        if self.norm_mm is not None:
            # add plucker pos embed
            if camera_plucker_embed is not None:
                camera_plucker_embed = rearrange(camera_plucker_embed, "b f n c -> b (f n) c")
                camera_plucker_embed = self.camera_plucker_proj(camera_plucker_embed)
                mm_context = mm_context + camera_plucker_embed
            if voxel_plucker_embed is not None:
                voxel_plucker_embed = self.voxel_plucker_proj(voxel_plucker_embed)
                x = x + voxel_plucker_embed
            h = self.norm_mm(x)
            mm_context = self.norm_mm_cond(mm_context)
            h = self.cross_attn_mm(h, mm_context)
            # apply zero linear
            if self.use_zero_linear:
                h = self.zero_linear(h)
            x = x + h
        #

        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, mm_context: Optional[torch.Tensor]=None,
                camera_plucker_embed: Optional[torch.Tensor]=None, voxel_plucker_embed: Optional[torch.Tensor]=None,
                num_parts: Optional[int] = None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, mm_context, 
                                                     camera_plucker_embed, voxel_plucker_embed,
                                                     num_parts,
                                                     use_reentrant=False)
        else:
            return self._forward(x, mod, context, mm_context, camera_plucker_embed, voxel_plucker_embed, num_parts)
        