from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PatchEmbed
try:
    from diffusers.models.transformers.transformer_2d import Transformer2DModel, Transformer2DModelOutput
except:
    from diffusers.models.transformer_2d import Transformer2DModel, Transformer2DModelOutput
from einops import rearrange, repeat
from jaxtyping import Float, Int64

from ...configs.base_config import InstantiateConfig
from ...utils import load_model, log_to_rank0
from ..attention_processor import MaskedAttnProcessor2_0
from ..embeddings import NoPEPatchEmbed, RotaryEmbeddingFast


@dataclass
class Transformer2DModelConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Transformer2DModel)

    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing."""
    transformer_ckpt_path: Optional[Union[List[str], str]] = None
    """Path to the transformer checkpoint."""
    height: int = 512
    """Image height."""
    width: int = 512
    """Image width."""
    vae_scale_factor: int = 8
    """Scale factor for the VAE."""
    patch_size: int = 2
    """Patch size for the input image."""
    use_2d_rope: bool = False
    """Whether to use 2D RoPE."""
    theta_2d_rope: float = 10000.0
    """Theta for 2D RoPE."""
    from_scratch: bool = False
    """Whether to initialize the model from scratch."""
    use_flash_attn: bool = False
    """Whether to use flash attn."""
    in_channels: Optional[int] = None
    """Number of input channels."""
    out_channels: Optional[int] = None
    """Number of output channels."""
    qk_norm: bool = False
    """Whether to use qk norm."""

    def from_pretrained(self, ckpt_path, **kwargs):
        if self.in_channels is not None:
            kwargs["in_channels"] = self.in_channels
        if self.out_channels is not None:
            kwargs["out_channels"] = self.out_channels

        if self.from_scratch:
            log_to_rank0("Initializing the model from scratch.")
            config = self._target.load_config(ckpt_path, subfolder="transformer")
            transformer = self._target.from_config(config, low_cpu_mem_usage=False, device_map=None, transformer_config=self, **kwargs)
        else:
            transformer = self._target.from_pretrained(
                ckpt_path, subfolder="transformer", ignore_mismatched_sizes=True, low_cpu_mem_usage=False, device_map=None, transformer_config=self, **kwargs
            )

        transformer.set_selfattn_processor(use_flash_attn=self.use_flash_attn, use_rope=self.use_2d_rope, qk_norm=self.qk_norm)
        transformer.set_crossattn_processor(use_flash_attn=self.use_flash_attn, qk_norm=self.qk_norm)

        if isinstance(self.transformer_ckpt_path, str):
            transformer = load_model(transformer, self.transformer_ckpt_path, rename_func=None)
        elif isinstance(self.transformer_ckpt_path, list):
            for transformer_ckpt_path in self.transformer_ckpt_path:
                transformer = load_model(transformer, transformer_ckpt_path, rename_func=None)

        return transformer


class Transformer2DModel(Transformer2DModel):
    _supports_gradient_checkpointing = True

    @register_to_config
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
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        transformer_config: Transformer2DModelConfig = None,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size=sample_size,
            num_vector_embeds=num_vector_embeds,
            patch_size=patch_size,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            attention_type=attention_type,
            caption_channels=caption_channels,
        )

        inner_dim = num_attention_heads * attention_head_dim
        patch_size = transformer_config.patch_size
        self.gradient_checkpointing = transformer_config.gradient_checkpointing
        self.transformer_config = transformer_config

        if self.is_input_patches:
            self.patch_size = patch_size
            interpolation_scale = self.config.sample_size // 64  # => 64 (= 512 pixart) has interpolation scale 1
            interpolation_scale = max(interpolation_scale, 1)
            if transformer_config.use_2d_rope:
                patch_embed_cls = NoPEPatchEmbed
            else:
                patch_embed_cls = PatchEmbed
            self.pos_embed = patch_embed_cls(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )

        if self.is_input_patches and norm_type == "ada_norm_single":
            self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

    def set_selfattn_processor(self, use_rope=False, use_flash_attn=False, qk_norm=False):
        log_to_rank0(f"set spatial attention processor for {type(self)}, use_rope={use_rope}, qk_norm={qk_norm}, use_flash_attn={use_flash_attn}")

        if use_rope:
            rope = RotaryEmbeddingFast(
                embed_dim=self.attention_head_dim,
                patch_resolution=(
                    self.transformer_config.height // self.transformer_config.vae_scale_factor // self.patch_size,
                    self.transformer_config.width // self.transformer_config.vae_scale_factor // self.patch_size,
                ),
                theta=self.transformer_config.theta_2d_rope,
            )
        else:
            rope = None

        for name, module in self.transformer_blocks.named_modules():
            if isinstance(module, Attention) and name.endswith("attn1"):
                processor = MaskedAttnProcessor2_0(
                    use_flash_attn=use_flash_attn,
                    rope=rope,
                    qk_norm=qk_norm,
                    embed_dim=self.attention_head_dim,
                )
                module.set_processor(processor)

    def set_crossattn_processor(self, use_flash_attn=False, qk_norm=False):
        log_to_rank0(f"set cross attention processor for {type(self)}, use_rope={False}, qk_norm={qk_norm}, use_flash_attn={use_flash_attn}")
        for name, module in self.transformer_blocks.named_modules():
            if isinstance(module, Attention) and name.endswith("attn2"):
                processor = MaskedAttnProcessor2_0(
                    use_flash_attn=use_flash_attn,
                    rope=None,
                    qk_norm=qk_norm,
                    embed_dim=self.attention_head_dim,
                )
                module.set_processor(processor)

    def _get_varlen_flashattn_args(self, hidden_states, num_frames, attention_mask, cross_attention_kwargs=None):
        attention_mask = attention_mask.to(hidden_states.dtype)
        attention_mask = repeat(attention_mask, "b l -> (b f) l", f=num_frames).contiguous()
        seqlens_in_batch_kv = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices_kv = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        cu_seqlens_kv = F.pad(torch.cumsum(seqlens_in_batch_kv, dim=0, dtype=torch.torch.int32), (1, 0))
        max_seqlen_in_batch_kv = seqlens_in_batch_kv.max().item()

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        cross_attention_kwargs["seqlens_in_batch"] = seqlens_in_batch_kv
        cross_attention_kwargs["indices"] = indices_kv
        cross_attention_kwargs["max_seqlen_in_batch"] = max_seqlen_in_batch_kv
        cross_attention_kwargs["cu_seqlens"] = cu_seqlens_kv

        return cross_attention_kwargs

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "b c f h w"],
        encoder_hidden_states: Optional[Float[torch.Tensor, "b l d"]] = None,
        timestep: Optional[Int64[torch.Tensor, "b"]] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        num_frames = hidden_states.shape[-3]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            if self.transformer_config.use_flash_attn:
                cross_attention_kwargs = self._get_varlen_flashattn_args(hidden_states, num_frames, encoder_attention_mask, cross_attention_kwargs)

        output = super().forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            added_cond_kwargs,
            class_labels,
            cross_attention_kwargs,
            attention_mask,
            encoder_attention_mask,
            return_dict,
        )

        reshape_to_5d = lambda x: rearrange(x, "(b f) c h w -> b c f h w", f=num_frames)
        if not return_dict:
            return (reshape_to_5d(output[0]),)
        return Transformer2DModelOutput(reshape_to_5d(output.sample))
