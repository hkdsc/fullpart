from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version

from ..embeddings import PatchEmbed3D
from .transformer_2d import Transformer2DModel, Transformer2DModelConfig, Transformer2DModelOutput


@dataclass
class Transformer3DModelConfig(Transformer2DModelConfig):
    _target: Type = field(default_factory=lambda: Transformer3DModel)

    pe_init_num_frames: int = 16
    """init number of frames for the positional embeddings, (if input is not consistent with this, PE matrix will be recalculated)"""
    base_num_frames: int = 16
    """base temporal resolution for the positional embeddings"""
    frame_patch_size: int = 1
    """Temporal patch size"""


class Transformer3DModel(Transformer2DModel):
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
        transformer_config: Transformer3DModelConfig = None,
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
            transformer_config=transformer_config,
        )

        
        inner_dim = num_attention_heads * attention_head_dim
        base_num_frames = transformer_config.base_num_frames
        frame_patch_size = transformer_config.frame_patch_size
        patch_size = transformer_config.patch_size

        if self.is_input_patches:
            self.frame_patch_size = frame_patch_size
            self.patch_size = patch_size
            interpolation_scale = self.config.sample_size // 64  # => 64 (= 512 pixart) has interpolation scale 1
            interpolation_scale = max(interpolation_scale, 1)
            self.pos_embed = PatchEmbed3D(
                num_frames=transformer_config.pe_init_num_frames,
                height=sample_size,
                width=sample_size,
                frame_patch_size=frame_patch_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                base_num_frames=base_num_frames,
                interpolation_scale=interpolation_scale,
            )

        if self.is_input_patches and norm_type == "ada_norm_single":
            self.proj_out = nn.Linear(inner_dim, frame_patch_size * patch_size * patch_size * self.out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            if self.transformer_config.use_flash_attn:
                num_frames = 1  # NOTE: we dont have num_frames for 3d transformer
                cross_attention_kwargs = self._get_varlen_flashattn_args(hidden_states, num_frames, encoder_attention_mask, cross_attention_kwargs)
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states, scale=lora_scale) if not USE_PEFT_BACKEND else self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states, scale=lora_scale) if not USE_PEFT_BACKEND else self.proj_in(hidden_states)

        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            # NOTE, this is the first difference from the 2D transformer
            num_frames, height, width = (
                hidden_states.shape[-3] // self.frame_patch_size,
                hidden_states.shape[-2] // self.patch_size,
                hidden_states.shape[-1] // self.patch_size,
            )
            hidden_states = self.pos_embed(hidden_states)

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")
                batch_size = hidden_states.shape[0]
                timestep, embedded_timestep = self.adaln_single(timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype)

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        for block in self.transformer_blocks:
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
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.proj_out(hidden_states, scale=lora_scale) if not USE_PEFT_BACKEND else self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states, scale=lora_scale) if not USE_PEFT_BACKEND else self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.squeeze(1)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            # NOTE, this is the second difference from the 2D transformer
            hidden_states = hidden_states.reshape(
                shape=(-1, num_frames, height, width, self.frame_patch_size, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nfhwtpqc->ncfthpwq", hidden_states)
            output = hidden_states.reshape(shape=(-1, self.out_channels, num_frames * self.frame_patch_size, height * self.patch_size, width * self.patch_size))
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
