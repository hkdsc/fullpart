import os
from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models.attention_processor import Attention
from diffusers.models import ModelMixin
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from einops import rearrange, repeat
import json
from safetensors.torch import load_file

from ...configs.base_config import PrintableConfig, InstantiateConfig
from ...utils import load_model
# new added
from ..normalization import AdaRMSNormSingle, RMSNorm

from trellis.modules.utils import convert_module_to_f16, convert_module_to_f32, convert_module_to_bf16

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

@torch.compile
def multiply_addition(a, b):
    return a * (b + 1)

class PixelUnshuffle3D(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.shape
        r = self.downscale_factor

        if in_depth % r != 0 or in_height % r != 0 or in_width % r != 0:
            raise ValueError(f"Input dimensions must be divisible by the downscale factor. "
                             f"Got input shape {x.shape} with downscale factor {r}")

        out_depth = in_depth // r
        out_height = in_height // r
        out_width = in_width // r
        out_channels = channels * (r ** 3)

        x = x.reshape(batch_size, channels, out_depth, r, out_height, r, out_width, r)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6) 
        x = x.reshape(batch_size, out_channels, out_depth, out_height, out_width)

        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, inner_dim, mult=4, scale=1.0, dropout=0.0, bias=False):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult * scale)
        self.linear1 = nn.Linear(dim, inner_dim, bias=bias)
        self.linear2 = nn.Linear(dim, inner_dim, bias=bias)
        self.linear3 = nn.Linear(inner_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    @torch.compile
    def silu_multiply(self, a, b):
        return F.silu(a) * b

    def forward(self, hidden_states):
        hidden_states_1 = self.linear1(hidden_states)
        hidden_states_2 = self.linear2(hidden_states)
        hidden_states = self.silu_multiply(hidden_states_1, hidden_states_2)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear3(hidden_states)
        return hidden_states


@dataclass
class TemporalAttentionConfig(PrintableConfig):
    attn_type: Literal["1d", "stfit", "neighbourhood_attn"] = "1d"
    """temporal attn type."""
    neighbourhood_attn_window_size: Union[int, Tuple[int, int, int]] = (5, 5, 5)
    """neighbourhood_attn window size."""
    neighbourhood_attn_dilation_size: Union[int, Tuple[int, int, int]] = (1, 1, 1)
    """neighbourhood_attn dilation size."""
    use_3d_rope: bool = True
    """Whether to use 3d rope in neighbourhood_attn and stfit."""
    theta_3d_rope: float = 10000.0
    """theta of 3d rope in neighbourhood_attn and stfit."""
    stfit_latent_dims_scale: int = 1
    """times of the latent dim number in stfit."""
    stfit_patch_size: Tuple[int, int] = (1, 2, 2)
    """merged token size in stfit."""


@dataclass
class TransformerXLModelConfigSingle(InstantiateConfig):
    _target: Type = field(default_factory=lambda: TransformerXLModelSingle)

    zero_linear: bool = False
    abandon_img_cond: bool = False
    drop_img_conds: float = 0.
    cornner_pos_emb: bool = False
    transformer_ckpt_path: Optional[Union[List[str], str]] = None
    id_emb: bool = False
    in_out_emb: bool = False
    cornner_in_out: bool = False
    ss_flow_weights_dir: str = ''

    def from_pretrained(self, ckpt_path, **kwargs):
        transformer = self._target(transformer_config=self, **kwargs)
        def rename_func(state_dict):
            new_dict = {}
            for k in state_dict.keys():
                ori_k = k
                if "transformer_blocks" not in k:  # global
                    if "scale_table" in k:
                        if "global_scale_table" not in k:
                            k = k.replace("scale_table", "global_scale_table")  # rename
                        if "weight" not in k:
                            k += ".weight"  # nn.Parameter -> nn.Embedding
                else:  # layer
                    # if "scale_table" in k and "scale_table.weight" not in k: 
                    if "scale_table" in k and "scale_table.weight" not in k:
                        if "scale_table_mm" not in k: 
                            k += ".weight"
                if k.startswith("transformer."):
                    k = k[12:]
                new_dict[k] = state_dict[ori_k]
            return new_dict
        print("=========================transformer_ckpt_path========================", self.transformer_ckpt_path)
        if isinstance(self.transformer_ckpt_path, str):
            transformer = load_model(transformer, self.transformer_ckpt_path, rename_func=rename_func)
        elif isinstance(self.transformer_ckpt_path, list):
            for transformer_ckpt_path in self.transformer_ckpt_path:
                transformer = load_model(transformer, transformer_ckpt_path, rename_func=rename_func)

        return transformer


from diffusers.models.attention import FeedForward, _chunked_feed_forward
from diffusers.utils.torch_utils import maybe_allow_in_graph


@maybe_allow_in_graph
class BasicTransformerXLBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        use_temp_attn: bool = False,
        image_temp_attn: bool = False,
        ffn_scale: float = 1.0,
        mm_blcok: bool = False,
        mm_dim: Optional[int] = None,
        use_zero_linear: bool = False,
        flat_block: bool = False,
    ):
        super().__init__()

        # Define 4 blocks. Each block has its own normalization layer.

        # 1. Self-Attn
        self.norm1 = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Temp-Attn
        self.image_temp_attn = image_temp_attn
        self.use_temp_attn = use_temp_attn
        if use_temp_attn:
            self.normt = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attnt = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )

            self.attnt.to_out[0].weight.data.zero_()
            self.attnt.to_out[0].bias.data.zero_()

        # 3. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )
        else:
            self.norm2 = None
            self.attn2 = None
        
        # 3.5 Multi-modality cross attention
        self.mm_block = mm_blcok
        self.use_zero_linear = use_zero_linear
        self.flat_block = flat_block
        if mm_blcok:
            assert mm_dim is not None
            # zero linear
            if self.use_zero_linear:
                self.zero_linear = nn.Linear(dim, dim)
                torch.nn.init.zeros_(self.zero_linear.weight)
                torch.nn.init.zeros_(self.zero_linear.bias)
            # before proj
            self.before_proj = PixArtAlphaCondProjection(mm_dim, dim)

            self.norm_mm = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.norm_mm_ca = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.mm_attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=mm_dim if self.before_proj is None else dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )
            # need init this block
            print("====initialize new transformer layers=====")
            def _basic_init(module):
                # Initialize transformer layers:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.mm_attn2.apply(_basic_init)
        else:
            self.norm_mm = None
            self.mm_attn2 = None
            self.zero_linear = None

        # 4. Feed-forward
        self.norm3 = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = SwiGLUFeedForward(
            dim,
            dropout=dropout,
            inner_dim=ff_inner_dim,
            scale=ffn_scale,
            bias=ff_bias,
        )

        # 5. Scale
        self.scale_table = nn.Embedding.from_pretrained(torch.randn(4, dim) / dim**0.5)

        # 6. mm scale
        if mm_blcok:
            self.scale_table_mm = nn.Embedding.from_pretrained(torch.randn(1, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self.register_buffer("tensor_0", torch.tensor([[0]]))
        self.register_buffer("tensor_1", torch.tensor([[1]]))
        self.register_buffer("tensor_2", torch.tensor([[2]]))
        self.register_buffer("tensor_3", torch.tensor([[3]]))

        if mm_blcok:
            self.register_buffer("tensor_mm", torch.tensor([[0]]))

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        spatial_attention_mask: Optional[torch.FloatTensor] = None,
        temporal_attention_mask: Optional[torch.FloatTensor] = None,
        spatial_temporal_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        mm_hidden_states: Optional[torch.FloatTensor] = None, 
        timestep: Optional[torch.LongTensor] = None,
        patch_resolution: Optional[Tuple[int, int, int]] = None,
        spatial_attn_mask_kwargs: Dict[str, Any] = None,
        temporal_attn_mask_kwargs: Dict[str, Any] = None,
        spatial_temporal_attn_mask_kwargs: Dict[str, Any] = None,
        cross_attn_mask_kwargs: Dict[str, Any] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        num_frames: int = 1,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # copied from diffusers/models/attention.py BasicTransformerBlock.forward
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        batch_size, num_patches = hidden_states.shape[0], hidden_states.shape[1]
        assert batch_size % num_frames == 0
        real_batch_size = batch_size // num_frames
        temporal_patch_resolution = patch_resolution[0] if patch_resolution is not None else None
        spatial_patch_resolution = patch_resolution[1:] if patch_resolution is not None else None

        timestep_reshape = timestep.reshape(real_batch_size, 4, -1).chunk(4, dim=1)
        scale_msa = self.scale_table(self.tensor_0) + timestep_reshape[0]
        scale_mta = self.scale_table(self.tensor_1) + timestep_reshape[1]
        scale_mca = self.scale_table(self.tensor_2) + timestep_reshape[2]
        scale_mlp = self.scale_table(self.tensor_3) + timestep_reshape[3]

        scale_msa = repeat(scale_msa, "b 1 d -> (b f) 1 d", f=num_frames)
        scale_mca = repeat(scale_mca, "b 1 d -> (b f) 1 d", f=num_frames)
        scale_mlp = repeat(scale_mlp, "b 1 d -> (b f) 1 d", f=num_frames)

        # scale mm
        if self.norm_mm is not None:
            scale_mca_mm = self.scale_table_mm(self.tensor_mm) + timestep_reshape[2]
            scale_mca_mm = repeat(scale_mca_mm, "b 1 d -> (b f) 1 d", f=num_frames)

        norm_hidden_states = multiply_addition(self.norm1(hidden_states), scale_msa)
        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=spatial_attention_mask,
            patch_resolution=spatial_patch_resolution,
            selfattn_mask_kwargs=spatial_attn_mask_kwargs,
        )
        hidden_states = hidden_states + attn_output

        # 2. Temp-Attention
        if self.use_temp_attn and (num_frames > 1 or self.image_temp_attn):
            if self.attnt.processor.token_merge is not None:
                hidden_states = rearrange(hidden_states, "(b f) p d -> b (f p) d", f=num_frames)
                norm_hidden_states = multiply_addition(self.normt(hidden_states), scale_mta)
                attn_output = self.attnt(
                    norm_hidden_states,
                    attention_mask=spatial_temporal_attention_mask,
                    patch_resolution=patch_resolution,
                    selfattn_mask_kwargs=spatial_temporal_attn_mask_kwargs,
                )
                hidden_states = hidden_states + attn_output
                hidden_states = rearrange(hidden_states, "b (f p) d -> (b f) p d", f=num_frames)
            else:
                scale_mta = repeat(scale_mta, "b 1 d -> (b p) 1 d", p=num_patches)
                hidden_states = rearrange(hidden_states, "(b f) p d -> (b p) f d", b=real_batch_size)
                norm_hidden_states = multiply_addition(self.normt(hidden_states), scale_mta)
                attn_output = self.attnt(
                    norm_hidden_states,
                    attention_mask=temporal_attention_mask if num_frames > 1 else None,
                    patch_resolution=temporal_patch_resolution,
                    selfattn_mask_kwargs=temporal_attn_mask_kwargs if num_frames > 1 else None,
                )
                hidden_states = hidden_states + attn_output
                hidden_states = rearrange(hidden_states, "(b p) f d -> (b f) p d", b=real_batch_size)

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = multiply_addition(self.norm2(hidden_states), scale_mca)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                patch_resolution=spatial_patch_resolution,
                selfattn_mask_kwargs=spatial_attn_mask_kwargs,
                crossattn_mask_kwargs=cross_attn_mask_kwargs,
            )
            hidden_states = attn_output + hidden_states
        
        # 4. Multi-modal Cross-Attention
        if self.mm_attn2 is not None:
            norm_hidden_states = multiply_addition(self.norm_mm(hidden_states), scale_mca_mm)
            # before proj
            mm_hidden_states = self.before_proj(mm_hidden_states)
            mm_hidden_states = self.norm_mm_ca(mm_hidden_states)
            attn_output = self.mm_attn2(
                norm_hidden_states,
                encoder_hidden_states=mm_hidden_states,
                attention_mask=None, # encoder_attention_mask,
                patch_resolution=spatial_patch_resolution,
                selfattn_mask_kwargs=spatial_attn_mask_kwargs,
                crossattn_mask_kwargs=None,#cross_attn_mask_kwargs,
            )
            # apply zero linear to avoid nan gradient
            if self.use_zero_linear:
                attn_output = self.zero_linear(attn_output)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = multiply_addition(self.norm3(hidden_states), scale_mlp)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

        return hidden_states


class TransformerXLModelSingle(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        transformer_config: TransformerXLModelConfigSingle = None,
    ):
        super().__init__()

        self.zero_linear = transformer_config.zero_linear
        self.abandon_img_cond = transformer_config.abandon_img_cond
        self.drop_img_conds = transformer_config.drop_img_conds
        self.cornner_pos_emb = transformer_config.cornner_pos_emb
        self.id_emb = transformer_config.id_emb
        self.in_out_emb = transformer_config.in_out_emb
        self.cornner_in_out = transformer_config.cornner_in_out
        self.ss_flow_weights_dir = transformer_config.ss_flow_weights_dir

        # initialize voxel flow model
        from .transformer_voxel_part import SparseStructureFlowModel
        with open(os.path.join(self.ss_flow_weights_dir, 'ss_flow_img_dit_L_16l8_fp16.json'), 'r') as f:
            voxel_branch_config = json.load(f)
        self.global_block_list_voxel = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22] # list(range(24)) # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22] # 12 blocks
        self.voxel_branch = SparseStructureFlowModel(**voxel_branch_config['args'], global_block_id_list=self.global_block_list_voxel, use_zero_linear=self.zero_linear,
                                                        abandon_img_cond=self.abandon_img_cond,
                                                        cornner_pos_emb=self.cornner_pos_emb,
                                                        id_emb=self.id_emb,
                                                        in_out_emb=self.in_out_emb,
                                                        cornner_in_out=self.cornner_in_out)
        self.voxel_branch.load_state_dict(load_file(os.path.join(self.ss_flow_weights_dir, 'ss_flow_img_dit_L_16l8_fp16.safetensors')),
                                          strict=False)
        self.convert_to_bf16()
    
    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.apply(convert_module_to_f32)
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.apply(convert_module_to_bf16)
        
    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        noisy_voxel_latent: Optional[torch.Tensor] = None,
        img_conds: Optional[torch.Tensor] = None,
        voxel_plucker: Optional[torch.Tensor] = None,
        camera: Optional[torch.Tensor] = None,
        force_drop_flag: bool = False,
        bboxes: Optional[Any] = None,
        np: Optional[int] = None,
    ):
        # test voxel flow model
        batch_size = noisy_voxel_latent.shape[0]
        assert np == noisy_voxel_latent.shape[2] # also np
        
        if self.training:
            assert batch_size == 1, "only support batch size = 1 for now"
            assert len(bboxes) == 1
        noisy_voxel_latent = rearrange(noisy_voxel_latent, 'b c f h w d -> (b f) c h w d')
        bboxes = torch.cat(bboxes)
        timestep_voxel = timestep

        voxel_hidden_states, t_emb_voxel, cond_voxel = self.voxel_branch.forward_pre_part(noisy_voxel_latent, timestep_voxel, img_conds, bboxes,
                                                                                          np)

        for i, voxel_block in enumerate(self.voxel_branch.blocks):
            voxel_hidden_states = voxel_block(voxel_hidden_states, t_emb_voxel, cond_voxel, num_parts=np)

        voxel_output = self.voxel_branch.forward_post(voxel_hidden_states, dtype=voxel_hidden_states.dtype)

        voxel_output = rearrange(voxel_output, "(b np) c h w d -> b c np h w d", np=np)
        
        if not return_dict:
            return (None, voxel_output)

        return Transformer2DModelOutput(sample=None, sample_voxel=voxel_output)
