from typing import Dict, Optional, Tuple
import numbers

import torch
import torch.nn as nn

from .embeddings import CombinedTimestepSizeEmbeddings


class AdaRMSNormSingle(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_scales: int = 4,
        use_text_condition: bool = False,
        use_resolution_condition: bool = False,
        use_aspect_ratio_condition: bool = False,
        use_frames_condition: bool = False,
        use_fps_condition: bool = False,
        split_conditions: bool = False,
    ):
        super().__init__()

        self.emb = CombinedTimestepSizeEmbeddings(
            embedding_dim,
            use_text_condition=use_text_condition,
            use_resolution_condition=use_resolution_condition,
            use_aspect_ratio_condition=use_aspect_ratio_condition,
            use_frames_condition=use_frames_condition,
            use_fps_condition=use_fps_condition,
            split_conditions=split_conditions,
        )

        self.split_conditions = split_conditions

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, num_scales * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        if self.split_conditions:
            embedded_timestep, embedded_spatial, embedded_temporal = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
            return self.linear(self.silu(embedded_timestep + embedded_spatial)), self.linear(self.silu(embedded_timestep + embedded_spatial + embedded_temporal)), embedded_timestep + embedded_spatial
        else:
            embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
            return self.linear(self.silu(embedded_timestep)), embedded_timestep


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):       
        if self.weight is not None:
            hidden_states = rmsnorm(hidden_states, self.weight, self.eps, self.weight.dtype)
        else:
            hidden_states = rmsnorm_without_weight(hidden_states, self.eps, hidden_states.dtype)

        return hidden_states


@torch.compile
def rmsnorm(hidden_states, weight, eps, dtype):
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states.to(dtype)
    hidden_states = hidden_states * weight
    return hidden_states


@torch.compile
def rmsnorm_without_weight(hidden_states, eps, dtype):
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states.to(dtype)
    return hidden_states
