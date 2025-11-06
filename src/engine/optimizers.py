from dataclasses import dataclass, field
from typing import Optional, Tuple, Type

import torch

from ..configs.base_config import PrintableConfig


@dataclass
class OptimizerConfig(PrintableConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.Adam
    """The optimizer class to use."""
    lr: float = 1e-4
    """The learning rate to use."""
    scale_lr: bool = False
    """Whether to scale learning rate."""
    eps: float = 1e-8
    """The epsilon value to use."""
    max_grad_norm: Optional[float] = None
    """The max norm to use for gradient clipping."""
    fused: Optional[bool] = None
    """Whether the fused implementation (CUDA only) is used."""

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("scale_lr")
        kwargs.pop("max_grad_norm")
        return self._target(params, **kwargs)


@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with AdamW"""

    _target: Type = torch.optim.AdamW

    betas: Tuple[float] = field(default_factory=lambda: (0.9, 0.999))
    """The betas to use."""
    weight_decay: float = 0.01
    """The weight decay to use."""
    max_grad_norm: Optional[float] = 1.0
    """The max norm to use for gradient clipping."""


class AdamOptimizerConfig(AdamWOptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.Adam
