from dataclasses import dataclass
from typing import Optional

from diffusers.optimization import get_scheduler

from ..configs.base_config import PrintableConfig


@dataclass
class SchedulerConfig(PrintableConfig):
    """Basic scheduler config"""

    name: str = "constant"
    """Learning rate scheduler type."""
    num_warmup_steps: Optional[int] = None
    """Learning rate warmup steps."""
    num_warmup_steps_rate: Optional[float] = None
    """Learning rate warmup steps. If num_warmup_steps is not None, this is ignored."""

    def setup(self, optimizer, gradient_accumulation_steps=1, num_training_steps=None):
        kwargs = vars(self).copy()
        kwargs.pop("num_warmup_steps_rate")
        if num_training_steps is not None:
            kwargs["num_training_steps"] = num_training_steps * gradient_accumulation_steps
        if self.num_warmup_steps is None and self.num_warmup_steps_rate is None:
            kwargs["num_warmup_steps"] = 0
        elif self.num_warmup_steps is not None:
            kwargs["num_warmup_steps"] = self.num_warmup_steps
        elif self.num_warmup_steps_rate is not None:
            kwargs["num_warmup_steps"] = int(kwargs["num_training_steps"] * self.num_warmup_steps_rate)

        return get_scheduler(optimizer=optimizer, **kwargs)
