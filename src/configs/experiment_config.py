from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
import yaml

from ..utils import CONSOLE, log_to_rank0
from .base_config import InstantiateConfig


@dataclass
class ExperimentConfig(InstantiateConfig):
    """Full config contents for running an experiment. Any experiment types (like training) will be
    subclassed from this, and must have their _target field defined accordingly."""

    output_dir: Path = Path("exps")
    """relative or absolute output directory to save all checkpoints and logging"""
    experiment_name: Optional[str] = None
    """Experiment name. If None, will automatically be set to dataset name"""
    timestamp: str = "{timestamp}"
    """Experiment timestamp."""
    relative_ckpt_dir: Path = Path("ckpts")
    """Relative path to save all checkpoints."""
    git_commit: str = ""
    """Current git commit hash."""
    unet_ckpt_path: Optional[str] = None
    """Alias for --manager.unet.unet_ckpt_path"""

    def set_timestamp(self) -> None:
        """Dynamically set the experiment timestamp"""
        if self.timestamp == "{timestamp}":
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def set_experiment_name(self) -> None:
        """Dynamically set the experiment name"""
        if self.experiment_name is None:
            self.experiment_name = "unnamed"

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment name
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.experiment_name}")

    def get_checkpoint_dir(self) -> Path:
        """Retrieve the checkpoint directory"""
        return Path(self.get_base_dir() / self.relative_ckpt_dir)

    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        if torch.distributed.get_rank() == 0:
            CONSOLE.rule("Config")
            CONSOLE.print(self)
            CONSOLE.rule("")

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        log_to_rank0(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")
