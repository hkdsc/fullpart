from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from ...configs.base_config import InstantiateConfig


@dataclass
class TextEncoderConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: TextEncoder)

    tokenizer_ckpt_path: Optional[str] = None
    """ckpt path for tokenizer"""

    text_encoder_ckpt_path: Optional[str] = None
    """ckpt path for text encoder"""

    max_length: int = 77
    """max length of tokens"""

    def from_pretrained(self) -> Any:
        """Returns the instantiated object using the config."""
        target = self._target(self)
        return target.tokenizer, target.text_encoder


class TextEncoderWrapper(nn.Module):
    def __init__(self, instance):
        super().__init__()
        self.register_module("_instance", instance)

    def __getattr__(self, name):
        if name in ["config", "dtype", "device"]:
            return getattr(self._instance, name)
        else:
            return super().__getattr__(name)

    def __call__(self, text_inputs, attention_mask, device):
        return self._instance(text_inputs.input_ids.to(device), attention_mask=attention_mask)


class TextEncoder(ModelMixin, ConfigMixin):
    text_encoder_config: TextEncoderConfig

    @register_to_config
    def __init__(
        self,
        text_encoder_config=None,
    ):
        super().__init__()
        self.text_encoder_config = text_encoder_config
