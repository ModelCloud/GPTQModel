# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Refuse plain-transformers loading of GPT-QModel packed GGUF checkpoints.

``transformers`` reserves ``quant_method: "gguf"`` for its own
``GgufHfQuantizer``, which expects a ``.gguf`` file. GPT-QModel writes packed
GGUF tensors in safetensors and publishes ``gptqmodel_gguf`` instead.
Registering a quantizer under that name turns an otherwise silent
random-weight load into an actionable error.
"""

from __future__ import annotations

from transformers.quantizers import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from .._quant_methods import GGUF_EXPORT_QUANT_METHOD


class GPTQModelGGUFQuantizationConfig(QuantizationConfigMixin):
    """Minimal HF config object for the ``gptqmodel_gguf`` marker."""

    def __init__(self, **kwargs):
        self.quant_method = GGUF_EXPORT_QUANT_METHOD
        self.bits = kwargs.get("bits")

    def to_dict(self):
        return {"quant_method": GGUF_EXPORT_QUANT_METHOD, "bits": self.bits}

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict)


class GPTQModelGGUFHfQuantizer(HfQuantizer):
    """Fail closed: transformers cannot read GPT-QModel's packed GGUF tensors."""

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        raise ValueError(
            "GPT-QModel packed GGUF checkpoints cannot be loaded by `transformers`; "
            "load them with `GPTQModel.load` or vLLM. "
            f"(quant_method: {GGUF_EXPORT_QUANT_METHOD!r})"
        )

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self, safe_serialization=None) -> bool:
        return False
