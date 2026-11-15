# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Refuse plain-transformers loading of GPT-QModel packed checkpoints.

``transformers`` 5.x only warns ``Unknown quantization type`` for a
``quant_method`` it does not implement and then returns a model whose packed
tensors were randomly initialized. Registering a quantizer under every
GPT-QModel-only name turns that silent quality cliff into an actionable error.
"""

from __future__ import annotations

from transformers.quantizers import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from .._quant_methods import REFUSED_HF_QUANT_METHODS


class GPTQModelRefusingQuantizationConfig(QuantizationConfigMixin):
    """Minimal HF config object for a GPT-QModel-only ``quant_method``."""

    def __init__(self, **kwargs):
        self.quant_method = kwargs.get("quant_method")
        self.bits = kwargs.get("bits")

    def to_dict(self):
        return {"quant_method": self.quant_method, "bits": self.bits}

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict)


class GPTQModelRefusingHfQuantizer(HfQuantizer):
    """Fail closed: transformers cannot read GPT-QModel's packed tensors."""

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        method = getattr(quantization_config, "quant_method", None)
        label = REFUSED_HF_QUANT_METHODS.get(method, method)
        raise ValueError(
            f"GPT-QModel {label} checkpoints cannot be loaded by `transformers`; "
            "load them with `GPTQModel.load` or vLLM. "
            f"(quant_method: {method!r})"
        )

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self, safe_serialization=None) -> bool:
        return False
