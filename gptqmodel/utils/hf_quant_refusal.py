# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Refuse plain-transformers loading of GPT-QModel-only quant methods.

``transformers`` dispatches a checkpoint's ``quantization_config.quant_method``
through its own registries. For a name it does not know,
``AutoHfQuantizer.supports_quant_method`` only logs "Unknown quantization type
... skip the quantization" and ``from_pretrained`` then random-initializes every
linear layer. Registering a refusing quantizer under each GPT-QModel-only name
turns that silent random-weight load into one actionable error.

The names live in :data:`gptqmodel._quant_methods.GPTQMODEL_ONLY_QUANT_METHODS`
so the package-init guard and the save-time ``hf_compat_warnings`` share one
table.
"""

from __future__ import annotations

from typing import Optional

from transformers.quantizers import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin


class GPTQModelOnlyQuantizationConfig(QuantizationConfigMixin):
    """Minimal HF config object for a GPT-QModel-only ``quant_method`` marker."""

    def __init__(self, quant_method: Optional[str] = None, **kwargs):
        self.quant_method = quant_method
        self.bits = kwargs.get("bits")

    def to_dict(self):
        return {"quant_method": self.quant_method, "bits": self.bits}

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict)


class GPTQModelOnlyHfQuantizer(HfQuantizer):
    """Fail closed: transformers cannot read this GPT-QModel layout."""

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        method = getattr(quantization_config, "quant_method", None)
        raise ValueError(
            f"GPT-QModel `{method}` checkpoints cannot be loaded by `transformers`; "
            "load them with `GPTQModel.load` or vLLM. "
            f"(quant_method: {method!r})"
        )

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self, safe_serialization=None) -> bool:
        return False
