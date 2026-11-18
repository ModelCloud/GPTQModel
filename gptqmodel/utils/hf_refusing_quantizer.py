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

from .._quant_methods import REFUSED_HF_QUANT_METHODS, refused_hf_load_remedy


class GPTQModelRefusingQuantizationConfig(QuantizationConfigMixin):
    """Minimal HF config object for a GPT-QModel-only ``quant_method``."""

    def __init__(self, **kwargs):
        self.quant_method = kwargs.get("quant_method")
        self.bits = kwargs.get("bits")
        # Preserve every other field: when the refusal is bypassed for a
        # checkpoint that ships its own modeling file, the remote class reads
        # ``bits_per_layer``, ``embed_bits``, ``block`` and ``block_axis`` back
        # out of this object.
        for key, value in kwargs.items():
            if key not in ("quant_method", "bits"):
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict)


def refusal_message(method, *, label: str | None = None, detail: str | None = None) -> str:
    """The actionable error for a checkpoint transformers cannot read.

    ``label`` overrides the human-readable format name for a method that is
    not itself in ``REFUSED_HF_QUANT_METHODS`` (the standard ``fp8`` name is
    owned by transformers and only refused for GPT-QModel's inverse layout).
    ``detail`` appends an extra config marker (for example
    ``weight_scale_semantics: 'inverse'``) so the refusal names why the
    standard ``fp8`` method is still a GPT-QModel-only layout.
    """

    label = label or REFUSED_HF_QUANT_METHODS.get(method, method)
    marker = f", {detail}" if detail else ""
    return (
        f"GPT-QModel {label} checkpoints cannot be loaded by `transformers`; "
        f"load them with {refused_hf_load_remedy(method)}. "
        f"(quant_method: {method!r}{marker})"
    )


class GPTQModelRefusingHfQuantizer(HfQuantizer):
    """Fail closed: transformers cannot read GPT-QModel's packed tensors."""

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        method = getattr(quantization_config, "quant_method", None)
        # A Tenstorrent block-float checkpoint that ships its own modeling file
        # is loaded by that class through ``auto_map`` + ``trust_remote_code``,
        # not by a stock transformers model. The writer records the shipped file
        # in the quantization config, so let this documented path through.
        # Everything else (including a bare ``tt-blockfloat`` config with no
        # shipped loader) still fails closed instead of random-initializing.
        shipped = getattr(quantization_config, "shipped_modeling_file", None)
        # ``checkpoint_format`` is the marker older writer versions emitted; it
        # is only written by ``save_blockfloat_model``, which also wrote the
        # modeling file and ``auto_map``.
        legacy_shipped = (
            getattr(quantization_config, "checkpoint_format", None) == "tt-blockfloat"
        )
        if method == "tt-blockfloat" and (shipped or legacy_shipped):
            super().__init__(quantization_config, **kwargs)
            return
        if method == "tt-blockfloat":
            # vLLM has no tt-blockfloat loader, so do not point at it.
            raise ValueError(
                "GPT-QModel Tenstorrent block-float checkpoints cannot be loaded "
                "by `transformers` without the `modeling_tt_blockfloat.py` file "
                "written by `save_blockfloat_model`; load them with "
                "`GPTQModel.load` or re-save the checkpoint. "
                f"(quant_method: {method!r})"
            )
        raise ValueError(refusal_message(method))

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self, safe_serialization=None) -> bool:
        return False
