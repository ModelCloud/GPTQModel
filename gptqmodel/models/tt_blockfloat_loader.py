# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Route a native Tenstorrent block-float checkpoint into ``GPTQModel.load``.

The checkpoint itself is an ordinary Hugging Face model directory: shards plus
``model.safetensors.index.json``. Nothing here inspects bespoke payload names --
:func:`~gptqmodel.utils.tt_blockfloat_quant.build_blockfloat_model` reads the
index and swaps any module that owns a ``.codes`` tensor, and this module only
decides *whether* a directory is a block-float checkpoint and wraps the result as
a ``BaseQModel``.
"""

from __future__ import annotations

import json
import os

from ..quantization import METHOD, QuantizeConfig
from ..utils.tt_blockfloat_quant import build_blockfloat_model


def read_quantization_config(model_local_path: str) -> dict:
    """The checkpoint's ``quantization_config`` block, or ``{}``."""

    config_path = os.path.join(model_local_path, "config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path) as handle:
        config = json.load(handle)
    quantization_config = config.get("quantization_config") or {}
    return dict(quantization_config) if isinstance(quantization_config, dict) else {}


def is_tt_blockfloat_checkpoint(model_local_path: str) -> bool:
    """Whether a local directory declares the ``tt-blockfloat`` method.

    Decided from ``config.json`` alone, the same field every other quantized
    checkpoint is dispatched on. No file-name sniffing.
    """

    if not isinstance(model_local_path, str) or not os.path.isdir(model_local_path):
        return False
    return read_quantization_config(model_local_path).get("quant_method") == METHOD.TT_BLOCKFLOAT


def build_tt_blockfloat_model(
    model_local_path: str, *, dtype=None
):
    """Build the torch model, returning ``(model, restored, info)``.

    ``dtype`` is passed through unchanged so ``None`` means "the checkpoint's
    own precision" (bf16 for a bf16 source), the same as ``from_pretrained``.
    """

    return build_blockfloat_model(model_local_path, dtype=dtype)


def quantize_config_from_checkpoint(model_local_path: str) -> QuantizeConfig:
    """Build the ``QuantizeConfig`` a loaded block-float model reports."""

    import torch

    qcfg = read_quantization_config(model_local_path)
    config = QuantizeConfig(
        method=METHOD.TT_BLOCKFLOAT,
        bits=int(qcfg.get("bits", 4)),
        block=int(qcfg.get("block", 16)),
        embed_bits=int(qcfg.get("embed_bits", 8)),
        sym=True,
        desc_act=False,
        # The kernel stores codes as ``uint8``. Leaving the generic default
        # (``int32``) makes any consumer that validates ``pack_dtype`` -- the
        # writer included -- reject a checkpoint that loads perfectly.
        pack_dtype=torch.uint8,
    )
    widths = qcfg.get("bits_per_layer")
    if isinstance(widths, dict) and widths:
        # Per-module widths are what make a mixed checkpoint re-exportable; the
        # flat ``bits`` field is only the fallback for an undeclared module.
        config.meta_set(key="bits_per_layer", value=dict(widths))
    return config


__all__ = [
    "build_tt_blockfloat_model",
    "is_tt_blockfloat_checkpoint",
    "quantize_config_from_checkpoint",
    "read_quantization_config",
]
