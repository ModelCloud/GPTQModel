# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Load a native Tenstorrent block-float checkpoint through ``GPTQModel.load``.

A ``tt-blockfloat`` checkpoint is not a GPTQ checkpoint: its decoder weights are
packed mantissa codes plus one exponent per block of 16 values, on the grid the
Tenstorrent hardware unpacks natively. None of the GPTQ keys (``qweight``,
``qzeros``, ``scales``, ``g_idx``) exist, so the generic quantized loader has
nothing to bind and would report every layer as missing.

This module binds them: the decoder linears become
:class:`~gptqmodel.nn_modules.qlinear.tt_blockfloat.TTBlockfloatLinear`
modules that hold the codes and exponents, and the embedding, norms and head
come from the checkpoint's own payloads.
"""

from __future__ import annotations

import json
import os

import torch

from ..quantization import METHOD, QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.tt_blockfloat_quant import (
    EMBED_PAYLOAD,
    NORMS_PAYLOAD,
    _decoder_blocks,
    _unpack_codes,
    read_blockfloat_checkpoint,
    recompose_blockfloat,
)


log = setup_logger()


def is_tt_blockfloat_checkpoint(model_local_path: str) -> bool:
    """Whether ``model_local_path`` holds a native block-float checkpoint."""

    if not isinstance(model_local_path, str) or not os.path.isdir(model_local_path):
        return False
    if not os.path.isfile(os.path.join(model_local_path, "blockfloat.json")):
        return False
    return os.path.isfile(os.path.join(model_local_path, "blockfloat.safetensors")) or os.path.isfile(
        os.path.join(model_local_path, "blockfloat.pt")
    )


def _read_quantization_config(model_local_path: str) -> dict:
    with open(os.path.join(model_local_path, "config.json")) as handle:
        config = json.load(handle)
    return dict(config.get("quantization_config") or {})


def _resolve_layer_module(blocks, dotted: str):
    """Resolve ``"<layer>.<submodule>"`` against the decoder block list.

    Checkpoint keys are relative to the decoder stack (``"0.self_attn.q_proj"``),
    not absolute paths from the model root, so the block list is the anchor.
    """

    index, _, rest = dotted.partition(".")
    block = blocks[int(index)]
    parts = rest.split(".")
    parent = block
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def build_tt_blockfloat_model(
    model_local_path: str,
    *,
    dtype: torch.dtype = torch.float32,
):
    """Build a torch model whose decoder linears hold block-float codes.

    Returns ``(model, restored_layers, info)``.
    """

    from transformers import AutoConfig, AutoModelForCausalLM

    from ..nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
    from ..utils.backend import BACKEND

    config = AutoConfig.from_pretrained(model_local_path)
    model = AutoModelForCausalLM.from_config(config).to(dtype=dtype)

    qcfg = _read_quantization_config(model_local_path)
    block = int(qcfg.get("block", 16))

    # Dense remainder: norms, and the head when it is not tied to the embedding.
    norms_path = os.path.join(model_local_path, NORMS_PAYLOAD)
    embed_path = os.path.join(model_local_path, EMBED_PAYLOAD)
    # Both are required. A decoder-only directory (``save_blockfloat_checkpoint``)
    # has neither, and loading it here would leave the embedding and every norm
    # at their random initial values while still returning a model.
    for path, name in ((norms_path, NORMS_PAYLOAD), (embed_path, EMBED_PAYLOAD)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{model_local_path} has no {name}, so it is a decoder-only "
                f"checkpoint. Load it with load_blockfloat_into_model() against "
                f"the base model, or re-save it with save_blockfloat_model()."
            )

    from safetensors.torch import load_file, safe_open

    _missing, unexpected = model.load_state_dict(load_file(norms_path), strict=False)
    unexpected = [name for name in unexpected if not name.endswith("lm_head.weight")]
    if unexpected:
        raise ValueError(
            f"{norms_path} contains tensors this config has no home for: "
            f"{sorted(unexpected)[:5]}"
        )

    # Embedding (and, when tied, the output head).
    with safe_open(embed_path, framework="pt") as handle:
        embed_bits = int((handle.metadata() or {}).get("bits", "8"))
        codes = handle.get_tensor("codes")
        shape = tuple(int(v) for v in handle.get_tensor("code_shape"))
        exponents = handle.get_tensor("exponents")
    weights = recompose_blockfloat(
        _unpack_codes(codes, embed_bits, shape), exponents, bits=embed_bits
    )
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(weights.to(dtype))

    # Decoder linears: replace the dense modules with code-holding ones.
    payload = read_blockfloat_checkpoint(model_local_path)
    blocks = _decoder_blocks(model)
    restored = {}
    for key, entry in payload["layers"].items():
        parent, attr = _resolve_layer_module(blocks, key)
        dense = getattr(parent, attr)
        has_bias = getattr(dense, "bias", None) is not None
        linear = TTBlockfloatLinear(
            bits=int(entry["bits"]),
            in_features=int(dense.in_features),
            out_features=int(dense.out_features),
            bias=has_bias,
            backend=BACKEND.TT_BLOCKFLOAT,
            adapter=None,
            name=key,
            dtype=dtype,
            block=block,
        )
        linear.set_quantized_state(entry["codes"], entry["exponents"])
        if has_bias:
            bias = entry["bias"]
            if bias is None:
                bias = dense.bias.detach().to(torch.float16)
            linear.bias = bias.to(torch.float16).contiguous()
        linear = linear.to(dtype=torch.float32 if dtype == torch.float32 else dtype)
        setattr(parent, attr, linear)
        restored[key] = list(entry["shape"])

    model.eval()
    return model, restored, {
        "bits": int(payload["bits"]),
        "bits_per_layer": sorted({int(e["bits"]) for e in payload["layers"].values()}),
        "block": block,
        "quantization_config": qcfg,
    }


def quantize_config_from_checkpoint(model_local_path: str) -> QuantizeConfig:
    """Build the ``QuantizeConfig`` a loaded block-float model reports."""

    qcfg = _read_quantization_config(model_local_path)
    return QuantizeConfig(
        method=METHOD.TT_BLOCKFLOAT,
        bits=int(qcfg.get("bits", 4)),
        block=int(qcfg.get("block", 16)),
        embed_bits=int(qcfg.get("embed_bits", 8)),
        sym=True,
        desc_act=False,
    )


__all__ = [
    "build_tt_blockfloat_model",
    "is_tt_blockfloat_checkpoint",
    "quantize_config_from_checkpoint",
]
