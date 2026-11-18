# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Test-only reference encoder for FP8 decode kernels.

Production GPT-QModel is FP8 load-only: it reads real checkpoints and never
quantizes or exports FP8 weights. Kernel tests still need synthetic FP8
payloads, so this module keeps the small reference encoder the tests used to
reach through ``TorchFP8Linear.pack_original``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.fp8 import _compute_scale_inv, _fp8_dtype_from_name, _weight_to_matrix
from gptqmodel.quantization.config import (
    _normalize_fp8_fmt,
    _normalize_fp8_weight_block_size,
    _normalize_fp8_weight_scale_method,
)


def quantize_fp8_weight(
    weight: torch.Tensor,
    *,
    format: str = "float8_e4m3fn",
    weight_scale_method: str = "row",
    weight_block_size: Optional[Tuple[int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one dense matrix as a synthetic FP8 checkpoint payload."""

    if weight.ndim != 2:
        raise ValueError(f"FP8 quantization expects a 2D weight matrix, got shape {tuple(weight.shape)}.")

    format = _normalize_fp8_fmt(format)
    if format == "float8_e8m0fnu":
        raise ValueError(
            "TorchFP8Linear does not quantize dense weights to float8_e8m0fnu; "
            "use that format only for dequantization of existing checkpoints."
        )
    block_size = _normalize_fp8_weight_block_size(weight_block_size)
    weight_scale_method = _normalize_fp8_weight_scale_method(
        weight_scale_method,
        weight_block_size=block_size,
    )
    fp8_dtype = _fp8_dtype_from_name(format)
    fp8_max = torch.finfo(fp8_dtype).max

    weight = weight.to(device="cpu", dtype=torch.float32).contiguous()

    if weight_scale_method == "tensor":
        scale_inv = _compute_scale_inv(weight.abs().amax(), fp8_max)
        quantized = torch.clamp(weight * scale_inv, min=-fp8_max, max=fp8_max).to(fp8_dtype)
        return quantized.contiguous(), scale_inv.to(torch.float32)

    if weight_scale_method == "row":
        scale_inv = _compute_scale_inv(weight.abs().amax(dim=1), fp8_max)
        quantized = torch.clamp(
            weight * scale_inv.unsqueeze(1),
            min=-fp8_max,
            max=fp8_max,
        ).to(fp8_dtype)
        return quantized.contiguous(), scale_inv.to(torch.float32).contiguous()

    if block_size is None:
        raise ValueError("FP8 block quantization requires `weight_block_size`.")

    block_rows, block_cols = block_size
    rows, cols = weight.shape
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError(
            f"FP8 block quantization expects shape {tuple(weight.shape)} to be divisible by block size "
            f"{block_size}."
        )

    row_blocks = rows // block_rows
    col_blocks = cols // block_cols
    blocks = weight.reshape(row_blocks, block_rows, col_blocks, block_cols)
    scale_inv = _compute_scale_inv(blocks.abs().amax(dim=(1, 3)), fp8_max)
    scaled = blocks * scale_inv.unsqueeze(1).unsqueeze(3)
    quantized = torch.clamp(scaled, min=-fp8_max, max=fp8_max).to(fp8_dtype).reshape(rows, cols)
    return quantized.contiguous(), scale_inv.to(torch.float32).contiguous()


def _install(module, weight: torch.Tensor, scale_inv: torch.Tensor, linear: nn.Module) -> None:
    if "weight" in module._buffers:
        module.weight = weight
    else:
        module.register_buffer("weight", weight)
    if "weight_scale_inv" in module._buffers:
        module.weight_scale_inv = scale_inv
    else:
        module.register_buffer("weight_scale_inv", scale_inv)

    if linear.bias is not None:
        bias = linear.bias.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if "bias" in module._buffers:
            module.bias = bias
        else:
            module.register_buffer("bias", bias)
    else:
        module.bias = None

    module._scaled_mm_hard_disabled = False
    module._native_dequant_disabled = False
    module._invalidate_compiled()


def pack_fp8_module(module, linear: nn.Module, *, smooth=None) -> None:
    """Install a synthetic FP8 payload computed from ``linear``'s dense weight."""

    weight = _weight_to_matrix(linear).to(device="cpu", dtype=torch.float32)
    if smooth is not None:
        from gptqmodel.nn_modules.qlinear.gguf import _apply_optional_smoother

        weight = _apply_optional_smoother(
            weight,
            smooth=smooth,
            group_size=module.smooth_block_size(),
        ).contiguous()
    qweight, scale_inv = quantize_fp8_weight(
        weight,
        format=module.fp8_format,
        weight_scale_method=module.weight_scale_method,
        weight_block_size=module.weight_block_size,
    )
    _install(module, qweight, scale_inv, linear)


def install_fp8_payload(module, linear: nn.Module, weight: torch.Tensor, scale_inv: torch.Tensor) -> None:
    """Install an explicit FP8 payload without recomputing its scales."""

    if weight.shape != (module.out_features, module.in_features) or weight.dtype != module.fp8_dtype:
        raise ValueError("Calibrated FP8 weight does not match packed storage geometry or dtype")
    if scale_inv.shape != module._scale_shape() or scale_inv.dtype != torch.float32:
        raise ValueError("Calibrated FP8 scale does not match packed storage geometry or dtype")
    if not torch.isfinite(scale_inv).all() or (scale_inv <= 0).any():
        raise ValueError("Calibrated FP8 inverse scales must be finite and positive")
    _install(
        module,
        weight.detach().to(device="cpu").contiguous(),
        scale_inv.detach().to(device="cpu").contiguous(),
        linear,
    )


LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _dynamic_override(dynamic, name: str):
    """Return the per-module FP8 override that applies to ``name``, if any."""

    if not dynamic:
        return None
    import re

    for pattern, overrides in dynamic.items():
        probe = pattern
        if probe[:2] in {"+:", "-:"}:
            probe = probe[2:]
        try:
            matched = re.search(probe, name) is not None
        except re.error:
            matched = False
        if matched:
            return dict(overrides or {})
    return None


def write_fp8_checkpoint(base, destination, *, config=None, dynamic=None, suffixes=LINEAR_SUFFIXES):
    """Write a load-only FP8 checkpoint from a dense model directory.

    GPT-QModel no longer quantizes FP8 (#5242), so a test that needs an FP8
    checkpoint on disk builds one here: every dense projection named by
    ``suffixes`` is replaced by a ``TorchFP8Linear`` holding a reference-encoded
    payload, and the resulting ``quantization_config`` is what the loader reads.
    ``dynamic`` mirrors ``FP8Config.dynamic`` and overrides the encoding of the
    modules its regexes match.
    """

    from pathlib import Path

    import torch as _torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.quantization.config import FP8Config

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if config is None:
        config = FP8Config(format="float8_e4m3fn", weight_scale_method="row")
    if dynamic:
        config = config.__class__(**{**config.to_dict(), "dynamic": dynamic})

    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=_torch.float32).eval()

    def factory(name, dense):
        overrides = _dynamic_override(dynamic, name) or {}
        module = TorchFP8Linear(
            bits=8,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=dense.in_features,
            out_features=dense.out_features,
            bias=dense.bias is not None,
            register_buffers=True,
            format=overrides.get("format", config.format),
            weight_scale_method=overrides.get("weight_scale_method", config.weight_scale_method),
            weight_scale_semantics="inverse",
            weight_block_size=overrides.get("weight_block_size"),
        )
        pack_fp8_module(module, dense)
        return module

    for layer in range(model.config.num_hidden_layers):
        for suffix in suffixes:
            name = f"model.layers.{layer}.{suffix}"
            try:
                dense = model.get_submodule(name)
            except AttributeError:
                continue
            model.set_submodule(name, factory(name, dense))

    tensors = {}
    seen = {}
    for name, value in model.state_dict().items():
        # Tied weights (lm_head <- embed_tokens) share storage; safetensors
        # refuses to write the same buffer twice, so keep the first name.
        key = (value.data_ptr(), tuple(value.shape))
        if key in seen:
            continue
        seen[key] = name
        tensors[name] = value.contiguous()
    save_file(tensors, str(destination / "model.safetensors"), metadata={"format": "pt"})
    model.config.quantization_config = config.to_dict()
    model.config.save_pretrained(destination)

    # A load needs a tokenizer; carry over the base model's files when present.
    import shutil as _shutil

    base = Path(base)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ):
        source_file = base / name
        if source_file.is_file():
            _shutil.copyfile(source_file, destination / name)
    return config
