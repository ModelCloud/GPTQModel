# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Humming weight-only CUDA GEMM runtime facade.

`Humming <https://github.com/vllm-project/humming>`_ is a standalone Apache-2.0
library of JIT-compiled CUDA kernels for weight-only quantized GEMM (W1-W8,
including the MX/NVFP4 family). It is distributed as the ``humming-kernels``
PyPI package, and both vLLM and SGLang consume it as an optional runtime
dependency rather than vendoring CUDA sources. GPTQModel follows the same
recipe: nothing is added to ``gptqmodel_ext`` and this module is the only place
that imports ``humming``.

The integration itself is small. Humming owns the packed layouts, so the
conversion from a GPTQ/AWQ checkpoint is delegated to the schema objects that
ship with the package (``GPTQWeightSchema`` / ``AWQWeightSchema``); the helpers
here only build the ``LayerConfig``, allocate the per-layer Stream-K
synchronization buffer, and hand the transformed tensors to
``humming_forward``.

Everything is import-guarded so a missing or unusable install degrades into a
normal kernel-selection rejection (``validate_once``) instead of an import
error for the whole package.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import torch

from .logger import setup_logger
from .rocm import IS_ROCM


log = setup_logger()

# Humming's dense GEMM targets SM75 (Turing) and newer; it has no HIP port.
HUMMING_MIN_CAPABILITY: Tuple[int, int] = (7, 5)

# Humming pads both GEMM dimensions to its kernel tiles. Padding is applied to
# the packed tensors only, so the layer keeps its logical in/out features.
HUMMING_PAD_N_TO_MULTIPLE = 256
HUMMING_PAD_K_TO_MULTIPLE = 128

# Stream-K synchronization state. Humming's own ``HummingLayer`` allocates the
# same size, and it must never be shared between layers: concurrent layers can
# run on different CUDA streams.
HUMMING_LOCKS = 2048

# Checkpoint layouts this port can hand to humming without reinterpreting bits.
# GPTQModel stores 3-, 5-, 6- and 7-bit GPTQ weights in its own word layouts
# (the 10-1-10-1-10 3-bit scheme and the split-plane 5/6/7-bit planar scheme),
# neither of which is the continuous little-endian packing humming reads. AWQ
# keeps only the classic interleaved INT4 layout, which is the single layout
# humming's AWQ converter undoes.
HUMMING_GPTQ_BITS: Tuple[int, ...] = (2, 4, 8)
HUMMING_AWQ_BITS: Tuple[int, ...] = (4,)

# Humming groups weight scales per power-of-two block; these are the layouts
# GPTQModel checkpoints actually use. ``-1`` is one group per output channel.
HUMMING_GROUP_SIZES: Tuple[int, ...] = (-1, 32, 64, 128)

_HUMMING_IMPORT_ERROR: Optional[BaseException] = None

try:
    from humming.forward import humming_forward as _humming_forward
    from humming.schema import BaseWeightSchema as _BaseWeightSchema
    from humming.schema import HummingInputSchema as _HummingInputSchema
    from humming.transform import (
        prepare_and_transform_humming_tensors as _prepare_and_transform_humming_tensors,
    )
except Exception as exc:  # pragma: no cover - depends on the installed wheels
    _humming_forward = None
    _BaseWeightSchema = None
    _HummingInputSchema = None
    _prepare_and_transform_humming_tensors = None
    _HUMMING_IMPORT_ERROR = exc

#: ``None`` when ``humming`` imported, otherwise the import failure. Kept under
#: the same name/shape as ``marlin_import_exception`` so callers and tests can
#: treat optional kernels uniformly.
humming_import_exception: Optional[BaseException] = _HUMMING_IMPORT_ERROR


def humming_runtime_available() -> bool:
    """Return whether the optional ``humming`` package imported successfully."""

    return humming_import_exception is None


def humming_runtime_error() -> str:
    """Return an actionable message for the current humming runtime state."""

    if humming_import_exception is not None:
        return (
            "Humming is not available: the optional `humming-kernels` package "
            "failed to import "
            f"({type(humming_import_exception).__name__}: {humming_import_exception}). "
            "Install it with `pip install humming-kernels` (CUDA 12.x/13.x, SM75+)."
        )
    if IS_ROCM:
        return (
            "Humming is CUDA-only: it ships NVIDIA SASS/PTX kernels and has no "
            "HIP port. Use a portable `*Torch`/`*Triton` backend on ROCm."
        )
    if not torch.cuda.is_available():
        return "Humming requires CUDA, but no CUDA device is available."
    return "Humming runtime is available."


def _require_humming() -> None:
    if humming_import_exception is not None:
        raise ModuleNotFoundError(humming_runtime_error())


def humming_device_capability(device: torch.device) -> Tuple[int, int]:
    """Return the SM capability of one concrete CUDA device."""

    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return int(major), int(minor)


def humming_supports_device(device: torch.device) -> bool:
    """Return whether humming can run on ``device`` (never raises)."""

    if IS_ROCM or not torch.cuda.is_available():
        return False
    target = torch.device(device)
    if target.type != "cuda":
        return False
    try:
        return humming_device_capability(target) >= HUMMING_MIN_CAPABILITY
    except Exception:
        return False


def humming_validate_device(device: torch.device | str | int) -> None:
    """Raise ``NotImplementedError`` unless humming can run on ``device``."""

    if IS_ROCM:
        raise NotImplementedError(humming_runtime_error())
    if not torch.cuda.is_available():
        raise NotImplementedError(humming_runtime_error())

    target = device if isinstance(device, torch.device) else torch.device(device)
    if target.type != "cuda":
        raise NotImplementedError(
            f"Humming requires a CUDA device, got `{target}`. "
            "Reload with a CPU/portable backend for CPU inference."
        )

    capability = humming_device_capability(target)
    if capability < HUMMING_MIN_CAPABILITY:
        raise NotImplementedError(
            f"Humming requires SM{HUMMING_MIN_CAPABILITY[0]}{HUMMING_MIN_CAPABILITY[1]}+ "
            f"(Turing or newer), but `{target}` reports SM{capability[0]}{capability[1]}."
        )


def humming_supported_device_indices() -> list[int]:
    """Return every visible CUDA ordinal humming accepts, in index order."""

    if humming_import_exception is not None or IS_ROCM or not torch.cuda.is_available():
        return []
    return [
        index
        for index in range(torch.cuda.device_count())
        if humming_supports_device(torch.device("cuda", index))
    ]


def create_humming_locks(device: torch.device | str | int) -> torch.Tensor:
    """Allocate the per-layer Stream-K synchronization buffer."""

    target = device if isinstance(device, torch.device) else torch.device(device)
    index = target.index
    if index is None:
        index = torch.cuda.current_device()
    return torch.zeros(HUMMING_LOCKS, dtype=torch.int32, device=torch.device("cuda", index))


def humming_compute_config(*, batch_invariant: bool = False, f16_accum: bool = False) -> str:
    """Return the per-call compute policy humming's GEMM expects as JSON.

    Both knobs change numerics, so they are explicit arguments rather than
    environment switches: a layer always runs the deterministic fp32-accumulate
    default, and callers that diff kernels or reproduce a reduction order opt in
    by building the policy here and passing it to :func:`humming_apply_linear`.
    """

    return json.dumps(
        {
            "use_batch_invariant": bool(batch_invariant),
            "use_f16_accum": bool(f16_accum),
            "gemm_type": "dense",
        }
    )


def _prepare_humming_tensors(
    *,
    source_tensors: Dict[str, torch.Tensor],
    weight_schema: Dict[str, Any],
    shape_n: int,
    shape_k: int,
    param_dtype: torch.dtype,
    device: torch.device,
    has_bias: bool,
):
    """Convert one checkpoint layout into transformed humming operands."""

    _require_humming()

    schema = _BaseWeightSchema.from_config(weight_schema)
    humming_schema, tensors = schema.convert_humming(
        tensors=source_tensors,
        shape_n_stacks=[shape_n],
        shape_k_stacks=[shape_k],
        param_dtype=param_dtype,
        device=device,
    )

    config, tensors = _prepare_and_transform_humming_tensors(
        tensors=tensors,
        shape_n=shape_n,
        shape_k=shape_k,
        weight_schema=humming_schema,
        input_schema=_HummingInputSchema(),
        pad_n_to_multiple=HUMMING_PAD_N_TO_MULTIPLE,
        pad_k_to_multiple=HUMMING_PAD_K_TO_MULTIPLE,
        has_bias=has_bias,
        torch_dtype=param_dtype,
        device=device,
    )
    return config, humming_schema, tensors


def prepare_humming_gptq_tensors(
    *,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    bits: int,
    group_size: int,
    sym: bool,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Convert a GPTQ checkpoint layer into transformed humming operands.

    ``qzeros`` must already carry *logical* zero points (GPTQModel's internal
    "v2" convention). ``sym=True`` ignores them entirely, matching humming's
    symmetric weight schema.
    """

    if not sym and (qzeros is None or qzeros.numel() == 0):
        raise NotImplementedError(
            "Humming GPTQ with sym=False requires the checkpoint's qzeros tensor."
        )

    source_tensors: Dict[str, torch.Tensor] = {
        "qweight": qweight,
        "scales": scales,
    }
    if not sym:
        source_tensors["qzeros"] = qzeros
    if bias is not None:
        source_tensors["bias"] = bias

    return _prepare_humming_tensors(
        source_tensors=source_tensors,
        weight_schema={
            "quant_method": "gptq",
            "bits": bits,
            "group_size": group_size,
            "desc_act": False,
            "sym": sym,
        },
        shape_n=out_features,
        shape_k=in_features,
        param_dtype=dtype,
        device=device,
        has_bias=bias is not None,
    )


def prepare_humming_awq_tensors(
    *,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Convert a classic AWQ GEMM checkpoint layer into humming operands.

    AWQ always dequantizes as ``(code - zero_point) * scale`` in GPTQModel, so
    the checkpoint's ``qzeros`` is forwarded whenever it exists. Without one the
    schema switches to humming's offset-binary mode, which is only correct for
    checkpoints whose codes are already signed.
    """

    zero_point = qzeros is not None and qzeros.numel() > 0

    source_tensors: Dict[str, torch.Tensor] = {
        "qweight": qweight,
        "scales": scales,
    }
    if zero_point:
        source_tensors["qzeros"] = qzeros
    if bias is not None:
        source_tensors["bias"] = bias

    return _prepare_humming_tensors(
        source_tensors=source_tensors,
        weight_schema={
            "quant_method": "awq",
            "bits": bits,
            "group_size": group_size,
            "zero_point": zero_point,
        },
        shape_n=out_features,
        shape_k=in_features,
        param_dtype=dtype,
        device=device,
        has_bias=bias is not None,
    )


def prepare_humming_fp8_tensors(
    *,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_method: str,
    weight_block_size: Optional[Tuple[int, int]],
    bias: Optional[torch.Tensor],
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Convert an FP8 checkpoint layer into transformed humming operands.

    ``weight`` is the stored E4M3 tensor and ``weight_scale`` must already be a
    *multiplier* (humming multiplies the codes by their scale, while GPTQModel
    checkpoints normally store ``weight_scale_inv``). Three layouts are accepted:

    * ``tensor`` -- one FP32 scalar for the whole matrix (``Fp8WeightSchema``).
    * ``row``    -- one FP32 value per output channel (humming's channel scale).
    * ``block``  -- ``[out // block_n, in // block_k]`` FP32 block scales
      (``Fp8WeightSchema`` with ``weight_block_size``).
    """

    if weight_scale_method == "tensor":
        weight_schema: Dict[str, Any] = {"quant_method": "fp8"}
        source_tensors: Dict[str, torch.Tensor] = {
            "weight": weight,
            "weight_scale": weight_scale.reshape(1).to(torch.float32),
        }
    elif weight_scale_method == "block":
        if weight_block_size is None:
            raise ValueError("humming FP8 block scaling requires `weight_block_size`.")
        block_n, block_k = weight_block_size
        weight_schema = {"quant_method": "fp8", "weight_block_size": [block_n, block_k]}
        source_tensors = {
            "weight": weight,
            "weight_scale_inv": weight_scale.to(torch.float32),
        }
    elif weight_scale_method == "row":
        # humming's generic schema covers per-output-channel FP32 scales; the
        # FP8 schema only models tensor and block layouts.
        weight_schema = {
            "quant_method": "humming",
            "b_dtype": "float8e4m3",
            "bs_dtype": "float32",
            "weight_scale_type": "channel",
        }
        source_tensors = {
            "weight": weight.view(torch.int32),
            "weight_scale": weight_scale.reshape(-1, 1).to(torch.float32),
        }
    else:
        raise ValueError(f"unsupported humming FP8 weight scale method: {weight_scale_method}")

    if bias is not None:
        source_tensors["bias"] = bias

    return _prepare_humming_tensors(
        source_tensors=source_tensors,
        weight_schema=weight_schema,
        shape_n=out_features,
        shape_k=in_features,
        param_dtype=dtype,
        device=device,
        has_bias=bias is not None,
    )


def prepare_humming_mxfp4_tensors(
    *,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Convert an OCP MXFP4 checkpoint layer into transformed humming operands.

    Both GPTQModel's MACaQ storage and humming use the same layout, so this is a
    handoff rather than a repack: nibble-packed E2M1 with the low nibble holding
    the even column, and one E8M0 exponent per 32-element block.
    """

    if weight.dtype != torch.uint8:
        raise ValueError(f"MXFP4 `weight` must be uint8, got {weight.dtype}")
    if weight_scale.dtype != torch.uint8:
        raise ValueError(f"MXFP4 `weight_scale` must be uint8 (E8M0 view), got {weight_scale.dtype}")

    source_tensors: Dict[str, torch.Tensor] = {
        "weight": weight,
        "weight_scale": weight_scale,
    }
    if bias is not None:
        source_tensors["bias"] = bias

    return _prepare_humming_tensors(
        source_tensors=source_tensors,
        weight_schema={"quant_method": "mxfp4"},
        shape_n=out_features,
        shape_k=in_features,
        param_dtype=dtype,
        device=device,
        has_bias=bias is not None,
    )


def humming_apply_linear(
    *,
    config,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    locks: Optional[torch.Tensor] = None,
    compute_config: Optional[str] = None,
) -> torch.Tensor:
    """Run one dense humming GEMM on a 2-D ``[M, K]`` activation.

    ``compute_config`` overrides the default dense/fp32 policy; pass
    ``humming_compute_config(batch_invariant=True)`` for a pinned reduction
    order. Layers use the default so inference numerics never depend on ambient
    process state.
    """

    _require_humming()
    return _humming_forward(
        config,
        inputs=inputs,
        weight=weight,
        weight_scale=weight_scale,
        zero_point=zero_point,
        bias=bias,
        locks=locks,
        compute_config=compute_config or humming_compute_config(),
    )


__all__ = [
    "HUMMING_AWQ_BITS",
    "HUMMING_GPTQ_BITS",
    "HUMMING_GROUP_SIZES",
    "HUMMING_LOCKS",
    "HUMMING_MIN_CAPABILITY",
    "HUMMING_PAD_K_TO_MULTIPLE",
    "HUMMING_PAD_N_TO_MULTIPLE",
    "create_humming_locks",
    "humming_apply_linear",
    "humming_compute_config",
    "humming_device_capability",
    "humming_import_exception",
    "humming_runtime_available",
    "humming_runtime_error",
    "humming_supported_device_indices",
    "humming_supports_device",
    "humming_validate_device",
    "prepare_humming_awq_tensors",
    "prepare_humming_fp8_tensors",
    "prepare_humming_gptq_tensors",
    "prepare_humming_mxfp4_tensors",
]
