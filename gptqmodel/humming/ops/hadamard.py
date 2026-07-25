# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .. import dtypes
from ..kernel.hadamard import HadamardKernel
from ..kernel.hadamard_quant import HadamardQuantInputKernel
from ..kernel.hadamard_quant_wide import HadamardQuantInputWideKernel

_QUANT_DTYPE_STR_TO_TORCH = {
    "int8": torch.int8,
    "int4": torch.uint8,
    "float4e2m1": torch.uint8,
    "float4e0m3": torch.uint8,
    "float8e4m3": torch.float8_e4m3fn,
    "float8e3m4": torch.uint8,
    "float8e5m2": torch.float8_e5m2,
}

_SCALE_DTYPE_TO_TORCH = {
    "float32": torch.float32,
    "float8e4m3": torch.float8_e4m3fn,
    "float8e8m0": torch.uint8,
}


def hadamard_transform(
    inputs: torch.Tensor,
    block_size: int,
    scale: float = 1.0,
    outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a normalized Walsh-Hadamard transform along the last dimension.

    For each contiguous chunk of ``block_size`` elements along the last
    dimension, computes ``y = (x @ H_N) * (rsqrt(N) * scale)`` where ``H_N`` is
    the Sylvester Hadamard matrix of order ``N = block_size``.

    Args:
        inputs: tensor of shape ``[..., K]``. ``K`` must be a multiple of
            ``block_size``. dtype must be one of float16/bfloat16/float32.
        block_size: transform length ``N``, a power of two in ``[2, 4096]``.
        scale: additional scalar multiplier applied after normalization.
        outputs: optional preallocated output tensor (same shape/dtype).
    """
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert block_size >= 2 and (block_size & (block_size - 1)) == 0, (
        f"block_size must be a power of 2 >= 2, got {block_size}"
    )
    assert inputs.size(-1) % block_size == 0, (
        f"last dim {inputs.size(-1)} must be divisible by block_size {block_size}"
    )
    assert inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)

    if outputs is None:
        outputs = torch.empty_like(inputs)
    else:
        assert outputs.shape == inputs.shape
        assert outputs.dtype == inputs.dtype
        assert outputs.is_contiguous()

    if not isinstance(inputs, FakeTensor):
        kernel = HadamardKernel(
            torch_dtype=inputs.dtype,
            block_size=block_size,
            has_scale=(scale != 1.0),
        )
        kernel(inputs=inputs, outputs=outputs, extra_scale=scale)

    return outputs


def hadamard_quant_input(
    inputs: torch.Tensor,
    block_size: int,
    quant_dtype: str,
    group_size: int | None = None,
    scale: float = 1.0,
    outputs: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    m_major_scale: bool = False,
    scale_dtype: str = "float32",
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Walsh-Hadamard transform + per-group symmetric quantization.

    Equivalent to::

        y = hadamard_transform(inputs, block_size, scale=scale)
        q, s = quant_input(y, quant_dtype, group_size=group_size)

    but in a single kernel.

    Args:
        inputs: shape ``[..., K]``, dtype fp16/bf16/fp32. ``K`` must be a
            multiple of ``block_size``.
        block_size: FHT length ``N``, power of 2 in [2, 4096].
        quant_dtype: one of ``"int8"``, ``"int4"``, ``"float4e2m1"``,
            ``"float4e0m3"``, ``"float8e4m3"``, ``"float8e3m4"``,
            ``"float8e5m2"``.
        group_size: per-group quantization size. Must divide ``block_size``
            (or be a multiple of it). ``None`` or ``0`` means channelwise
            (= ``inputs.size(-1)``).
        scale: extra scalar absorbed into the returned ``scales``.
    """
    assert inputs.is_cuda and inputs.is_contiguous()
    assert inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert block_size >= 2 and (block_size & (block_size - 1)) == 0, (
        "block_size must be a power of 2 >= 2; for no-rotation quant use ops.quant_input"
    )
    assert inputs.size(-1) % block_size == 0
    if group_size is None or group_size == 0:
        group_size = inputs.size(-1)
    assert group_size >= 1
    if group_size > block_size:
        assert group_size % block_size == 0, "block_size must divide group_size"
    else:
        assert (group_size & (group_size - 1)) == 0, "group_size must be a power of 2 when <= block_size"
        assert block_size % group_size == 0, "group_size must divide block_size"
    assert quant_dtype in _QUANT_DTYPE_STR_TO_TORCH, f"unsupported quant_dtype: {quant_dtype}"
    assert scale_dtype in _SCALE_DTYPE_TO_TORCH, f"unsupported scale_dtype: {scale_dtype}"

    out_torch_dtype = _QUANT_DTYPE_STR_TO_TORCH[quant_dtype]
    scale_torch_dtype = _SCALE_DTYPE_TO_TORCH[scale_dtype]
    last_dim = inputs.size(-1)

    if quant_dtype in ("int4", "float4e2m1", "float4e0m3"):
        assert last_dim % 2 == 0
        out_shape = inputs.shape[:-1] + (last_dim // 2,)
    else:
        out_shape = inputs.shape

    num_groups_total = last_dim // group_size
    mx_pack = m_major_scale and scale_dtype == "float8e8m0"
    if m_major_scale:
        m_pad = (inputs.numel() // last_dim + 3) // 4 * 4
        scales_shape = ((num_groups_total + 3) // 4, m_pad) if mx_pack else (num_groups_total, m_pad)
    else:
        scales_shape = inputs.shape[:-1] + (num_groups_total,)
    if outputs is None:
        outputs = torch.empty(out_shape, dtype=out_torch_dtype, device=inputs.device)
    else:
        assert outputs.shape == out_shape
        assert outputs.dtype == out_torch_dtype
        assert outputs.device == inputs.device
        assert outputs.is_contiguous()
    if scales is None:
        if mx_pack:
            scales = torch.empty(scales_shape, dtype=torch.int32, device=inputs.device)
        else:
            scales = torch.empty(scales_shape, dtype=scale_torch_dtype, device=inputs.device)
    else:
        assert scales.shape == scales_shape
        assert scales.dtype == (torch.int32 if mx_pack else scale_torch_dtype)
        assert scales.device == inputs.device
        assert scales.is_contiguous()

    if global_scale is not None:
        assert global_scale.dtype == torch.float32
        assert global_scale.device == inputs.device

    if not isinstance(inputs, FakeTensor):
        target_dt = dtypes.DataType.from_str(quant_dtype)
        if group_size > block_size:
            kernel = HadamardQuantInputWideKernel(
                source_torch_dtype=inputs.dtype,
                target_dtype=target_dt,
                block_size=block_size,
                group_size=group_size,
                has_extra_scale=(scale != 1.0),
                m_major=m_major_scale,
                scale_dtype=scale_dtype,
                has_global_scale=(global_scale is not None),
            )
        else:
            kernel = HadamardQuantInputKernel(
                source_torch_dtype=inputs.dtype,
                target_dtype=target_dt,
                block_size=block_size,
                group_size=group_size,
                has_extra_scale=(scale != 1.0),
                m_major=m_major_scale,
                scale_dtype=scale_dtype,
                has_global_scale=(global_scale is not None),
            )
        kernel(
            inputs=inputs,
            outputs=outputs,
            scales=scales.view(torch.uint8) if mx_pack else scales,
            extra_scale=scale,
            global_scale=global_scale,
        )

    if scale_dtype == "float8e8m0" and not mx_pack:
        scales = scales.view(torch.float8_e8m0fnu)

    return outputs, scales
