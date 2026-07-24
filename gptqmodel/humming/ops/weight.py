# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import math

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .. import dtypes
from ..kernel.dequant_weight import DequantKernel
from ..kernel.pack_weight import PackWeightKernel
from ..kernel.process_mxfp4 import ProcessMxfp4W4A8Kernel
from ..kernel.quant_weight import QuantWeightKernel
from ..kernel.repack_weight import RepackWeightKernel
from ..kernel.unpack_weight import UnpackWeightKernel


def dequant_weight(
    inputs: torch.Tensor,
    exponent_bits: int,
    mantissa_bits: int,
    is_signed: bool,
) -> torch.Tensor:
    assert inputs.dtype == torch.int32
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    outputs = torch.empty_like(inputs, dtype=torch.float32)

    if not isinstance(inputs, FakeTensor):
        kernel = DequantKernel()
        kernel(
            inputs=inputs,
            outputs=outputs,
            exponent_bits=exponent_bits,
            mantissa_bits=mantissa_bits,
            is_signed=is_signed,
        )

    return outputs


def pack_weight(inputs: torch.Tensor, num_bits: int) -> torch.Tensor:
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert inputs.size(-1) % 32 == 0
    assert inputs.size(-1) * num_bits % 32 == 0
    assert inputs.dtype == torch.int32

    output_shape = inputs.shape[:-1] + (inputs.size(-1) * num_bits // 32,)
    outputs = torch.empty(output_shape, dtype=torch.int32, device=inputs.device)

    if not isinstance(inputs, FakeTensor):
        kernel = PackWeightKernel(num_bits=num_bits)
        kernel(inputs=inputs, outputs=outputs)

    return outputs


def quant_weight(
    inputs: torch.Tensor,
    source_dtype_str: str,
    target_dtype_str: str,
    group_size: int,
    has_scale: bool,
    use_e8m0_scale: bool,
    has_zero_point: bool,
    is_fp_zero_point: bool,
    allow_negative_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size = inputs.size(-1) if group_size <= 0 else group_size
    source_dtype = dtypes.DataType.from_str(source_dtype_str)
    target_dtype = dtypes.DataType.from_str(target_dtype_str)

    assert inputs.is_cuda
    assert inputs.is_contiguous()
    outputs = torch.empty_like(inputs, dtype=torch.int32)

    if has_scale:
        scale_shape = inputs.shape[:-1] + (inputs.size(-1) // group_size,)
        scale_dtype = torch.float8_e8m0fnu if use_e8m0_scale else torch.float32
        scales = torch.empty(scale_shape, device=inputs.device, dtype=scale_dtype)
        zero_point = torch.empty(scale_shape, device=inputs.device, dtype=torch.int32)
    else:
        scales = torch.empty(0)

    if has_scale and has_zero_point:
        dtype = torch.float32 if is_fp_zero_point else torch.int32
        zero_point = torch.empty(scale_shape, device=inputs.device, dtype=dtype)
    else:
        zero_point = torch.empty(0)

    if not isinstance(inputs, FakeTensor):
        kernel = QuantWeightKernel(
            source_dtype=source_dtype,
            target_dtype=target_dtype,
            group_size=group_size,
            has_scale=has_scale,
            has_zero_point=has_zero_point,
            use_e8m0_scale=use_e8m0_scale,
            is_fp_zero_point=is_fp_zero_point,
            allow_negative_scale=allow_negative_scale,
        )
        kernel(inputs=inputs, outputs=outputs, scales=scales, zero_point=zero_point)

    return outputs, scales, zero_point


def repack_weight(
    inputs: torch.Tensor,
    weight_bits: int,
    activation_bits: int,
    is_weight_packed: bool,
    should_preprocess_for_int2fp: bool = False,
    should_preprocess_with_zp: bool = False,
    use_wgmma: bool = False,
    use_fused_e8m0_scale: bool = False,
    interleave_mode: int = 3,
    group_size_zp: int = 0,
    padded_shape_n: int | None = None,
    padded_shape_k: int | None = None,
    zero_point: torch.Tensor | None = None,
    use_packed_k_layout: bool = False,
) -> torch.Tensor:
    assert inputs.ndim in [2, 3]
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert inputs.dtype == torch.int32
    device = inputs.device
    num_experts = 1 if inputs.ndim == 2 else inputs.size(0)
    shape_n = inputs.size(-2)
    shape_k = inputs.size(-1)
    if is_weight_packed:
        assert shape_k * 32 % weight_bits == 0
        shape_k = shape_k * 32 // weight_bits

    if should_preprocess_with_zp:
        assert zero_point is not None and zero_point.dtype == torch.int32
        group_size_zp = shape_k if group_size_zp == 0 else group_size_zp
        zero_point_shape = inputs.shape[:-1] + (math.ceil(shape_k / group_size_zp),)

        if is_weight_packed:
            assert shape_n * weight_bits % 32 == 0
            packed_shape_n = shape_n * weight_bits // 32
            zero_point_shape = zero_point_shape[:-2] + (packed_shape_n,) + zero_point_shape[-1:]

        assert zero_point.shape == zero_point_shape

    pack_size_k = 64 if use_packed_k_layout else 256 // activation_bits
    output_shape: tuple[int, ...] = (
        shape_k // pack_size_k,
        shape_n * pack_size_k * weight_bits // 32,
    )
    if inputs.ndim == 3:
        output_shape = (num_experts,) + output_shape

    outputs = torch.empty(output_shape, dtype=torch.int32, device=device)

    if not isinstance(inputs, FakeTensor):
        kernel = RepackWeightKernel(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            is_weight_packed=is_weight_packed,
            should_preprocess_for_int2fp=should_preprocess_for_int2fp,
            should_preprocess_with_zp=should_preprocess_with_zp,
            use_wgmma=use_wgmma,
            use_fused_e8m0_scale=use_fused_e8m0_scale,
            group_size_zp=group_size_zp,
            use_packed_k_layout=use_packed_k_layout,
        )

        kernel(
            inputs=inputs,
            outputs=outputs,
            zero_point=zero_point,
            padded_shape_n=padded_shape_n,
            padded_shape_k=padded_shape_k,
            interleave_mode=interleave_mode,
        )

    return outputs


def unpack_weight(inputs: torch.Tensor, num_bits: int) -> torch.Tensor:
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert inputs.size(-1) % num_bits == 0
    assert inputs.dtype == torch.int32

    shape_k = inputs.size(-1) // num_bits * 32
    output_shape = inputs.shape[:-1] + (shape_k,)
    outputs = torch.empty(output_shape, dtype=torch.int32, device=inputs.device)

    if not isinstance(inputs, FakeTensor):
        kernel = UnpackWeightKernel(num_bits=num_bits)
        kernel(inputs=inputs, outputs=outputs)

    return outputs


def process_mxfp4_w4a8_weight(
    inputs: torch.Tensor,
    delta_scale_offsets: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    assert inputs.dtype == torch.int32
    assert delta_scale_offsets.dtype == torch.uint8
    assert inputs.nelement() >= delta_scale_offsets.nelement() * 4

    def is_power_of_two(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    repeat_count = inputs.nelement() // delta_scale_offsets.nelement() // 4
    assert is_power_of_two(repeat_count)
    delta_scale_offsets = delta_scale_offsets.repeat_interleave(repeat_count, -1)

    outputs = inputs if inplace else torch.empty_like(inputs)

    kernel = ProcessMxfp4W4A8Kernel()
    kernel(inputs, outputs, delta_scale_offsets)

    return outputs
