# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import FakeTensor

# Hidden float formats the hardware supports but PTX cannot emit: quantize using
# the base PTX type below and patch the cubin's cvt to the hidden format.
_HIDDEN_BASE = {"float8e3m4": "float8e5m2", "float4e0m3": "float4e2m1"}
_HIDDEN_PATCH_MODE = {"float8e3m4": "cvt_e3m4", "float4e0m3": "cvt_e0m3"}


@triton.jit
def calc_scale(tensor, dtype):
    if dtype == "float8e4m3":
        absmax = tl.maximum(tl.max(tl.abs(tensor)), 1e-30)
        scale_x = absmax / 448
    elif dtype == "float8e5m2":
        absmax = tl.maximum(tl.max(tl.abs(tensor)), 1e-30)
        scale_x = absmax / 57344
    elif dtype == "float4e2m1":
        absmax = tl.maximum(tl.max(tl.abs(tensor)), 1e-30)
        scale_x = absmax / 6
    elif dtype == "float8e3m4":
        absmax = tl.maximum(tl.max(tl.abs(tensor)), 1e-30)
        scale_x = absmax / 30
    elif dtype == "float4e0m3":
        absmax = tl.maximum(tl.max(tl.abs(tensor)), 1e-30)
        scale_x = absmax / 7
    elif dtype == "int8":
        maxval = tl.maximum(tl.max(tensor), 1e-30)
        minval = tl.minimum(tl.min(tensor), -1e-30)
        scale_x = tl.maximum(maxval / 127, -minval / 128)
    elif dtype == "int4":
        maxval = tl.maximum(tl.max(tensor), 1e-30)
        minval = tl.minimum(tl.min(tensor), -1e-30)
        scale_x = tl.maximum(maxval / 7, -minval / 8)
    else:
        tl.static_assert(False, "unsupported dtype: " + dtype)
    return scale_x


@triton.jit
def finalize_scale(scale, scale_dtype: tl.constexpr, gs, inv_gs):
    if scale_dtype == "float8e8m0":
        scale = scale * inv_gs
        s_uint = scale.to(tl.uint32, bitcast=True)
        s_uint = (s_uint + 0x007FFFFF) & 0x7F800000
        scale = s_uint.to(tl.float32, bitcast=True)
        s_store = (s_uint >> 23).to(tl.uint8)
        scale = scale * gs
    elif scale_dtype == "float8e4m3":
        scale = scale * inv_gs
        s_store = scale.to(tl.float8e4nv)
        scale = s_store.to(tl.float32) * gs
    else:
        s_store = scale
    return scale, s_store


@triton.jit
def quant_tensor(tensor, dtype):
    if dtype == "float8e4m3":
        tensor = tensor.to(tl.float8e4nv)
    elif dtype == "float8e5m2" or dtype == "float8e3m4":
        # float8e3m4 emits the e5m2 cvt; the cubin is patched to e3m4 afterwards.
        tensor = tensor.to(tl.float8e5)
    elif dtype == "int8":
        tensor = tl.inline_asm_elementwise(
            asm="cvt.rni.s8.f32 $0, $1;",
            constraints="=r,f",
            args=[tensor],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tensor = tensor.to(tl.int8)
    else:
        tl.static_assert(False, "unsupported dtype: " + dtype)

    return tensor


@triton.jit
def quant_tensor_x2(tensor1, tensor2, dtype):
    if dtype == "int4":
        tensor = tl.inline_asm_elementwise(
            asm="""{
            .reg .s32 r1, r2, a1, a2;
            cvt.rni.s32.f32 r1, $1;
            cvt.rni.s32.f32 r2, $2;
            and.b32 a1, r1, 0xF;
            and.b32 a2, r2, 0xF;
            mad.lo.s32 $0, a2, 16, a1;
            }""",
            constraints="=r,f,f",
            args=[tensor1, tensor2],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tensor = tensor.to(tl.uint8)
    elif dtype == "float4e2m1" or dtype == "float4e0m3":
        tensor = tl.inline_asm_elementwise(
            asm="""{
            .reg .b8 t;
            cvt.rn.satfinite.e2m1x2.f32 t, $2, $1;
            cvt.u16.u8 $0, t;
            }""",
            constraints="=h,f,f",
            args=[tensor1, tensor2],
            dtype=tl.int16,
            is_pure=True,
            pack=1,
        )
        tensor = tensor.to(tl.uint8)
    else:
        tl.static_assert(False, "unsupported dtype: " + dtype)

    return tensor


@triton.jit
def _quant_tensor_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    num_blocks,
    is_dynamic: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    dtype: tl.constexpr,
    M_ROWS,
    M_MAJOR: tl.constexpr,
    SCALE_DTYPE: tl.constexpr,
    HAS_GLOBAL_SCALE: tl.constexpr,
    global_scale_ptr,
    MX_PACK: tl.constexpr,
):
    block_id = tl.program_id(0).to(tl.int64)
    tl.static_assert(N % GROUP_SIZE == 0)

    gs = 1.0
    if HAS_GLOBAL_SCALE:
        gs = tl.load(global_scale_ptr).to(tl.float32)
    inv_gs = 1.0 / gs

    row_num_blocks = N // GROUP_SIZE

    for g in tl.static_range(GROUPS_PER_BLOCK):
        group_id = block_id * GROUPS_PER_BLOCK + g
        in_range = group_id < num_blocks
        row_id = group_id // row_num_blocks
        col_block_id = group_id % row_num_blocks
        offset = row_id * stride_x + col_block_id * GROUP_SIZE
        if MX_PACK:
            # byte offset into the (num_groups/4, M_ROWS) int32 buffer: 4 K-groups
            # share a word, M is contiguous within a word-row.
            scale_off = (col_block_id // 4) * (M_ROWS * 4) + row_id * 4 + (col_block_id % 4)
        elif M_MAJOR:
            scale_off = col_block_id * M_ROWS + row_id
        else:
            scale_off = row_id * row_num_blocks + col_block_id

        if dtype == "int4" or dtype == "float4e2m1" or dtype == "float4e0m3":
            cols = tl.arange(0, BLOCK // 2)
            mask = (cols < GROUP_SIZE // 2) & in_range
            cols1 = cols * 2
            cols2 = cols * 2 + 1

            x1 = tl.load(x_ptr + (offset + cols1), mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + (offset + cols2), mask=mask, other=0.0).to(tl.float32)

            if is_dynamic:
                scale = tl.maximum(calc_scale(x1, dtype), calc_scale(x2, dtype))
                scale, s_store = finalize_scale(scale, SCALE_DTYPE, gs, inv_gs)
            else:
                scale = tl.load(scale_ptr + col_block_id, mask=in_range)
            inv_scale = 1 / scale
            x_q = quant_tensor_x2(x1 * inv_scale, x2 * inv_scale, dtype)
            tl.store(xq_ptr + group_id * GROUP_SIZE // 2 + cols, x_q, mask=mask)

            if is_dynamic:
                if MX_PACK:
                    s_store = s_store.to(tl.uint8, bitcast=True)
                tl.store(scale_ptr + scale_off, s_store, mask=in_range)
        else:
            cols = tl.arange(0, BLOCK)
            mask = (cols < GROUP_SIZE) & in_range
            x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
            if is_dynamic:
                scale = calc_scale(x, dtype)
                scale, s_store = finalize_scale(scale, SCALE_DTYPE, gs, inv_gs)
            else:
                scale = tl.load(scale_ptr + col_block_id, mask=in_range)
            inv_scale = 1 / scale
            x_q = quant_tensor(x * inv_scale, dtype)
            tl.store(xq_ptr + group_id * GROUP_SIZE + cols, x_q, mask=mask)

            if is_dynamic:
                if MX_PACK:
                    s_store = s_store.to(tl.uint8, bitcast=True)
                tl.store(scale_ptr + scale_off, s_store, mask=in_range)


def quant_input(
    inputs: torch.Tensor,
    dtype: str,
    scales: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    group_size: int | None = None,
    m_major_scale: bool = False,
    scale_dtype: str = "float32",
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    storage_dtype = _HIDDEN_BASE.get(dtype, dtype)
    if storage_dtype in ["int4", "float4e2m1"]:
        output_shape = (*inputs.shape[:-1], inputs.size(-1) // 2)
        output_dtype = torch.uint8
    elif storage_dtype == "int8":
        output_shape = inputs.shape
        output_dtype = torch.int8
    elif storage_dtype == "float8e4m3":
        output_shape = inputs.shape
        output_dtype = torch.float8_e4m3fn
    elif storage_dtype == "float8e5m2":
        output_shape = inputs.shape
        output_dtype = torch.float8_e5m2
    else:
        raise ValueError("unsupported dtype: " + dtype)

    if outputs is None:
        outputs = torch.empty(output_shape, dtype=output_dtype, device=inputs.device)

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert outputs.shape == output_shape
    assert outputs.dtype == output_dtype
    assert outputs.device == inputs.device

    is_dynamic = scales is None
    inputs = inputs.view(-1, inputs.size(-1))
    if group_size is None or group_size == 0:
        group_size = inputs.size(1)
    assert inputs.size(1) % group_size == 0
    num_blocks = inputs.nelement() // group_size

    m_rows = inputs.size(0)
    num_groups = inputs.size(1) // group_size
    m_rows_stride = (m_rows + 3) // 4 * 4 if m_major_scale else m_rows
    if m_major_scale:
        assert is_dynamic, "m_major_scale requires dynamic quantization"

    mx_pack_m_major = m_major_scale and scale_dtype in ("float8e8m0", "float8e4m3") and is_dynamic
    num_groups_packed = (num_groups + 3) // 4

    if scale_dtype == "float32":
        scale_torch_dtype = torch.float32
    elif scale_dtype == "float8e4m3":
        scale_torch_dtype = torch.float8_e4m3fn
    elif scale_dtype == "float8e8m0":
        scale_torch_dtype = torch.uint8
    else:
        raise ValueError("unsupported scale_dtype: " + scale_dtype)

    if scale_dtype != "float32":
        assert is_dynamic, "non-float32 scale_dtype requires dynamic quantization"

    has_global_scale = global_scale is not None

    scales_packed = None
    if is_dynamic:
        if mx_pack_m_major:
            scales_packed = torch.empty(
                (num_groups_packed, m_rows_stride),
                dtype=torch.int32,
                device=inputs.device,
            )
            scales = scales_packed.view(torch.uint8)
        else:
            shape = (num_groups, m_rows_stride) if m_major_scale else (m_rows, num_groups)
            scales = torch.empty(shape, dtype=scale_torch_dtype, device=inputs.device)

    if not isinstance(inputs, FakeTensor):
        assert inputs.is_cuda
        BLOCK = triton.next_power_of_2(group_size)
        # Merge multiple groups per block to reduce scheduling overhead.
        groups_per_block = 1
        if group_size <= 256 and num_blocks >= 131072:
            # Small group_size (e.g. 128) with massive block count
            groups_per_block = min(1024 // group_size, num_blocks)
        grid_blocks = (num_blocks + groups_per_block - 1) // groups_per_block
        packed = storage_dtype in ("int4", "float4e2m1")
        effective_block = BLOCK // 2 if packed else BLOCK
        num_warps = min(max(effective_block // 256, 1), 8)
        num_stages = 1

        launch_args = (
            inputs,
            outputs,
            scales,
            inputs.stride(0),
            num_blocks,
            is_dynamic,
            inputs.size(1),
            group_size,
            BLOCK,
            groups_per_block,
            dtype,
            m_rows_stride,
            m_major_scale,
            scale_dtype,
            has_global_scale,
            global_scale,
            mx_pack_m_major,
        )
        launch_kwargs = dict(num_warps=num_warps, num_stages=num_stages)

        patch_mode = _HIDDEN_PATCH_MODE.get(dtype)
        if patch_mode is not None:
            from ..utils.cubin import triton_warmup_and_patch

            triton_warmup_and_patch(
                _quant_tensor_kernel,
                *launch_args,
                mode=patch_mode,
                grid=(grid_blocks,),
                **launch_kwargs,
            )

        _quant_tensor_kernel[(grid_blocks,)](*launch_args, **launch_kwargs)

    if scales is None:
        scales = torch.empty(0)
    elif mx_pack_m_major:
        scales = scales_packed
    elif m_major_scale:
        scales = scales.view(num_groups, m_rows_stride)
    else:
        scales = scales.view(*outputs.shape[:-1], scales.size(-1))

    if scale_dtype == "float8e8m0" and scales.numel() > 0 and not mx_pack_m_major:
        scales = scales.view(torch.float8_e8m0fnu)

    # Hidden formats are raw bytes reinterpreted by the MMA; expose them as uint8.
    if dtype in _HIDDEN_BASE and outputs.dtype != torch.uint8:
        outputs = outputs.view(torch.uint8)

    return outputs, scales
