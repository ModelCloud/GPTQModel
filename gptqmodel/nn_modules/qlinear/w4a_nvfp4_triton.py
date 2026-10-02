# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GB10 NVFP4 activation packing with fused hardware scale swizzling."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_E4M3_SEARCH_RADIUS = tl.constexpr(8)


@triton.jit
def _e2m1_codes(x):
    magnitude = tl.minimum(tl.abs(x), 6.0)
    code = tl.full(magnitude.shape, 0, tl.uint8)
    code = tl.where(magnitude > 0.25, 1, code)
    code = tl.where(magnitude >= 0.75, 2, code)
    code = tl.where(magnitude > 1.25, 3, code)
    code = tl.where(magnitude >= 1.75, 4, code)
    code = tl.where(magnitude > 2.5, 5, code)
    code = tl.where(magnitude >= 3.5, 6, code)
    code = tl.where(magnitude > 5.0, 7, code)
    return code | (tl.cast(x < 0, tl.uint8) << 3)


@triton.jit
def _e2m1_values(codes):
    magnitude = codes & 7
    value = tl.full(codes.shape, 0.0, tl.float32)
    value = tl.where(magnitude == 1, 0.5, value)
    value = tl.where(magnitude == 2, 1.0, value)
    value = tl.where(magnitude == 3, 1.5, value)
    value = tl.where(magnitude == 4, 2.0, value)
    value = tl.where(magnitude == 5, 3.0, value)
    value = tl.where(magnitude == 6, 4.0, value)
    value = tl.where(magnitude == 7, 6.0, value)
    return tl.where((codes & 8) != 0, -value, value)


@triton.jit
def _quantize_nvfp4_input_four_six(X, GlobalScale, Packed, SwizzledScales,
                                   M: tl.constexpr, K: tl.constexpr,
                                   PADDED_M: tl.constexpr):
    """Legacy version-2 packer retained for checkpoints without a recipe."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    block = tl.arange(0, 8)
    element = tl.arange(0, 16)
    x = tl.load(X + row * K + group * 128 + block[:, None] * 16 + element[None, :],
                row < M, other=0).to(tl.float32)
    global_scale = tl.load(GlobalScale).to(tl.float32)
    maximum = tl.max(tl.abs(x), axis=1)
    inverse_four = tl.minimum(tl.maximum(tl.div_rn(maximum, 4.0 * global_scale), 2.0**-9), 448.0)
    inverse_six = tl.minimum(tl.maximum(tl.div_rn(maximum, 6.0 * global_scale), 2.0**-9), 448.0)
    inverse_four = tl.where(maximum > 0, inverse_four, 1.0)
    inverse_six = tl.where(maximum > 0, inverse_six, 1.0)
    local_four = inverse_four.to(tl.float8e4nv)
    local_six = inverse_six.to(tl.float8e4nv)
    scale_four = local_four.to(tl.float32)[:, None] * global_scale
    scale_six = local_six.to(tl.float32)[:, None] * global_scale
    codes_four = _e2m1_codes(tl.div_rn(x, scale_four))
    codes_six = _e2m1_codes(tl.div_rn(x, scale_six))
    delta_four = x - _e2m1_values(codes_four) * scale_four
    delta_six = x - _e2m1_values(codes_six) * scale_six
    use_four = tl.sum(delta_four * delta_four, axis=1) <= tl.sum(delta_six * delta_six, axis=1)
    local = tl.where(use_four, local_four, local_six)
    codes = tl.where(use_four[:, None], codes_four, codes_six)
    low, high = tl.split(tl.reshape(codes, (64, 2)))
    offset = tl.arange(0, 64)
    tl.store(Packed + row * (K // 2) + group * 64 + offset, low | (high << 4), row < M)
    scale_offset = (
        group * PADDED_M * 8 + (row // 128) * 1024 + (block // 4) * 512
        + (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4
    )
    tl.store(SwizzledScales + scale_offset, tl.where(row < M, local, 0.0))


@triton.jit
def _quantize_nvfp4_input_nvidia(X, GlobalScale, Packed, SwizzledScales,
                                 M: tl.constexpr, K: tl.constexpr,
                                 PADDED_M: tl.constexpr):
    """Exact NVIDIA/TorchAO max-scaled E2M1 plus E4M3 block recipe."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    block = tl.arange(0, 8)
    element = tl.arange(0, 16)
    x = tl.load(X + row * K + group * 128 + block[:, None] * 16 + element[None, :],
                row < M, other=0).to(tl.float32)
    global_scale = tl.load(GlobalScale).to(tl.float32)
    maximum = tl.max(tl.abs(x), axis=1)
    inverse = tl.minimum(
        tl.maximum(tl.div_rn(maximum, 6.0 * global_scale), 2.0**-9), 448.0
    )
    inverse = tl.where(maximum > 0, inverse, 1.0)
    local = inverse.to(tl.float8e4nv)
    codes = _e2m1_codes(tl.div_rn(x, local.to(tl.float32)[:, None] * global_scale))
    low, high = tl.split(tl.reshape(codes, (64, 2)))
    offset = tl.arange(0, 64)
    tl.store(Packed + row * (K // 2) + group * 64 + offset, low | (high << 4), row < M)
    scale_offset = (
        group * PADDED_M * 8 + (row // 128) * 1024 + (block // 4) * 512
        + (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4
    )
    tl.store(SwizzledScales + scale_offset, tl.where(row < M, local, 0.0))


@triton.jit
def _quantize_nvfp4_input(X, GlobalScale, Packed, SwizzledScales,
                          M: tl.constexpr, K: tl.constexpr, PADDED_M: tl.constexpr,
                          GRID_SEARCH: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    block = tl.arange(0, 8)
    element = tl.arange(0, 16)
    x = tl.load(X + row * K + group * 128 + block[:, None] * 16 + element[None, :],
                row < M, other=0).to(tl.float32)
    global_scale = tl.load(GlobalScale).to(tl.float32)
    maximum = tl.max(tl.abs(x), axis=1)
    inverse_four = tl.minimum(tl.maximum(tl.div_rn(maximum, 4.0 * global_scale), 2.0**-9), 448.0)
    inverse_six = tl.minimum(tl.maximum(tl.div_rn(maximum, 6.0 * global_scale), 2.0**-9), 448.0)
    inverse_four = tl.where(maximum > 0, inverse_four, 1.0)
    inverse_six = tl.where(maximum > 0, inverse_six, 1.0)
    local_four = inverse_four.to(tl.float8e4nv)
    local_six = inverse_six.to(tl.float8e4nv)
    scale_four = local_four.to(tl.float32)[:, None] * global_scale
    scale_six = local_six.to(tl.float32)[:, None] * global_scale
    codes_four = _e2m1_codes(tl.div_rn(x, scale_four))
    codes_six = _e2m1_codes(tl.div_rn(x, scale_six))
    delta_four = x - _e2m1_values(codes_four) * scale_four
    delta_six = x - _e2m1_values(codes_six) * scale_six
    error_four = tl.sum(delta_four * delta_four, axis=1)
    error_six = tl.sum(delta_six * delta_six, axis=1)
    use_four = error_four <= error_six
    local = tl.where(use_four, local_four, local_six)
    codes = tl.where(use_four[:, None], codes_four, codes_six)
    error = tl.where(use_four, error_four, error_six)

    # Refine each Four-Over-Six code assignment with its least-squares scalar,
    # rounded back to a real E4M3 hardware scale.  Keep only lower-SSE results.
    values_four = _e2m1_values(codes_four)
    denominator_four = tl.sum(values_four * values_four, axis=1)
    optimal_four = tl.sum(x * values_four, axis=1) / tl.maximum(denominator_four, 1.0)
    inverse_four_refined = tl.minimum(
        tl.maximum(optimal_four / global_scale, 2.0**-9), 448.0
    )
    inverse_four_refined = tl.where(denominator_four > 0, inverse_four_refined, 1.0)
    local_four_refined = inverse_four_refined.to(tl.float8e4nv)
    scale_four_refined = local_four_refined.to(tl.float32)[:, None] * global_scale
    codes_four_refined = _e2m1_codes(tl.div_rn(x, scale_four_refined))
    delta_four_refined = x - _e2m1_values(codes_four_refined) * scale_four_refined
    error_four_refined = tl.sum(delta_four_refined * delta_four_refined, axis=1)
    use_refined = error_four_refined < error
    local = tl.where(use_refined, local_four_refined, local)
    codes = tl.where(use_refined[:, None], codes_four_refined, codes)
    error = tl.where(use_refined, error_four_refined, error)

    values_six = _e2m1_values(codes_six)
    denominator_six = tl.sum(values_six * values_six, axis=1)
    optimal_six = tl.sum(x * values_six, axis=1) / tl.maximum(denominator_six, 1.0)
    inverse_six_refined = tl.minimum(
        tl.maximum(optimal_six / global_scale, 2.0**-9), 448.0
    )
    inverse_six_refined = tl.where(denominator_six > 0, inverse_six_refined, 1.0)
    local_six_refined = inverse_six_refined.to(tl.float8e4nv)
    scale_six_refined = local_six_refined.to(tl.float32)[:, None] * global_scale
    codes_six_refined = _e2m1_codes(tl.div_rn(x, scale_six_refined))
    delta_six_refined = x - _e2m1_values(codes_six_refined) * scale_six_refined
    error_six_refined = tl.sum(delta_six_refined * delta_six_refined, axis=1)
    use_refined = error_six_refined < error
    local = tl.where(use_refined, local_six_refined, local)
    codes = tl.where(use_refined[:, None], codes_six_refined, codes)

    # One final coordinate step from the best assignment catches a code change
    # introduced by either branch while preserving monotone block SSE.
    values = _e2m1_values(codes)
    denominator = tl.sum(values * values, axis=1)
    optimal = tl.sum(x * values, axis=1) / tl.maximum(denominator, 1.0)
    inverse_refined = tl.minimum(tl.maximum(optimal / global_scale, 2.0**-9), 448.0)
    inverse_refined = tl.where(denominator > 0, inverse_refined, 1.0)
    local_refined = inverse_refined.to(tl.float8e4nv)
    scale_refined = local_refined.to(tl.float32)[:, None] * global_scale
    codes_refined = _e2m1_codes(tl.div_rn(x, scale_refined))
    delta_refined = x - _e2m1_values(codes_refined) * scale_refined
    error_refined = tl.sum(delta_refined * delta_refined, axis=1)
    use_refined = error_refined < tl.minimum(error, error_six_refined)
    local = tl.where(use_refined, local_refined, local)
    codes = tl.where(use_refined[:, None], codes_refined, codes)

    if GRID_SEARCH:
        # Search the bounded neighborhood of the coordinate-refined scale on
        # the real positive E4M3 bit grid. Positive finite E4M3FN bit patterns
        # 1..126 are monotonic hardware scales.
        scale = local.to(tl.float32)[:, None] * global_scale
        delta = x - _e2m1_values(codes) * scale
        error = tl.sum(delta * delta, axis=1)
        center = local.to(tl.uint8, bitcast=True).to(tl.int32)
        for bit_offset in tl.static_range(-_E4M3_SEARCH_RADIUS,
                                          _E4M3_SEARCH_RADIUS + 1):
            candidate_bits = tl.minimum(tl.maximum(center + bit_offset, 1), 126).to(tl.uint8)
            candidate = candidate_bits.to(tl.float8e4nv, bitcast=True)
            candidate_scale = candidate.to(tl.float32)[:, None] * global_scale
            candidate_codes = _e2m1_codes(tl.div_rn(x, candidate_scale))
            candidate_delta = x - _e2m1_values(candidate_codes) * candidate_scale
            candidate_error = tl.sum(candidate_delta * candidate_delta, axis=1)
            use_candidate = candidate_error < error
            error = tl.where(use_candidate, candidate_error, error)
            local = tl.where(use_candidate, candidate, local)
            codes = tl.where(use_candidate[:, None], candidate_codes, codes)
    low, high = tl.split(tl.reshape(codes, (64, 2)))
    packed = low | (high << 4)
    offset = tl.arange(0, 64)
    tl.store(Packed + row * (K // 2) + group * 64 + offset, packed, row < M)

    # Inverse of NVIDIA SWIZZLE_32_4_4 for a [padded_rows, 8] group matrix.
    scale_offset = (
        group * PADDED_M * 8 + (row // 128) * 1024 + (block // 4) * 512
        + (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4
    )
    tl.store(SwizzledScales + scale_offset, tl.where(row < M, local, 0.0))


def nvfp4_pack_and_swizzle(x: torch.Tensor, global_scale: torch.Tensor, *, recipe: str = "least_squares"
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce packed E2M1 input and all per-group swizzled E4M3 scales."""
    from ...quantization.activation_floatx import normalize_nvfp4_recipe

    recipe = normalize_nvfp4_recipe(recipe)
    if x.ndim != 2 or x.shape[1] % 128 or x.device.type != "cuda":
        raise ValueError("NVFP4 packing expects CUDA [tokens, K] with K divisible by 128.")
    if recipe not in {
        "nvidia", "nvidia_headroom", "four_six", "least_squares", "least_squares_headroom", "least_squares_grid"
    }:
        raise ValueError(f"Unsupported NVFP4 scale recipe: {recipe}.")
    rows, width = x.shape
    if rows == 0:
        return (torch.empty((0, width // 2), device=x.device, dtype=torch.float4_e2m1fn_x2),
                torch.empty((width // 128, 0), device=x.device, dtype=torch.float8_e4m3fn))
    padded_rows = triton.cdiv(rows, 128) * 128
    packed = torch.empty((rows, width // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((width // 128, padded_rows * 8), device=x.device, dtype=torch.float8_e4m3fn)
    kernel = {
        "nvidia": _quantize_nvfp4_input_nvidia,
        "nvidia_headroom": _quantize_nvfp4_input_nvidia,
        "four_six": _quantize_nvfp4_input_four_six,
        "least_squares": _quantize_nvfp4_input,
        "least_squares_headroom": _quantize_nvfp4_input,
        "least_squares_grid": _quantize_nvfp4_input,
    }[recipe]
    if recipe in {"least_squares", "least_squares_headroom", "least_squares_grid"}:
        kernel[(padded_rows, width // 128)](
            x, global_scale, packed, scales, rows, width, padded_rows,
            GRID_SEARCH=recipe == "least_squares_grid", num_warps=4,
        )
    else:
        kernel[(padded_rows, width // 128)](
            x, global_scale, packed, scales, rows, width, padded_rows,
            num_warps=4,
        )
    return packed.view(torch.float4_e2m1fn_x2), scales


@triton.jit
def _decode_nvfp4(Packed, Scales, GlobalScale, Out,
                  M: tl.constexpr, K: tl.constexpr, PADDED_M: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    element = tl.arange(0, 128)
    block = element // 16
    encoded = tl.load(Packed + row * (K // 2) + group * 64 + element // 2)
    code = (encoded >> ((element % 2) * 4)) & 15
    magnitude = code & 7
    value = tl.full((128,), 0.0, tl.float32)
    value = tl.where(magnitude == 1, 0.5, value)
    value = tl.where(magnitude == 2, 1.0, value)
    value = tl.where(magnitude == 3, 1.5, value)
    value = tl.where(magnitude == 4, 2.0, value)
    value = tl.where(magnitude == 5, 3.0, value)
    value = tl.where(magnitude == 6, 4.0, value)
    value = tl.where(magnitude == 7, 6.0, value)
    value = tl.where((code & 8) != 0, -value, value)
    scale_offset = (
        group * PADDED_M * 8 + (row // 128) * 1024 + (block // 4) * 512
        + (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4
    )
    local = tl.load(Scales + scale_offset).to(tl.float32)
    global_scale = tl.load(GlobalScale).to(tl.float32)
    tl.store(Out + row * K + group * 128 + element, value * local * global_scale)


def nvfp4_decode(codes: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor,
                 width: int, output_dtype: torch.dtype) -> torch.Tensor:
    """Decode packed E2M1 and SWIZZLE_32_4_4 scales at a nonlinear consumer."""
    if codes.ndim != 2 or codes.dtype != torch.float4_e2m1fn_x2 or codes.shape[1] * 2 != width:
        raise ValueError("Malformed packed NVFP4 operand.")
    rows = codes.shape[0]
    padded_rows = triton.cdiv(rows, 128) * 128
    if scales.dtype != torch.float8_e4m3fn or scales.shape != (width // 128, padded_rows * 8):
        raise ValueError("Malformed NVFP4 block scales.")
    output = torch.empty((rows, width), device=codes.device, dtype=output_dtype)
    if rows:
        _decode_nvfp4[(rows, width // 128)](
            codes.view(torch.uint8), scales, global_scale, output, rows, width, padded_rows, num_warps=4,
        )
    return output


@triton.jit
def _accumulate_fp4_planes(Both, GroupScales, GlobalScale, TokenScale, Bias, Accumulator, Output,
                           M: tl.constexpr, N: tl.constexpr, FIRST: tl.constexpr,
                           LAST: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_TOKEN_SCALE: tl.constexpr,
                           BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < M * N
    plane_offset = (index // N) * (2 * N) + index % N
    low = tl.load(Both + plane_offset, active, other=0).to(tl.float32)
    high = tl.load(Both + plane_offset + N, active, other=0).to(tl.float32)
    scale = tl.load(GroupScales + index % N, active, other=0).to(tl.float32)
    if FIRST:
        total = tl.full((BLOCK,), 0.0, tl.float32)
    else:
        total = tl.load(Accumulator + index, active, other=0).to(tl.float32)
    total += (low + 4.0 * high) * scale
    if LAST:
        total *= tl.load(GlobalScale).to(tl.float32)
        if HAS_TOKEN_SCALE:
            total *= tl.load(TokenScale + index // N, active, other=1.0)
        if HAS_BIAS:
            total += tl.load(Bias + index % N, active, other=0).to(tl.float32)
        tl.store(Output + index, total, active)
    else:
        tl.store(Accumulator + index, total, active)


def nvfp4_accumulate_group(both: torch.Tensor, group_scales: torch.Tensor,
                           global_scale: torch.Tensor,
                           bias: torch.Tensor | None, accumulator: torch.Tensor,
                           output: torch.Tensor, *, first: bool, last: bool,
                           token_scale: torch.Tensor | None = None) -> None:
    """Fuse exact low/high INT4 planes with one GPTQ group scale."""
    rows, doubled_columns = both.shape
    columns = doubled_columns // 2
    _accumulate_fp4_planes[(triton.cdiv(rows * columns, 256),)](
        both, group_scales, global_scale, token_scale, bias, accumulator, output,
        rows, columns, first, last, bias is not None, token_scale is not None, 256,
        num_warps=4, enable_fp_fusion=False,
    )


__all__ = ["nvfp4_pack_and_swizzle", "nvfp4_decode", "nvfp4_accumulate_group"]
