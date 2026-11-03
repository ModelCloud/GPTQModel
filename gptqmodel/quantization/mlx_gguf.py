# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp

"""Metal packing kernels for GGUF block formats."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _gguf_q1_0_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q1_0_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            // Match the existing NumPy reference's eight-lane float32 sum.
            float sums[8];
            for (uint lane = 0; lane < 8; ++lane) {
                sums[lane] = metal::abs(float(weights[block * 128 + lane]));
            }
            for (uint k = 8; k < 128; k += 8) {
                for (uint lane = 0; lane < 8; ++lane) {
                    sums[lane] += metal::abs(
                        float(weights[block * 128 + k + lane]));
                }
            }
            float absolute_sum = ((sums[0] + sums[1]) + (sums[2] + sums[3]))
                + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
            ushort scale_bits = as_type<ushort>(half(absolute_sum / 128.0f));
            uint offset = block * 18;
            packed[offset] = uchar(scale_bits & 255);
            packed[offset + 1] = uchar(scale_bits >> 8);
            for (uint byte = 0; byte < 16; ++byte) {
                uchar bits = 0;
                for (uint bit = 0; bit < 8; ++bit) {
                    bits |= uchar(
                        float(weights[block * 128 + byte * 8 + bit]) >= 0.0f)
                        << bit;
                }
                packed[offset + 2 + byte] = bits;
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_q2_0_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q2_0_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float maximum = 0.0f;
            for (uint k = 0; k < 64; ++k) {
                maximum = metal::max(
                    maximum, metal::abs(float(weights[block * 64 + k])));
            }
            float inverse = maximum == 0.0f ? 0.0f : 1.0f / maximum;
            ushort scale_bits = as_type<ushort>(half(maximum));
            uint offset = block * 18;
            packed[offset] = uchar(scale_bits & 255);
            packed[offset + 1] = uchar(scale_bits >> 8);
            for (uint byte = 0; byte < 16; ++byte) {
                uchar codes = 0;
                for (uint lane = 0; lane < 4; ++lane) {
                    float value = float(weights[block * 64 + byte * 4 + lane])
                        * inverse;
                    uint code = value >= 0.5f ? 2 : (value <= -0.5f ? 0 : 1);
                    codes |= uchar(code << (lane * 2));
                }
                packed[offset + 2 + byte] = codes;
            }
        """,
    )


_Q5_K_WARPS = 4


@lru_cache(maxsize=1)
def _gguf_q5_k_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q5_k_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            threadgroup float values[uint(WARPS) * 256u];
            threadgroup float minima[uint(WARPS) * 8u];
            threadgroup float scales[uint(WARPS) * 8u];
            threadgroup uint scale_codes[uint(WARPS) * 8u];
            threadgroup uint min_codes[uint(WARPS) * 8u];
            uint lane = thread_position_in_threadgroup.x & 31u;
            uint warp = thread_position_in_threadgroup.x >> 5;
            uint block = threadgroup_position_in_grid.y * uint(WARPS) + warp;
            if (block >= uint(BLOCKS)) {
                return;
            }
            uint slot = warp * 8u;
            uint values_offset = warp * 256u;
            uint weight_offset = block * 256;
            // Read the block once, coalesced: every lane takes one element of
            // each 32-element sub-block. Staging it in shared memory lets the
            // min/max pass read four lanes per sub-block and the packing pass
            // read lane `lane` of each sub-block, instead of the strided
            // eight-lane scan this kernel used to issue from global memory.
            #pragma unroll
            for (uint group = 0; group < 8; ++group) {
                values[values_offset + group * 32 + lane] =
                    float(weights[weight_offset + group * 32 + lane]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Four lanes per sub-block, eight elements each, so every lane
            // participates and the serial scan is only eight steps deep.
            uint group = lane >> 2;
            uint part = lane & 3u;
            uint start = values_offset + group * 32 + part * 8;
            float low = values[start];
            float high = low;
            #pragma unroll
            for (uint k = 1; k < 8; ++k) {
                float value = values[start + k];
                low = metal::min(low, value);
                high = metal::max(high, value);
            }
            low = metal::min(low, simd_shuffle_xor(low, 1u));
            high = metal::max(high, simd_shuffle_xor(high, 1u));
            low = metal::min(low, simd_shuffle_xor(low, 2u));
            high = metal::max(high, simd_shuffle_xor(high, 2u));
            float minimum = metal::max(-low, 0.0f);
            float scale = (high + minimum) / 31.0f;
            if (part == 0) {
                minima[slot + group] = minimum;
                scales[slot + group] = scale;
            }
            float max_scale = simd_max(scale);
            float max_minimum = simd_max(minimum);
            float base = max_scale / 63.0f;
            float min_base = max_minimum / 63.0f;
            uint offset = block * 176;
            if (lane == 0) {
                ushort d = as_type<ushort>(half(base));
                ushort dmin = as_type<ushort>(half(min_base));
                packed[offset] = uchar(d & 255);
                packed[offset + 1] = uchar(d >> 8);
                packed[offset + 2] = uchar(dmin & 255);
                packed[offset + 3] = uchar(dmin >> 8);
            }
            if (lane < 8) {
                scale_codes[slot + lane] = base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(scales[slot + lane] / base)), 0, 63))
                    : 0;
                min_codes[slot + lane] = min_base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(minima[slot + lane] / min_base)), 0, 63))
                    : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 4) {
                packed[offset + 4 + lane] = uchar(
                    (scale_codes[slot + lane] & 63)
                    | ((scale_codes[slot + lane + 4] & 48) << 2));
                packed[offset + 8 + lane] = uchar(
                    (min_codes[slot + lane] & 63)
                    | ((min_codes[slot + lane + 4] & 48) << 2));
                packed[offset + 12 + lane] = uchar(
                    (scale_codes[slot + lane + 4] & 15)
                    | ((min_codes[slot + lane + 4] & 15) << 4));
            }
            uchar high_bits = 0;
            #pragma unroll
            for (uint pair = 0; pair < 8; pair += 2) {
                float step0 = base * float(scale_codes[slot + pair]);
                float step1 = base * float(scale_codes[slot + pair + 1]);
                float bias0 = min_base * float(min_codes[slot + pair]);
                float bias1 = min_base * float(min_codes[slot + pair + 1]);
                float shifted0 =
                    values[values_offset + pair * 32 + lane] + bias0;
                float shifted1 =
                    values[values_offset + (pair + 1) * 32 + lane] + bias1;
                uint code0 = step0 > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(shifted0 / step0)), 0, 31))
                    : 0;
                uint code1 = step1 > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(shifted1 / step1)), 0, 31))
                    : 0;
                high_bits |= uchar(
                    ((code0 >> 4) & 1) << pair
                    | ((code1 >> 4) & 1) << (pair + 1));
                packed[offset + 48 + pair * 16 + lane] = uchar(
                    (code0 & 15) | ((code1 & 15) << 4));
            }
            packed[offset + 16 + lane] = high_bits;
        """,
    )


def _packed_warp_launch(blocks, warps):
    """Give each 256-element GGUF block one warp, packing `warps` per group.

    A single-warp threadgroup keeps only one dependent 512-byte read in flight
    per block, so packing several warps per threadgroup raises resident warps
    per SM and hides that latency.
    """
    groups = (blocks + warps - 1) // warps
    return {
        "grid": (32 * warps, groups, 1),
        "threadgroup": (32 * warps, 1, 1),
        "template": [("WARPS", warps), ("BLOCKS", blocks)],
    }


_Q6_K_WARPS = 4


@lru_cache(maxsize=1)
def _gguf_q6_k_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q6_k_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            threadgroup float scales[uint(WARPS) * 16u];
            threadgroup uint scale_codes[uint(WARPS) * 16u];
            uint lane = thread_position_in_threadgroup.x & 31u;
            uint warp = thread_position_in_threadgroup.x >> 5;
            uint block = threadgroup_position_in_grid.y * uint(WARPS) + warp;
            if (block >= uint(BLOCKS)) {
                return;
            }
            uint scale_offset = warp * 16u;
            // Cache this lane's eight weights for the block in registers so the
            // packing pass below does not issue a second stream of global reads.
            float own[8];
            #pragma unroll
            for (uint segment = 0; segment < 2; ++segment) {
                #pragma unroll
                for (uint value_lane = 0; value_lane < 4; ++value_lane) {
                    own[segment * 4 + value_lane] = float(weights[
                        block * 256 + segment * 128 + value_lane * 32 + lane]);
                }
            }
            float scale = 0.0f;
            if (lane < 16) {
                float maximum = 0.0f;
                for (uint k = 0; k < 16; ++k) {
                    maximum = metal::max(maximum,
                        metal::abs(float(
                            weights[block * 256 + lane * 16 + k])));
                }
                scale = maximum / 31.0f;
                scales[scale_offset + lane] = scale;
            }
            float largest_scale = simd_max(scale);
            float base = largest_scale / 127.0f;
            uint offset = block * 210;
            if (lane < 16) {
                uint code = base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(scales[scale_offset + lane] / base)),
                        0, 127))
                    : 0;
                scale_codes[scale_offset + lane] = code;
                packed[offset + 192 + lane] = uchar(code);
            }
            if (lane == 0) {
                ushort scale_bits = as_type<ushort>(half(base));
                packed[offset + 208] = uchar(scale_bits & 255);
                packed[offset + 209] = uchar(scale_bits >> 8);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint segment = 0; segment < 2; ++segment) {
                uint raw[4];
                for (uint value_lane = 0; value_lane < 4; ++value_lane) {
                    uint index = segment * 128 + value_lane * 32 + lane;
                    uint group = index / 16;
                    float step = base * float(scale_codes[scale_offset + group]);
                    int code = step > 0.0f
                        ? metal::clamp(int(metal::rint(
                            own[segment * 4 + value_lane] / step)), -32, 31)
                        : 0;
                    raw[value_lane] = uint(code + 32);
                }
                packed[offset + segment * 64 + lane] = uchar(
                    (raw[0] & 15) | ((raw[2] & 15) << 4));
                packed[offset + segment * 64 + 32 + lane] = uchar(
                    (raw[1] & 15) | ((raw[3] & 15) << 4));
                packed[offset + 128 + segment * 32 + lane] = uchar(
                    ((raw[0] >> 4) & 3) | (((raw[1] >> 4) & 3) << 2)
                    | (((raw[2] >> 4) & 3) << 4)
                    | (((raw[3] >> 4) & 3) << 6));
            }
        """,
    )


_Q4_K_WARPS = 4


@lru_cache(maxsize=1)
def _gguf_q4_k_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q4_k_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            threadgroup float values[uint(WARPS) * 256u];
            threadgroup float minima[uint(WARPS) * 8u];
            threadgroup float scales[uint(WARPS) * 8u];
            threadgroup uint scale_codes[uint(WARPS) * 8u];
            threadgroup uint min_codes[uint(WARPS) * 8u];
            uint lane = thread_position_in_threadgroup.x & 31u;
            uint warp = thread_position_in_threadgroup.x >> 5;
            uint block = threadgroup_position_in_grid.y * uint(WARPS) + warp;
            if (block >= uint(BLOCKS)) {
                return;
            }
            uint slot = warp * 8u;
            uint values_offset = warp * 256u;
            uint weight_offset = block * 256;
            // Read the block once, coalesced: every lane takes one element of
            // each 32-element sub-block. Staging it in shared memory lets the
            // min/max pass read four lanes per sub-block and the packing pass
            // read lane `lane` of each sub-block, instead of the strided
            // eight-lane scan this kernel used to issue from global memory.
            #pragma unroll
            for (uint group = 0; group < 8; ++group) {
                values[values_offset + group * 32 + lane] =
                    float(weights[weight_offset + group * 32 + lane]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Four lanes per sub-block, eight elements each, so every lane
            // participates and the serial scan is only eight steps deep.
            uint group = lane >> 2;
            uint part = lane & 3u;
            uint start = values_offset + group * 32 + part * 8;
            float low = values[start];
            float high = low;
            #pragma unroll
            for (uint k = 1; k < 8; ++k) {
                float value = values[start + k];
                low = metal::min(low, value);
                high = metal::max(high, value);
            }
            low = metal::min(low, simd_shuffle_xor(low, 1u));
            high = metal::max(high, simd_shuffle_xor(high, 1u));
            low = metal::min(low, simd_shuffle_xor(low, 2u));
            high = metal::max(high, simd_shuffle_xor(high, 2u));
            float minimum = metal::max(-low, 0.0f);
            float scale = (high + minimum) / 15.0f;
            if (part == 0) {
                minima[slot + group] = minimum;
                scales[slot + group] = scale;
            }
            float max_scale = simd_max(scale);
            float max_minimum = simd_max(minimum);
            float base = max_scale / 63.0f;
            float min_base = max_minimum / 63.0f;
            uint offset = block * 144;
            if (lane == 0) {
                ushort d = as_type<ushort>(half(base));
                ushort dmin = as_type<ushort>(half(min_base));
                packed[offset] = uchar(d & 255);
                packed[offset + 1] = uchar(d >> 8);
                packed[offset + 2] = uchar(dmin & 255);
                packed[offset + 3] = uchar(dmin >> 8);
            }
            if (lane < 8) {
                scale_codes[slot + lane] = base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(scales[slot + lane] / base)), 0, 63))
                    : 0;
                min_codes[slot + lane] = min_base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(minima[slot + lane] / min_base)), 0, 63))
                    : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 4) {
                packed[offset + 4 + lane] = uchar(
                    (scale_codes[slot + lane] & 63)
                    | ((scale_codes[slot + lane + 4] & 48) << 2));
                packed[offset + 8 + lane] = uchar(
                    (min_codes[slot + lane] & 63)
                    | ((min_codes[slot + lane + 4] & 48) << 2));
                packed[offset + 12 + lane] = uchar(
                    (scale_codes[slot + lane + 4] & 15)
                    | ((min_codes[slot + lane + 4] & 15) << 4));
            }
            #pragma unroll
            for (uint pair = 0; pair < 8; pair += 2) {
                float step0 = base * float(scale_codes[slot + pair]);
                float step1 = base * float(scale_codes[slot + pair + 1]);
                float bias0 = min_base * float(min_codes[slot + pair]);
                float bias1 = min_base * float(min_codes[slot + pair + 1]);
                float shifted0 =
                    values[values_offset + pair * 32 + lane] + bias0;
                float shifted1 =
                    values[values_offset + (pair + 1) * 32 + lane] + bias1;
                uint code0 = step0 > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(shifted0 / step0)), 0, 15))
                    : 0;
                uint code1 = step1 > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(shifted1 / step1)), 0, 15))
                    : 0;
                packed[offset + 16 + pair * 16 + lane] = uchar(
                    code0 | (code1 << 4));
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_q4_0_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q4_0_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float values[32];
            float signed_maximum = float(weights[block * 32]);
            float maximum = metal::abs(signed_maximum);
            for (uint k = 0; k < 32; ++k) {
                float value = float(weights[block * 32 + k]);
                values[k] = value;
                float magnitude = metal::abs(value);
                if (magnitude > maximum) {
                    maximum = magnitude;
                    signed_maximum = value;
                }
            }
            float scale = signed_maximum / -8.0f;
            float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
            ushort scale_bits = as_type<ushort>(half(scale));
            uint offset = block * 18;
            packed[offset] = uchar(scale_bits & 255);
            packed[offset + 1] = uchar(scale_bits >> 8);
            for (uint k = 0; k < 16; ++k) {
                int low = int(metal::floor(values[k] * inverse + 8.5f));
                int high = int(metal::floor(values[k + 16] * inverse + 8.5f));
                // The GGUF CPU reference multiplies in float64. Use a fused
                // multiply-add to compare the exact float32 operand product
                // with each half-integer bin boundary before packing.
                float low_delta = metal::fma(
                    values[k], inverse, 8.5f - float(low));
                if (low_delta < 0.0f) {
                    --low;
                } else if (metal::fma(
                               values[k], inverse, 7.5f - float(low)) >= 0.0f) {
                    ++low;
                }
                float high_delta = metal::fma(
                    values[k + 16], inverse, 8.5f - float(high));
                if (high_delta < 0.0f) {
                    --high;
                } else if (metal::fma(
                               values[k + 16], inverse,
                               7.5f - float(high)) >= 0.0f) {
                    ++high;
                }
                low = metal::clamp(low, 0, 15);
                high = metal::clamp(high, 0, 15);
                packed[offset + 2 + k] = uchar(low | (high << 4));
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_tq1_0_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_tq1_0_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float maximum = 0.0f;
            for (uint k = 0; k < 256; ++k) {
                maximum = metal::max(
                    maximum, metal::abs(float(weights[block * 256 + k])));
            }
            float inverse = maximum == 0.0f ? 0.0f : 1.0f / maximum;
            uint offset = block * 54;
            uint powers[5] = {81, 27, 9, 3, 1};
            for (uint byte = 0; byte < 32; ++byte) {
                uint value = 0;
                for (uint digit = 0; digit < 5; ++digit) {
                    float normalized = float(weights[
                        block * 256 + digit * 32 + byte]) * inverse;
                    uint code = normalized >= 0.5f
                        ? 2 : (normalized <= -0.5f ? 0 : 1);
                    value += code * powers[digit];
                }
                packed[offset + byte] = uchar((value * 256 + 242) / 243);
            }
            for (uint byte = 0; byte < 16; ++byte) {
                uint value = 0;
                for (uint digit = 0; digit < 5; ++digit) {
                    float normalized = float(weights[
                        block * 256 + 160 + digit * 16 + byte]) * inverse;
                    uint code = normalized >= 0.5f
                        ? 2 : (normalized <= -0.5f ? 0 : 1);
                    value += code * powers[digit];
                }
                packed[offset + 32 + byte] = uchar((value * 256 + 242) / 243);
            }
            for (uint byte = 0; byte < 4; ++byte) {
                uint value = 0;
                for (uint digit = 0; digit < 4; ++digit) {
                    float normalized = float(weights[
                        block * 256 + 240 + digit * 4 + byte]) * inverse;
                    uint code = normalized >= 0.5f
                        ? 2 : (normalized <= -0.5f ? 0 : 1);
                    value += code * powers[digit];
                }
                packed[offset + 48 + byte] = uchar((value * 256 + 242) / 243);
            }
            ushort scale_bits = as_type<ushort>(half(maximum));
            packed[offset + 52] = uchar(scale_bits & 255);
            packed[offset + 53] = uchar(scale_bits >> 8);
        """,
    )


_TQ1_0_WARPS = 4


@lru_cache(maxsize=1)
def _gguf_tq1_0_parallel_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_tq1_0_pack_parallel",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint lane = thread_position_in_threadgroup.x & 31u;
            uint warp = thread_position_in_threadgroup.x >> 5;
            uint block = threadgroup_position_in_grid.y * uint(WARPS) + warp;
            if (block >= uint(BLOCKS)) {
                return;
            }
            // Cache this lane's eight weights for the block in registers: the
            // maximum pass and the first packing pass read exactly these
            // elements, so they no longer issue a second stream of global
            // reads.
            float own[8];
            #pragma unroll
            for (uint g = 0; g < 8; ++g) {
                own[g] = float(weights[block * 256 + g * 32 + lane]);
            }
            float maximum = 0.0f;
            #pragma unroll
            for (uint g = 0; g < 8; ++g) {
                maximum = metal::max(maximum, metal::abs(own[g]));
            }
            maximum = simd_max(maximum);
            float inverse = maximum == 0.0f ? 0.0f : 1.0f / maximum;
            // The two tail passes need lane `lane + 16` of sub-blocks 5..7 and
            // lanes 16..31 of sub-block 7, so fetch them with warp shuffles.
            // Hoisted out of the divergent branches below.
            float s5 = simd_shuffle_xor(own[5], 16u);
            float s6 = simd_shuffle_xor(own[6], 16u);
            float t16 = simd_shuffle_xor(own[7], 16u);
            float t20 = simd_shuffle_xor(own[7], 20u);
            float t24 = simd_shuffle_xor(own[7], 24u);
            float t28 = simd_shuffle_xor(own[7], 28u);
            uint offset = block * 54;
            uint powers[5] = {81, 27, 9, 3, 1};
            uint value = 0;
            #pragma unroll
            for (uint digit = 0; digit < 5; ++digit) {
                float normalized = own[digit] * inverse;
                uint code = normalized >= 0.5f
                    ? 2 : (normalized <= -0.5f ? 0 : 1);
                value += code * powers[digit];
            }
            packed[offset + lane] = uchar((value * 256 + 242) / 243);
            if (lane < 16) {
                float src[5] = {own[5], s5, own[6], s6, own[7]};
                value = 0;
                #pragma unroll
                for (uint digit = 0; digit < 5; ++digit) {
                    float normalized = src[digit] * inverse;
                    uint code = normalized >= 0.5f
                        ? 2 : (normalized <= -0.5f ? 0 : 1);
                    value += code * powers[digit];
                }
                packed[offset + 32 + lane] = uchar(
                    (value * 256 + 242) / 243);
            }
            if (lane < 4) {
                float src[4] = {t16, t20, t24, t28};
                value = 0;
                #pragma unroll
                for (uint digit = 0; digit < 4; ++digit) {
                    float normalized = src[digit] * inverse;
                    uint code = normalized >= 0.5f
                        ? 2 : (normalized <= -0.5f ? 0 : 1);
                    value += code * powers[digit];
                }
                packed[offset + 48 + lane] = uchar(
                    (value * 256 + 242) / 243);
            }
            if (lane == 0) {
                ushort scale_bits = as_type<ushort>(half(maximum));
                packed[offset + 52] = uchar(scale_bits & 255);
                packed[offset + 53] = uchar(scale_bits >> 8);
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_tq2_0_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_tq2_0_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float maximum = 0.0f;
            for (uint k = 0; k < 256; ++k) {
                maximum = metal::max(
                    maximum, metal::abs(float(weights[block * 256 + k])));
            }
            float inverse = maximum == 0.0f ? 0.0f : 1.0f / maximum;
            uint offset = block * 66;
            for (uint segment = 0; segment < 2; ++segment) {
                for (uint byte = 0; byte < 32; ++byte) {
                    uchar bits = 0;
                    for (uint lane = 0; lane < 4; ++lane) {
                        float normalized = float(weights[
                            block * 256 + segment * 128 + lane * 32 + byte])
                            * inverse;
                        uint code = normalized >= 0.5f
                            ? 2 : (normalized <= -0.5f ? 0 : 1);
                        bits |= uchar(code << (2 * lane));
                    }
                    packed[offset + segment * 32 + byte] = bits;
                }
            }
            ushort scale_bits = as_type<ushort>(half(maximum));
            packed[offset + 64] = uchar(scale_bits & 255);
            packed[offset + 65] = uchar(scale_bits >> 8);
        """,
    )


_TQ2_0_WARPS = 4


@lru_cache(maxsize=1)
def _gguf_tq2_0_parallel_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_tq2_0_pack_parallel",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint lane = thread_position_in_threadgroup.x & 31u;
            uint warp = thread_position_in_threadgroup.x >> 5;
            uint block = threadgroup_position_in_grid.y * uint(WARPS) + warp;
            if (block >= uint(BLOCKS)) {
                return;
            }
            uint base_index = block * 256;
            // Keep this lane's eight weights in registers: the packing pass
            // below would otherwise re-read the whole block from memory.
            float own[8];
            #pragma unroll
            for (uint value_lane = 0; value_lane < 8; ++value_lane) {
                own[value_lane] = float(
                    weights[base_index + value_lane * 32 + lane]);
            }
            float maximum = 0.0f;
            #pragma unroll
            for (uint value_lane = 0; value_lane < 8; ++value_lane) {
                maximum = metal::max(maximum, metal::abs(own[value_lane]));
            }
            maximum = simd_max(maximum);
            float inverse = maximum == 0.0f ? 0.0f : 1.0f / maximum;
            uint offset = block * 66;
            for (uint segment = 0; segment < 2; ++segment) {
                uchar bits = 0;
                #pragma unroll
                for (uint value_lane = 0; value_lane < 4; ++value_lane) {
                    float normalized = own[segment * 4 + value_lane] * inverse;
                    uint code = normalized >= 0.5f
                        ? 2 : (normalized <= -0.5f ? 0 : 1);
                    bits |= uchar(code << (2 * value_lane));
                }
                packed[offset + segment * 32 + lane] = bits;
            }
            if (lane == 0) {
                ushort scale_bits = as_type<ushort>(half(maximum));
                packed[offset + 64] = uchar(scale_bits & 255);
                packed[offset + 65] = uchar(scale_bits >> 8);
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_mxfp4_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_mxfp4_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            uint maximum_bits = 0;
            for (uint k = 0; k < 32; ++k) {
                maximum_bits = metal::max(maximum_bits,
                    as_type<uint>(float(weights[block * 32 + k]))
                        & 0x7fffffffu);
            }
            int power = int(maximum_bits >> 23) - 127;
            if (maximum_bits == 0) {
                power = -125;
            } else if (maximum_bits < 0x00800000u) {
                uint leading = 31u - metal::clz(maximum_bits);
                power = int(leading) - 149;
                int next_power = power + 1;
                uint cutoff = next_power == -126 ? 22
                    : (next_power >= -128 ? 11
                    : (next_power == -129 ? 5
                    : (next_power == -130 ? 2
                    : (next_power == -131 ? 1 : 0))));
                uint next_bits = 1u << (leading + 1u);
                if (maximum_bits >= next_bits - cutoff) power = next_power;
            } else {
                // NumPy rounds log2 to float32 before floor. Near powers of
                // two that rounds to the next integer; these ULP cutoffs
                // reproduce its rounding for normal float32 weights.
                int next_power = power + 1;
                uint cutoff = 0;
                if (next_power <= -64) cutoff = 44;
                else if (next_power <= -32) cutoff = 22;
                else if (next_power <= -16) cutoff = 11;
                else if (next_power <= -8) cutoff = 5;
                else if (next_power <= -4) cutoff = 2;
                else if (next_power <= -2) cutoff = 1;
                else if (next_power >= 65) cutoff = 44;
                else if (next_power >= 33) cutoff = 22;
                else if (next_power >= 17) cutoff = 11;
                else if (next_power >= 9) cutoff = 5;
                else if (next_power >= 5) cutoff = 2;
                else if (next_power >= 3) cutoff = 1;
                if (maximum_bits >= ((maximum_bits & 0x7f800000u)
                    + 0x00800000u - cutoff)) power = next_power;
            }
            uint exponent = uint(metal::clamp(power + 125, 0, 254));
            uint scale_bits = exponent < 2
                ? (0x00200000u << exponent) : ((exponent - 1) << 23);
            float scale = as_type<float>(scale_bits);
            // Metal flushes subnormal arithmetic. Scale 2 also has a
            // subnormal rounding midpoint, so compare in a normal range.
            float multiplier = exponent <= 2 ? 0x1p126f : 1.0f;
            float comparison_scale = exponent <= 2
                ? (exponent == 0 ? 0.25f : (exponent == 1 ? 0.5f : 1.0f)) : scale;
            float fp4[8] = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f
            };
            uchar codes[32];
            for (uint k = 0; k < 32; ++k) {
                float weight = float(weights[block * 32 + k]);
                uint weight_bits = as_type<uint>(weight);
                uint sign = weight_bits >> 31;
                uint magnitude_bits = weight_bits & 0x7fffffffu;
                // Recover subnormal inputs from their mantissa; multiplying
                // the original float would flush it before the rescaling.
                float magnitude = exponent <= 2 && magnitude_bits < 0x00800000u
                    ? float(magnitude_bits) * 0x1p-23f
                    : metal::abs(weight) * multiplier;
                uint best = 0;
                float best_error = magnitude;
                for (uint candidate = 1; candidate < 8; ++candidate) {
                    float error = metal::abs(
                        comparison_scale * fp4[candidate] - magnitude);
                    if (error < best_error) {
                        best_error = error;
                        best = candidate;
                    }
                }
                codes[k] = uchar(best | ((sign && best) ? 8 : 0));
            }
            uint offset = block * 17;
            packed[offset] = uchar(exponent);
            for (uint k = 0; k < 16; ++k) {
                packed[offset + 1 + k] = uchar(codes[k] | (codes[k + 16] << 4));
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_nvfp4_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_nvfp4_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            uint input_offset = block * 64;
            uint output_offset = block * 36;
            float fp4[8] = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f
            };
            for (uint group = 0; group < 4; ++group) {
                uint group_offset = input_offset + group * 16;
                float maximum = 0.0f;
                for (uint k = 0; k < 16; ++k) {
                    float magnitude = metal::abs(float(weights[group_offset + k]));
                    if (maximum < magnitude) maximum = magnitude;
                }

                float scale_source = maximum / 6.0f;
                uchar encoded_scale = 0;
                if (scale_source > 0.0f) {
                    scale_source = metal::min(scale_source, 448.0f);
                    uint bits = as_type<uint>(scale_source);
                    int source_exponent = int((bits >> 23) & 255u) - 127;
                    int exponent = source_exponent + 7;
                    if (exponent <= 0) {
                        int mantissa = int(scale_source * 512.0f + 0.5f);
                        encoded_scale = uchar(metal::clamp(mantissa, 0, 7));
                    } else if (exponent >= 15) {
                        encoded_scale = 0x7e;
                    } else {
                        int mantissa = int((bits >> 20) & 7u)
                            + int((bits >> 19) & 1u);
                        if (mantissa > 7) {
                            mantissa = 0;
                            ++exponent;
                        }
                        encoded_scale = exponent >= 15
                            ? uchar(0x7e)
                            : uchar((exponent << 3) | mantissa);
                    }
                }
                packed[output_offset + group] = encoded_scale;

                uint exponent = (uint(encoded_scale) >> 3) & 15u;
                uint mantissa = uint(encoded_scale) & 7u;
                float scale = encoded_scale == 0 || encoded_scale == 0x7f
                    ? 0.0f
                    : (exponent == 0
                        ? float(mantissa) * 0x1p-10f
                        : metal::ldexp(
                            1.0f + float(mantissa) * 0.125f,
                            int(exponent) - 8));
                for (uint k = 0; k < 8; ++k) {
                    uchar codes[2];
                    for (uint half_index = 0; half_index < 2; ++half_index) {
                        float weight = float(weights[
                            group_offset + k + half_index * 8]);
                        uint sign = as_type<uint>(weight) >> 31;
                        float magnitude = metal::abs(weight);
                        uint best = 0;
                        float best_error = magnitude;
                        for (uint candidate = 1; candidate < 8; ++candidate) {
                            float error = metal::abs(
                                scale * fp4[candidate] - magnitude);
                            if (error < best_error) {
                                best_error = error;
                                best = candidate;
                            }
                        }
                        codes[half_index] = uchar(
                            best | ((sign && best) ? 8 : 0));
                    }
                    packed[output_offset + 4 + group * 8 + k] = uchar(
                        codes[0] | (codes[1] << 4));
                }
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_q8_0_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q8_0_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float maximum = 0.0f;
            for (uint k = 0; k < 32; ++k) {
                maximum = metal::max(
                    maximum, metal::abs(float(weights[block * 32 + k])));
            }
            float scale = maximum / 127.0f;
            float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
            ushort scale_bits = as_type<ushort>(half(scale));
            uint offset = block * 34;
            packed[offset] = uchar(scale_bits & 255);
            packed[offset + 1] = uchar(scale_bits >> 8);
            for (uint k = 0; k < 32; ++k) {
                int code = metal::clamp(
                    int(metal::rint(float(weights[block * 32 + k]) * inverse)),
                    -128, 127);
                packed[offset + 2 + k] = uchar(code);
            }
        """,
    )


def gguf_quantize_weight_mlx(weight, qtype: str):
    """Pack a 2D weight matrix into supported GGUF block bytes on Metal."""
    import mlx.core as mx

    normalized = qtype.upper()
    if normalized not in (
        "Q1_0", "Q1_0_G128", "Q2_0", "Q4_0", "Q4_K", "Q4_K_S", "Q4_K_M",
        "Q5_K", "Q5_K_S", "Q5_K_M", "Q6_K", "TQ1_0", "TQ2_0",
        "MXFP4", "NVFP4", "Q8_0",
    ):
        raise ValueError(
            "MLX GGUF packing supports Q1_0, Q1_0_g128, Q2_0, Q4_0, "
            "Q4_K, Q5_K, Q6_K, TQ1_0, TQ2_0, MXFP4, NVFP4, and Q8_0"
        )
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("weight must have float16, bfloat16, or float32 dtype")
    if normalized.startswith("Q1_0"):
        block_size = 128
    elif normalized.startswith(("Q4_K", "Q5_K")) or normalized in (
        "Q6_K", "TQ1_0", "TQ2_0"
    ):
        block_size = 256
    elif normalized in ("Q2_0", "NVFP4"):
        block_size = 64
    else:
        block_size = 32
    if weight.ndim != 2 or weight.shape[1] == 0 or weight.shape[1] % block_size:
        raise ValueError(
            f"weight must be 2D with nonzero input width divisible by {block_size}"
        )
    if weight.shape[0] == 0:
        raise ValueError("weight must have at least one row")
    rows, columns = weight.shape
    blocks = rows * (columns // block_size)
    if normalized.startswith("Q1_0"):
        kernel, bytes_per_block = _gguf_q1_0_kernel(), 18
    elif normalized == "Q2_0":
        kernel, bytes_per_block = _gguf_q2_0_kernel(), 18
    elif normalized.startswith("Q4_K"):
        kernel, bytes_per_block = _gguf_q4_k_kernel(), 144
    elif normalized.startswith("Q5_K"):
        kernel, bytes_per_block = _gguf_q5_k_kernel(), 176
    elif normalized == "Q6_K":
        kernel, bytes_per_block = _gguf_q6_k_kernel(), 210
    elif normalized == "TQ1_0":
        kernel = (
            _gguf_tq1_0_parallel_kernel()
            if blocks >= 65536
            else _gguf_tq1_0_kernel()
        )
        bytes_per_block = 54
    elif normalized == "TQ2_0":
        kernel = (
            _gguf_tq2_0_parallel_kernel()
            if blocks >= 65536
            else _gguf_tq2_0_kernel()
        )
        bytes_per_block = 66
    elif normalized == "MXFP4":
        kernel, bytes_per_block = _gguf_mxfp4_kernel(), 17
    elif normalized == "NVFP4":
        kernel, bytes_per_block = _gguf_nvfp4_kernel(), 36
    elif normalized == "Q4_0":
        kernel, bytes_per_block = _gguf_q4_0_kernel(), 18
    else:
        kernel, bytes_per_block = _gguf_q8_0_kernel(), 34
    if normalized == "Q6_K":
        launch = _packed_warp_launch(blocks, _Q6_K_WARPS)
    elif normalized.startswith("Q4_K"):
        launch = _packed_warp_launch(blocks, _Q4_K_WARPS)
    elif normalized.startswith("Q5_K"):
        launch = _packed_warp_launch(blocks, _Q5_K_WARPS)
    elif normalized == "TQ1_0" and blocks >= 65536:
        launch = _packed_warp_launch(blocks, _TQ1_0_WARPS)
    elif normalized == "TQ2_0" and blocks >= 65536:
        launch = _packed_warp_launch(blocks, _TQ2_0_WARPS)
    else:
        launch = {"grid": (blocks, 1, 1), "threadgroup": (min(blocks, 256), 1, 1)}
    direct_input = (
        normalized.startswith(("Q1_0", "Q4_K", "Q5_K"))
        or normalized in (
            "Q2_0", "Q4_0", "Q6_K", "TQ1_0", "TQ2_0", "MXFP4", "NVFP4", "Q8_0"
        )
    ) and weight.dtype in (
        mx.float16, mx.bfloat16, mx.float32,
    )
    kernel_weight = (
        mx.contiguous(weight) if direct_input else weight.astype(mx.float32)
    )
    packed = kernel(
        inputs=[kernel_weight],
        output_shapes=[(rows, columns // block_size * bytes_per_block)],
        output_dtypes=[mx.uint8],
        **launch,
    )[0]
    mx.eval(packed)
    return packed
