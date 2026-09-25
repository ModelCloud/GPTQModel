# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

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
                sums[lane] = metal::abs(weights[block * 128 + lane]);
            }
            for (uint k = 8; k < 128; k += 8) {
                for (uint lane = 0; lane < 8; ++lane) {
                    sums[lane] += metal::abs(weights[block * 128 + k + lane]);
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
                    bits |= uchar(weights[block * 128 + byte * 8 + bit] >= 0.0f)
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
                    maximum, metal::abs(weights[block * 64 + k]));
            }
            float inverse = maximum == 0.0f ? 0.0f : 1.0f / maximum;
            ushort scale_bits = as_type<ushort>(half(maximum));
            uint offset = block * 18;
            packed[offset] = uchar(scale_bits & 255);
            packed[offset + 1] = uchar(scale_bits >> 8);
            for (uint byte = 0; byte < 16; ++byte) {
                uchar codes = 0;
                for (uint lane = 0; lane < 4; ++lane) {
                    float value = weights[block * 64 + byte * 4 + lane]
                        * inverse;
                    uint code = value >= 0.5f ? 2 : (value <= -0.5f ? 0 : 1);
                    codes |= uchar(code << (lane * 2));
                }
                packed[offset + 2 + byte] = codes;
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_q5_k_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q5_k_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float minima[8];
            float scales[8];
            float max_scale = 0.0f;
            float max_minimum = 0.0f;
            for (uint group = 0; group < 8; ++group) {
                float low = weights[block * 256 + group * 32];
                float high = low;
                for (uint k = 1; k < 32; ++k) {
                    float value = weights[block * 256 + group * 32 + k];
                    low = metal::min(low, value);
                    high = metal::max(high, value);
                }
                float minimum = metal::max(-low, 0.0f);
                float scale = (high + minimum) / 31.0f;
                minima[group] = minimum;
                scales[group] = scale;
                max_scale = metal::max(max_scale, scale);
                max_minimum = metal::max(max_minimum, minimum);
            }
            float base = max_scale / 63.0f;
            float min_base = max_minimum / 63.0f;
            uint offset = block * 176;
            ushort d = as_type<ushort>(half(base));
            ushort dmin = as_type<ushort>(half(min_base));
            packed[offset] = uchar(d & 255);
            packed[offset + 1] = uchar(d >> 8);
            packed[offset + 2] = uchar(dmin & 255);
            packed[offset + 3] = uchar(dmin >> 8);
            uint scale_codes[8];
            uint min_codes[8];
            for (uint group = 0; group < 8; ++group) {
                scale_codes[group] = base > 0.0f
                    ? uint(metal::clamp(int(metal::rint(scales[group] / base)), 0, 63))
                    : 0;
                min_codes[group] = min_base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(minima[group] / min_base)), 0, 63))
                    : 0;
            }
            for (uint k = 0; k < 4; ++k) {
                packed[offset + 4 + k] = uchar(
                    (scale_codes[k] & 63) | ((scale_codes[k + 4] & 48) << 2));
                packed[offset + 8 + k] = uchar(
                    (min_codes[k] & 63) | ((min_codes[k + 4] & 48) << 2));
                packed[offset + 12 + k] = uchar(
                    (scale_codes[k + 4] & 15) | ((min_codes[k + 4] & 15) << 4));
            }
            for (uint k = 0; k < 32; ++k) {
                packed[offset + 16 + k] = 0;
            }
            for (uint group = 0; group < 8; group += 2) {
                float step0 = base * float(scale_codes[group]);
                float step1 = base * float(scale_codes[group + 1]);
                float bias0 = min_base * float(min_codes[group]);
                float bias1 = min_base * float(min_codes[group + 1]);
                for (uint k = 0; k < 32; ++k) {
                    float shifted0 = weights[block * 256 + group * 32 + k] + bias0;
                    float shifted1 = weights[
                        block * 256 + (group + 1) * 32 + k] + bias1;
                    uint code0 = step0 > 0.0f
                        ? uint(metal::clamp(int(metal::rint(shifted0 / step0)), 0, 31))
                        : 0;
                    uint code1 = step1 > 0.0f
                        ? uint(metal::clamp(int(metal::rint(shifted1 / step1)), 0, 31))
                        : 0;
                    packed[offset + 16 + k] |= uchar(
                        ((code0 >> 4) & 1) << group
                        | ((code1 >> 4) & 1) << (group + 1));
                    packed[offset + 48 + group * 16 + k] = uchar(
                        (code0 & 15) | ((code1 & 15) << 4));
                }
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_q6_k_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q6_k_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float scales[16];
            float largest_scale = 0.0f;
            for (uint group = 0; group < 16; ++group) {
                float maximum = 0.0f;
                for (uint k = 0; k < 16; ++k) {
                    maximum = metal::max(maximum,
                        metal::abs(weights[block * 256 + group * 16 + k]));
                }
                scales[group] = maximum / 31.0f;
                largest_scale = metal::max(largest_scale, scales[group]);
            }
            float base = largest_scale / 127.0f;
            uint offset = block * 210;
            uint scale_codes[16];
            for (uint group = 0; group < 16; ++group) {
                uint code = base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(scales[group] / base)), 0, 127))
                    : 0;
                scale_codes[group] = code;
                packed[offset + 192 + group] = uchar(code);
            }
            ushort scale_bits = as_type<ushort>(half(base));
            packed[offset + 208] = uchar(scale_bits & 255);
            packed[offset + 209] = uchar(scale_bits >> 8);
            for (uint segment = 0; segment < 2; ++segment) {
                for (uint k = 0; k < 32; ++k) {
                    uint raw[4];
                    for (uint lane = 0; lane < 4; ++lane) {
                        uint index = segment * 128 + lane * 32 + k;
                        uint group = index / 16;
                        float step = base * float(scale_codes[group]);
                        int code = step > 0.0f
                            ? metal::clamp(int(metal::rint(
                                weights[block * 256 + index] / step)), -32, 31)
                            : 0;
                        raw[lane] = uint(code + 32);
                    }
                    packed[offset + segment * 64 + k] = uchar(
                        (raw[0] & 15) | ((raw[2] & 15) << 4));
                    packed[offset + segment * 64 + 32 + k] = uchar(
                        (raw[1] & 15) | ((raw[3] & 15) << 4));
                    packed[offset + 128 + segment * 32 + k] = uchar(
                        ((raw[0] >> 4) & 3) | (((raw[1] >> 4) & 3) << 2)
                        | (((raw[2] >> 4) & 3) << 4)
                        | (((raw[3] >> 4) & 3) << 6));
                }
            }
        """,
    )


@lru_cache(maxsize=1)
def _gguf_q4_k_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q4_k_pack",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float minima[8];
            float scales[8];
            float max_scale = 0.0f;
            float max_minimum = 0.0f;
            for (uint group = 0; group < 8; ++group) {
                float low = weights[block * 256 + group * 32];
                float high = low;
                for (uint k = 1; k < 32; ++k) {
                    float value = weights[block * 256 + group * 32 + k];
                    low = metal::min(low, value);
                    high = metal::max(high, value);
                }
                float minimum = metal::max(-low, 0.0f);
                float scale = (high + minimum) / 15.0f;
                minima[group] = minimum;
                scales[group] = scale;
                max_scale = metal::max(max_scale, scale);
                max_minimum = metal::max(max_minimum, minimum);
            }
            float base = max_scale / 63.0f;
            float min_base = max_minimum / 63.0f;
            uint offset = block * 144;
            ushort d = as_type<ushort>(half(base));
            ushort dmin = as_type<ushort>(half(min_base));
            packed[offset] = uchar(d & 255);
            packed[offset + 1] = uchar(d >> 8);
            packed[offset + 2] = uchar(dmin & 255);
            packed[offset + 3] = uchar(dmin >> 8);
            uint scale_codes[8];
            uint min_codes[8];
            for (uint group = 0; group < 8; ++group) {
                scale_codes[group] = base > 0.0f
                    ? uint(metal::clamp(int(metal::rint(scales[group] / base)), 0, 63))
                    : 0;
                min_codes[group] = min_base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(minima[group] / min_base)), 0, 63))
                    : 0;
            }
            for (uint k = 0; k < 4; ++k) {
                packed[offset + 4 + k] = uchar(
                    (scale_codes[k] & 63) | ((scale_codes[k + 4] & 48) << 2));
                packed[offset + 8 + k] = uchar(
                    (min_codes[k] & 63) | ((min_codes[k + 4] & 48) << 2));
                packed[offset + 12 + k] = uchar(
                    (scale_codes[k + 4] & 15) | ((min_codes[k + 4] & 15) << 4));
            }
            for (uint group = 0; group < 8; group += 2) {
                float step0 = base * float(scale_codes[group]);
                float step1 = base * float(scale_codes[group + 1]);
                float bias0 = min_base * float(min_codes[group]);
                float bias1 = min_base * float(min_codes[group + 1]);
                for (uint k = 0; k < 32; ++k) {
                    float shifted0 = weights[block * 256 + group * 32 + k] + bias0;
                    float shifted1 = weights[
                        block * 256 + (group + 1) * 32 + k] + bias1;
                    uint code0 = step0 > 0.0f
                        ? uint(metal::clamp(int(metal::rint(shifted0 / step0)), 0, 15))
                        : 0;
                    uint code1 = step1 > 0.0f
                        ? uint(metal::clamp(int(metal::rint(shifted1 / step1)), 0, 15))
                        : 0;
                    packed[offset + 16 + group * 16 + k] = uchar(code0 | (code1 << 4));
                }
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
            float signed_maximum = weights[block * 32];
            float maximum = metal::abs(signed_maximum);
            for (uint k = 0; k < 32; ++k) {
                float value = weights[block * 32 + k];
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
                    maximum, metal::abs(weights[block * 256 + k]));
            }
            float inverse = maximum == 0.0f ? 0.0f : 1.0f / maximum;
            uint offset = block * 54;
            uint powers[5] = {81, 27, 9, 3, 1};
            for (uint byte = 0; byte < 32; ++byte) {
                uint value = 0;
                for (uint digit = 0; digit < 5; ++digit) {
                    float normalized = weights[
                        block * 256 + digit * 32 + byte] * inverse;
                    uint code = normalized >= 0.5f
                        ? 2 : (normalized <= -0.5f ? 0 : 1);
                    value += code * powers[digit];
                }
                packed[offset + byte] = uchar((value * 256 + 242) / 243);
            }
            for (uint byte = 0; byte < 16; ++byte) {
                uint value = 0;
                for (uint digit = 0; digit < 5; ++digit) {
                    float normalized = weights[
                        block * 256 + 160 + digit * 16 + byte] * inverse;
                    uint code = normalized >= 0.5f
                        ? 2 : (normalized <= -0.5f ? 0 : 1);
                    value += code * powers[digit];
                }
                packed[offset + 32 + byte] = uchar((value * 256 + 242) / 243);
            }
            for (uint byte = 0; byte < 4; ++byte) {
                uint value = 0;
                for (uint digit = 0; digit < 4; ++digit) {
                    float normalized = weights[
                        block * 256 + 240 + digit * 4 + byte] * inverse;
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
                    maximum, metal::abs(weights[block * 256 + k]));
            }
            float inverse = maximum == 0.0f ? 0.0f : 1.0f / maximum;
            uint offset = block * 66;
            for (uint segment = 0; segment < 2; ++segment) {
                for (uint byte = 0; byte < 32; ++byte) {
                    uchar bits = 0;
                    for (uint lane = 0; lane < 4; ++lane) {
                        float normalized = weights[
                            block * 256 + segment * 128 + lane * 32 + byte]
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
                    as_type<uint>(weights[block * 32 + k]) & 0x7fffffffu);
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
            uint exponent = uint(power + 125) & 255u;
            uint scale_bits = exponent < 2
                ? (0x00200000u << exponent) : ((exponent - 1) << 23);
            float scale = as_type<float>(scale_bits);
            // Metal flushes subnormal arithmetic. For e8m0 scales 0 and 1,
            // move both operands into the normal range before comparison.
            float multiplier = exponent < 2 ? 0x1p126f : 1.0f;
            float comparison_scale = exponent < 2
                ? (exponent == 0 ? 0.25f : 0.5f) : scale;
            float fp4[16] = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
                0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
            };
            uchar codes[32];
            for (uint k = 0; k < 32; ++k) {
                float weight = weights[block * 32 + k] * multiplier;
                uint best = 0;
                float best_error = metal::abs(weight);
                for (uint candidate = 1; candidate < 16; ++candidate) {
                    float error = metal::abs(
                        comparison_scale * fp4[candidate] - weight);
                    if (error < best_error) {
                        best_error = error;
                        best = candidate;
                    }
                }
                codes[k] = uchar(best);
            }
            uint offset = block * 17;
            packed[offset] = uchar(exponent);
            for (uint k = 0; k < 16; ++k) {
                packed[offset + 1 + k] = uchar(codes[k] | (codes[k + 16] << 4));
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
                    maximum, metal::abs(weights[block * 32 + k]));
            }
            float scale = maximum / 127.0f;
            float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
            ushort scale_bits = as_type<ushort>(half(scale));
            uint offset = block * 34;
            packed[offset] = uchar(scale_bits & 255);
            packed[offset + 1] = uchar(scale_bits >> 8);
            for (uint k = 0; k < 32; ++k) {
                int code = metal::clamp(
                    int(metal::rint(weights[block * 32 + k] * inverse)),
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
        "MXFP4", "Q8_0",
    ):
        raise ValueError(
            "MLX GGUF packing supports Q1_0, Q1_0_g128, Q2_0, Q4_0, "
            "Q4_K, Q5_K, Q6_K, TQ1_0, TQ2_0, MXFP4, and Q8_0"
        )
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("weight must have float16, bfloat16, or float32 dtype")
    if normalized.startswith("Q1_0"):
        block_size = 128
    elif normalized.startswith(("Q4_K", "Q5_K")) or normalized in (
        "Q6_K", "TQ1_0", "TQ2_0"
    ):
        block_size = 256
    elif normalized == "Q2_0":
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
        kernel, bytes_per_block = _gguf_tq1_0_kernel(), 54
    elif normalized == "TQ2_0":
        kernel, bytes_per_block = _gguf_tq2_0_kernel(), 66
    elif normalized == "MXFP4":
        kernel, bytes_per_block = _gguf_mxfp4_kernel(), 17
    elif normalized == "Q4_0":
        kernel, bytes_per_block = _gguf_q4_0_kernel(), 18
    else:
        kernel, bytes_per_block = _gguf_q8_0_kernel(), 34
    packed = kernel(
        inputs=[weight.astype(mx.float32)],
        grid=(blocks, 1, 1),
        threadgroup=(min(blocks, 256), 1, 1),
        output_shapes=[(rows, columns // block_size * bytes_per_block)],
        output_dtypes=[mx.uint8],
    )[0]
    mx.eval(packed)
    return packed
