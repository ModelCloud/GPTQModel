# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Metal packing kernels for GGUF block formats."""

from functools import lru_cache


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
    """Pack a 2D weight matrix into GGUF Q4_0 or Q8_0 bytes on Metal."""
    import mlx.core as mx

    normalized = qtype.upper()
    if normalized not in ("Q4_0", "Q8_0"):
        raise ValueError("MLX GGUF packing supports Q4_0 and Q8_0")
    if weight.ndim != 2 or weight.shape[1] == 0 or weight.shape[1] % 32:
        raise ValueError("weight must be 2D with nonzero input width divisible by 32")
    if weight.shape[0] == 0:
        raise ValueError("weight must have at least one row")
    rows, columns = weight.shape
    blocks = rows * (columns // 32)
    kernel = _gguf_q4_0_kernel() if normalized == "Q4_0" else _gguf_q8_0_kernel()
    bytes_per_block = 18 if normalized == "Q4_0" else 34
    packed = kernel(
        inputs=[weight.astype(mx.float32)],
        grid=(blocks, 1, 1),
        threadgroup=(min(blocks, 256), 1, 1),
        output_shapes=[(rows, columns // 32 * bytes_per_block)],
        output_dtypes=[mx.uint8],
    )[0]
    mx.eval(packed)
    return packed
