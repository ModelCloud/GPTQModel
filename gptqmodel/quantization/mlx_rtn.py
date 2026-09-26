# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native MLX weight-only RTN quantization."""

from functools import lru_cache


@lru_cache(maxsize=3)
def _rtn_kernel(output_type="float32"):
    import mlx.core as mx

    output_cast = {
        "float32": "float",
        "float16": "half",
        "bfloat16": "bfloat16_t",
    }[output_type]
    return mx.fast.metal_kernel(
        name="gptqmodel_rtn_group_parallel",
        input_names=["weight"],
        output_names=["quantized", "scales", "zeros"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint group = threadgroup_position_in_grid.y;
            uint row = group / GROUPS;
            uint start = (group % GROUPS) * GROUP_SIZE;
            uint end = min(start + uint(GROUP_SIZE), uint(COLS));
            float minimum = 0.0f;
            float maximum = 0.0f;
            for (uint col = start + lane; col < end; col += 32) {
                float value = float(weight[row * COLS + col]);
                minimum = min(minimum, value);
                maximum = max(maximum, value);
            }
            minimum = simd_min(minimum);
            maximum = simd_max(maximum);
            if (SYMMETRIC) {
                maximum = max(-minimum, maximum);
                if (minimum < 0.0f) minimum = -maximum;
            }
            if (minimum == 0.0f && maximum == 0.0f) {
                minimum = -1.0f;
                maximum = 1.0f;
            }
            float scale = (maximum - minimum) / float(MAXQ);
            float zero = SYMMETRIC ? float((MAXQ + 1) / 2) :
                metal::rint(-minimum / scale);
            if (lane == 0) {
                scales[group] = scale;
                zeros[group] = zero;
            }
            for (uint col = start + lane; col < end; col += 32) {
                float value = float(weight[row * COLS + col]);
                float code = metal::clamp(metal::rint(value / scale) + zero,
                                          0.0f, float(MAXQ));
                float dequantized = scale * (code - zero);
                quantized[row * COLS + col] = OUTPUT_CAST(dequantized);
            }
        """.replace("OUTPUT_CAST", output_cast),
    )


def quantize_rtn_weight_mlx(weight, *, bits=4, group_size=128, sym=True):
    """Return ``(dequantized_weight, scales, zeros, group_indices)``.

    The tensors follow Torch ``RTN.quantize`` for a two-dimensional linear
    weight with no smoothing. The caller keeps the weight resident in MLX.
    ``group_size=-1`` means one group per output row.
    """
    import mlx.core as mx

    if weight.ndim != 2 or not weight.shape[0] or not weight.shape[1]:
        raise ValueError("weight must be a nonempty 2D matrix")
    if bits not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError("bits must be an integer from 2 through 8")
    if group_size != -1 and group_size <= 0:
        raise ValueError("group_size must be positive or -1")

    rows, cols = weight.shape
    effective = cols if group_size == -1 else group_size
    groups = (cols + effective - 1) // effective
    direct_input = weight.dtype in (mx.float16, mx.bfloat16, mx.float32)
    kernel_weight = mx.contiguous(weight) if direct_input else weight.astype(mx.float32)
    output_type = {
        mx.float16: "float16",
        mx.bfloat16: "bfloat16",
    }.get(weight.dtype, "float32")
    output_dtype = weight.dtype if direct_input else mx.float32
    quantized, scales, zeros = _rtn_kernel(output_type)(
        inputs=[kernel_weight],
        template=[
            ("ROWS", rows),
            ("COLS", cols),
            ("GROUPS", groups),
            ("GROUP_SIZE", effective),
            ("MAXQ", (1 << bits) - 1),
            ("SYMMETRIC", int(sym)),
        ],
        grid=(32, rows * groups, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, cols), (rows, groups), (rows, groups)],
        output_dtypes=[output_dtype, mx.float32, mx.float32],
    )
    if not direct_input:
        quantized = quantized.astype(weight.dtype)
    indices = mx.arange(cols, dtype=mx.int32) // effective
    mx.eval(quantized, scales, zeros, indices)
    return quantized, scales, zeros, indices


__all__ = ["quantize_rtn_weight_mlx"]
