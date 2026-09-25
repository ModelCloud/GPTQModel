# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native MLX weight-only RTN quantization."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _rtn_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_rtn_group",
        input_names=["weight"],
        output_names=["quantized", "scales", "zeros"],
        source="""
            uint group = thread_position_in_grid.x;
            if (group >= ROWS * GROUPS) return;
            uint row = group / GROUPS;
            uint start = (group % GROUPS) * GROUP_SIZE;
            uint end = min(start + uint(GROUP_SIZE), uint(COLS));
            float minimum = 0.0f;
            float maximum = 0.0f;
            for (uint col = start; col < end; ++col) {
                float value = weight[row * COLS + col];
                minimum = min(minimum, value);
                maximum = max(maximum, value);
            }
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
            scales[group] = scale;
            zeros[group] = zero;
            for (uint col = start; col < end; ++col) {
                float value = weight[row * COLS + col];
                float code = metal::clamp(metal::rint(value / scale) + zero,
                                          0.0f, float(MAXQ));
                quantized[row * COLS + col] = scale * (code - zero);
            }
        """,
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
    quantized, scales, zeros = _rtn_kernel()(
        inputs=[weight.astype(mx.float32)],
        template=[
            ("ROWS", rows),
            ("COLS", cols),
            ("GROUPS", groups),
            ("GROUP_SIZE", effective),
            ("MAXQ", (1 << bits) - 1),
            ("SYMMETRIC", int(sym)),
        ],
        grid=(rows * groups, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols), (rows, groups), (rows, groups)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    quantized = quantized.astype(weight.dtype)
    indices = mx.arange(cols, dtype=mx.int32) // effective
    mx.eval(quantized, scales, zeros, indices)
    return quantized, scales, zeros, indices


__all__ = ["quantize_rtn_weight_mlx"]
