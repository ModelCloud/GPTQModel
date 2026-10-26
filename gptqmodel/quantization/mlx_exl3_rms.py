# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Native MLX blockwise RMS reductions for EXL3 regularization."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _exl3_block_rms_columns_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_block_rms_columns",
        input_names=["input"],
        output_names=["output"],
        source="""
            uint column = thread_position_in_grid.x;
            if (column >= COLUMNS) return;

            float total = 0.0f;
            for (uint block = 0; block < ROWS; block += 32u) {
                float block_sum = 0.0f;
                uint stop = metal::min(block + 32u, uint(ROWS));
                for (uint row = block; row < stop; ++row) {
                    float value = input[row * COLUMNS + column];
                    block_sum += value * value;
                }
                total += block_sum;
            }
            output[column] = metal::sqrt(total / float(ROWS));
        """,
    )


@lru_cache(maxsize=1)
def _exl3_block_rms_rows_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_block_rms_rows",
        input_names=["input"],
        output_names=["output"],
        source="""
            threadgroup float values[32];

            uint lane = thread_position_in_threadgroup.x;
            uint row = threadgroup_position_in_grid.x;
            float total = 0.0f;

            for (uint block = 0; block < COLUMNS; block += 32u) {
                uint column = block + lane;
                float value = column < COLUMNS
                    ? input[row * COLUMNS + column]
                    : 0.0f;
                values[lane] = value * value;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint offset = 16u; offset > 0u; offset >>= 1u) {
                    if (lane < offset) values[lane] += values[lane + offset];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lane == 0u) total += values[0];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lane == 0u) {
                output[row] = metal::sqrt(total / float(COLUMNS));
            }
        """,
    )


def exl3_block_rms_mlx(matrix, *, axis: int):
    """Compute EXL3's float32 RMS scales in blocks of 32 values.

    ``axis=0`` returns one scale per column with shape ``(1, columns)``.
    ``axis=1`` returns one scale per row with shape ``(rows, 1)``. This matches
    the keep-dimension layout used by EXL3 regularization.
    """
    import mlx.core as mx

    matrix = mx.array(matrix)
    if matrix.ndim != 2 or any(dimension == 0 for dimension in matrix.shape):
        raise ValueError("matrix must be a nonempty rank-two array")
    if matrix.dtype != mx.float32:
        raise ValueError("matrix must have float32 dtype")
    if not isinstance(axis, int) or isinstance(axis, bool) or axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    finite = mx.all(mx.isfinite(matrix))
    mx.eval(finite)
    if not bool(finite.item()):
        raise ValueError("matrix must contain only finite values")

    rows, columns = matrix.shape
    if axis == 0:
        output_shape = (1, columns)
        output = _exl3_block_rms_columns_kernel()(
            inputs=[mx.contiguous(matrix)],
            template=[("ROWS", rows), ("COLUMNS", columns)],
            grid=(columns, 1, 1),
            threadgroup=(min(columns, 256), 1, 1),
            output_shapes=[output_shape],
            output_dtypes=[mx.float32],
        )[0]
    else:
        output_shape = (rows, 1)
        output = _exl3_block_rms_rows_kernel()(
            inputs=[mx.contiguous(matrix)],
            template=[("COLUMNS", columns)],
            grid=(rows * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[output_shape],
            output_dtypes=[mx.float32],
        )[0]
    mx.eval(output)
    return output


__all__ = ["exl3_block_rms_mlx"]
