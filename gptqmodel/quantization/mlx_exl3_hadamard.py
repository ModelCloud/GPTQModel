# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Native MLX normalized Hadamard transforms for EXL3 quantization."""

from functools import lru_cache

_EXL3_HADAMARD_SIZE = 128


@lru_cache(maxsize=32)
def _exl3_hadamard_kernel(columns: int, axis: int):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_hadamard_128",
        input_names=["input"],
        output_names=["output"],
        source="""
            threadgroup float current[128];
            threadgroup float next_values[128];

            uint lane = thread_position_in_threadgroup.x;
            uint vector = threadgroup_position_in_grid.x;
            uint element;
            if (AXIS == 1) {
                uint row = vector / COLUMN_BLOCKS;
                uint column = (vector % COLUMN_BLOCKS) * 128 + lane;
                element = row * COLUMNS + column;
            } else {
                uint row = (vector / COLUMNS) * 128 + lane;
                uint column = vector % COLUMNS;
                element = row * COLUMNS + column;
            }

            float value = input[element];

            for (uint stride = 1; stride < 32; stride <<= 1) {
                float other = simd_shuffle_xor(value, stride);
                value = (lane & stride) == 0
                    ? value + other
                    : other - value;
            }

            current[lane] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = 32; stride < 128; stride <<= 1) {
                uint partner = lane ^ stride;
                float own = current[lane];
                float other = current[partner];
                next_values[lane] = (lane & stride) == 0
                    ? own + other
                    : other - own;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                current[lane] = next_values[lane];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            output[element] = current[lane] * 0.08838834764831845f;
        """,
    )


def exl3_hadamard_128_mlx(matrix, *, axis: int):
    """Apply EXL3's normalized 128-point Hadamard transform by block.

    ``matrix`` must be a finite float32 rank-two MLX array. ``axis=0`` applies
    the transform independently to each 128-row block, while ``axis=1``
    applies it to each 128-column block. The output is float32 with the same
    shape. EXL3's order-128 transform is orthonormal and self-inverse.
    """
    import mlx.core as mx

    matrix = mx.array(matrix)
    if matrix.ndim != 2 or any(dimension == 0 for dimension in matrix.shape):
        raise ValueError("matrix must be a nonempty rank-two array")
    if matrix.dtype != mx.float32:
        raise ValueError("matrix must have float32 dtype")
    if not isinstance(axis, int) or isinstance(axis, bool) or axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if matrix.shape[axis] % _EXL3_HADAMARD_SIZE:
        raise ValueError("the transformed dimension must be divisible by 128")

    finite = mx.all(mx.isfinite(matrix))
    mx.eval(finite)
    if not bool(finite.item()):
        raise ValueError("matrix must contain only finite values")

    rows, columns = matrix.shape
    column_blocks = max(1, columns // _EXL3_HADAMARD_SIZE)
    vector_count = (
        rows * column_blocks if axis == 1 else (rows // _EXL3_HADAMARD_SIZE) * columns
    )
    output = _exl3_hadamard_kernel(columns, axis)(
        inputs=[mx.contiguous(matrix)],
        template=[
            ("AXIS", axis),
            ("COLUMNS", columns),
            ("COLUMN_BLOCKS", column_blocks),
        ],
        grid=(vector_count * _EXL3_HADAMARD_SIZE, 1, 1),
        threadgroup=(_EXL3_HADAMARD_SIZE, 1, 1),
        output_shapes=[matrix.shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(output)
    return output


__all__ = ["exl3_hadamard_128_mlx"]
