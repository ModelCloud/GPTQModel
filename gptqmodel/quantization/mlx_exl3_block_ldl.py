# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Native MLX block-LDL factorization for EXL3 quantization."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _exl3_block_ldl_normalize_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_block_ldl_normalize",
        input_names=["factor", "diagonal_inverse"],
        output_names=["normalized"],
        source=r"""
            uint index = thread_position_in_grid.x;
            if (index >= ELEMENTS) return;

            uint row = index / COLUMNS;
            uint column = index - row * COLUMNS;
            uint row_block = row / BLOCK_SIZE;
            uint column_block = column / BLOCK_SIZE;
            uint within_column = column - column_block * BLOCK_SIZE;

            if (row_block < column_block) {
                normalized[index] = 0.0f;
                return;
            }
            if (row_block == column_block) {
                normalized[index] =
                    (row % BLOCK_SIZE == within_column) ? 1.0f : 0.0f;
                return;
            }

            float total = 0.0f;
            uint factor_base = column_block * BLOCK_SIZE;
            uint inverse_base =
                column_block * BLOCK_SIZE * BLOCK_SIZE + within_column;
            for (uint reduction = 0; reduction < BLOCK_SIZE; ++reduction) {
                total = metal::fma(
                    factor[row * COLUMNS + factor_base + reduction],
                    diagonal_inverse[
                        inverse_base + reduction * BLOCK_SIZE
                    ],
                    total
                );
            }
            normalized[index] = total;
        """,
    )


def _all_finite(array, *, stream):
    import mlx.core as mx

    return bool(mx.all(mx.isfinite(array, stream=stream), stream=stream).item())


def exl3_block_ldl_mlx(hessian, *, block_size: int = 16):
    """Return EXL3's block-unit-lower factor for a positive-definite Hessian.

    MLX currently exposes Cholesky on its CPU stream. The diagonal-block
    inverses are computed there as well, while a fused Metal kernel performs
    the dense block-column normalization and writes exact identity diagonal
    blocks. The input is not modified.
    """
    import mlx.core as mx

    hessian = mx.array(hessian)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square rank-two array")
    if hessian.shape[0] == 0:
        raise ValueError("hessian must be nonempty")
    if hessian.dtype != mx.float32:
        raise ValueError("hessian must have float32 dtype")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")

    columns = hessian.shape[0]
    if columns % block_size:
        raise ValueError("hessian width must be divisible by block_size")
    elements = columns * columns
    if elements > 0xFFFFFFFF:
        raise ValueError("hessian is too large for the Metal normalization kernel")
    if not _all_finite(hessian, stream=mx.cpu):
        raise ValueError("hessian must contain only finite values")

    try:
        factor = mx.linalg.cholesky(hessian, stream=mx.cpu)
        mx.eval(factor)
    except Exception as error:
        raise ValueError("hessian must be positive definite") from error

    factor_diagonal = mx.diag(factor, stream=mx.cpu)
    if not _all_finite(factor, stream=mx.cpu) or not bool(
        mx.all(factor_diagonal > 0, stream=mx.cpu).item()
    ):
        raise ValueError("hessian must be positive definite")

    block_count = columns // block_size
    factor_blocks = factor.reshape(
        block_count, block_size, block_count, block_size
    ).transpose(0, 2, 1, 3)
    block_indices = mx.arange(block_count, stream=mx.cpu)
    diagonal_blocks = factor_blocks[block_indices, block_indices]
    diagonal_inverse = mx.linalg.tri_inv(diagonal_blocks, stream=mx.cpu)
    mx.eval(diagonal_inverse)
    if not _all_finite(diagonal_inverse, stream=mx.cpu):
        raise ValueError("hessian block normalization produced nonfinite values")

    normalized = _exl3_block_ldl_normalize_kernel()(
        inputs=[mx.contiguous(factor), mx.contiguous(diagonal_inverse)],
        template=[
            ("ELEMENTS", elements),
            ("COLUMNS", columns),
            ("BLOCK_SIZE", block_size),
        ],
        grid=(elements, 1, 1),
        threadgroup=(min(elements, 256), 1, 1),
        output_shapes=[hessian.shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(normalized)
    if not _all_finite(normalized, stream=mx.gpu):
        raise ValueError("hessian block normalization produced nonfinite values")
    return normalized


__all__ = ["exl3_block_ldl_mlx"]
