# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Native MLX Hessian finalization for EXL3 quantization."""

from functools import lru_cache

from .mlx_exl3_block_ldl import exl3_block_ldl_mlx


@lru_cache(maxsize=2)
def _exl3_hessian_transform_kernel(axis: int):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name=f"gptqmodel_exl3_hessian_transform_axis_{axis}",
        input_names=["input", "signs", "parameters"],
        output_names=["output"],
        source=r"""
            threadgroup float current[128];
            threadgroup float next_values[128];

            uint lane = thread_position_in_threadgroup.x;
            uint vector = threadgroup_position_in_grid.x;
            uint row;
            uint column;
            if (AXIS == 1) {
                row = vector / COLUMN_BLOCKS;
                column = (vector % COLUMN_BLOCKS) * 128 + lane;
            } else {
                row = (vector / COLUMNS) * 128 + lane;
                column = vector % COLUMNS;
            }

            uint element = row * COLUMNS + column;
            float value = input[element];
            if (AXIS == 1) {
                value *= parameters[0];
                if (row == column) value += parameters[1];
                value *= signs[column];
            } else {
                value *= signs[row];
            }
            current[lane] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = 1; stride < 128; stride <<= 1) {
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

            // The two order-128 transforms contribute 1/sqrt(128) each.
            // Applying their combined factor once avoids an intermediate
            // rounding pass and the factor is exactly representable.
            output[element] = current[lane] * (AXIS == 0 ? 0.0078125f : 1.0f);
        """,
    )


def _all_finite(array, *, stream):
    import mlx.core as mx

    return bool(mx.all(mx.isfinite(array, stream=stream), stream=stream).item())


def _validate_transform_size(columns):
    if columns * columns > 0xFFFFFFFF:
        raise ValueError("hessian is too large for the Metal transform kernel")


def _transform_hessian(hessian, signs, parameters):
    import mlx.core as mx

    columns = hessian.shape[0]
    column_blocks = columns // 128
    right = _exl3_hessian_transform_kernel(1)(
        inputs=[mx.contiguous(hessian), mx.contiguous(signs), parameters],
        template=[
            ("AXIS", 1),
            ("COLUMNS", columns),
            ("COLUMN_BLOCKS", column_blocks),
        ],
        grid=(columns * column_blocks * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[hessian.shape],
        output_dtypes=[mx.float32],
    )[0]
    transformed = _exl3_hessian_transform_kernel(0)(
        inputs=[right, mx.contiguous(signs), parameters],
        template=[
            ("AXIS", 0),
            ("COLUMNS", columns),
            ("COLUMN_BLOCKS", column_blocks),
        ],
        grid=(column_blocks * columns * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[hessian.shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(transformed)
    return transformed


def exl3_finalize_hessian_mlx(
    hessian,
    signs,
    *,
    sample_count: int,
    sigma_reg: float = 0.025,
    block_size: int = 16,
):
    """Finalize an accumulated EXL3 Hessian and return its LDLQ inputs.

    ``hessian`` is the float32 sum captured during calibration and ``signs``
    contains one float32 ``-1`` or ``+1`` value per input channel. The result
    is ``(fallback, transformed_hessian, factor, regularized_diagonal)``.
    ``factor`` is ``None`` for an empty or numerically uncalibrated capture;
    otherwise it is the normalized block-lower factor with a zero scalar
    diagonal expected by EXL3 LDLQ.
    """
    import math

    import mlx.core as mx

    hessian = mx.array(hessian)
    signs = mx.array(signs)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square rank-two array")
    if hessian.shape[0] == 0:
        raise ValueError("hessian must be nonempty")
    if hessian.dtype != mx.float32:
        raise ValueError("hessian must have float32 dtype")
    columns = hessian.shape[0]
    if columns % 128:
        raise ValueError("hessian width must be divisible by 128")
    _validate_transform_size(columns)
    if signs.ndim != 1 or signs.shape[0] != columns:
        raise ValueError("signs must be a vector with one value per Hessian column")
    if signs.dtype != mx.float32:
        raise ValueError("signs must have float32 dtype")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    if columns % block_size:
        raise ValueError("hessian width must be divisible by block_size")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 0
    ):
        raise ValueError("sample_count must be a nonnegative integer")
    if not isinstance(sigma_reg, (int, float)) or isinstance(sigma_reg, bool):
        raise TypeError("sigma_reg must be a finite nonnegative number")
    sigma_reg = float(sigma_reg)
    if not math.isfinite(sigma_reg) or sigma_reg < 0:
        raise ValueError("sigma_reg must be a finite nonnegative number")
    if not _all_finite(hessian, stream=mx.cpu):
        raise ValueError("hessian must contain only finite values")
    if not _all_finite(signs, stream=mx.cpu) or not bool(
        mx.all(mx.abs(signs, stream=mx.cpu) == 1, stream=mx.cpu).item()
    ):
        raise ValueError("signs must contain only -1 or +1")

    if sample_count:
        inverse_count = 1.0 / sample_count
        base_diagonal = mx.diag(hessian, stream=mx.cpu) * inverse_count
        diagonal_mean = mx.mean(base_diagonal, stream=mx.cpu)
        mx.eval(base_diagonal, diagonal_mean)
        diagonal_mean_value = float(diagonal_mean.item())
        fallback = diagonal_mean_value < 1e-20
        regularization = sigma_reg * diagonal_mean_value
    else:
        inverse_count = 0.0
        base_diagonal = mx.zeros((columns,), dtype=mx.float32, stream=mx.cpu)
        fallback = True
        regularization = 0.0

    diagonal = base_diagonal + regularization
    parameters = mx.array([inverse_count, regularization], dtype=mx.float32)
    transformed = _transform_hessian(hessian, signs, parameters)
    mx.eval(diagonal)
    if not _all_finite(transformed, stream=mx.gpu) or not _all_finite(
        diagonal, stream=mx.cpu
    ):
        raise ValueError("Hessian finalization produced nonfinite values")

    if fallback:
        return True, transformed, None, diagonal

    factor = exl3_block_ldl_mlx(transformed, block_size=block_size)
    indices = mx.arange(columns, stream=mx.cpu)
    factor = factor.at[indices, indices].add(-mx.diag(factor, stream=mx.cpu))
    mx.eval(factor)
    if not _all_finite(factor, stream=mx.gpu):
        raise ValueError("Hessian factorization produced nonfinite values")
    return False, transformed, factor, diagonal


__all__ = ["exl3_finalize_hessian_mlx"]
