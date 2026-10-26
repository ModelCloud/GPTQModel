# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Native MLX Hessian-compensated LDLQ for EXL3 weights."""

from functools import lru_cache

from .mlx_exl3 import exl3_quantize_tiles_mlx
from .mlx_exl3_tiles import (
    exl3_from_tensor_core_tiles_mlx,
    exl3_to_tensor_core_tiles_mlx,
)


@lru_cache(maxsize=1)
def _exl3_ldlq_product_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_ldlq_product",
        input_names=["factor", "error"],
        output_names=["product"],
        source=r"""
            uint index = thread_position_in_grid.x;
            if (index >= OUTPUT_ELEMENTS) return;

            uint target_row = index / COLUMNS;
            uint column = index % COLUMNS;
            float accumulator = 0.0f;
            for (uint reduction_row = 0; reduction_row < REDUCTION_ROWS; ++reduction_row) {
                float coefficient = factor[reduction_row * TARGET_ROWS + target_row];
                float residual = error[reduction_row * COLUMNS + column];
                accumulator = metal::fma(coefficient, residual, accumulator);
            }
            product[index] = accumulator;
        """,
    )


def _exl3_ldlq_product_mlx(factor, error):
    """Compute ``factor.T @ error`` with explicit float32 accumulation."""
    import mlx.core as mx

    factor = mx.contiguous(factor)
    error = mx.contiguous(error)
    reduction_rows, target_rows = factor.shape
    columns = error.shape[1]
    output_shape = (target_rows, columns)
    output_elements = target_rows * columns
    return _exl3_ldlq_product_kernel()(
        inputs=[factor, error],
        template=[
            ("REDUCTION_ROWS", reduction_rows),
            ("TARGET_ROWS", target_rows),
            ("COLUMNS", columns),
            ("OUTPUT_ELEMENTS", output_elements),
        ],
        grid=(output_elements, 1, 1),
        threadgroup=(min(output_elements, 256), 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]


def _validate_positive_multiple(value, *, name: str, multiple: int):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value % multiple
    ):
        raise ValueError(f"{name} must be a positive multiple of {multiple}")


def exl3_ldlq_quantize_mlx(
    weight,
    ldl_factor,
    *,
    bits: int,
    codebook: str = "mcg",
    buffer_rows: int = 128,
    workspace_bytes: int = 256 << 20,
):
    """Quantize an EXL3 weight with block-LDL error feedback.

    ``weight`` uses EXL3's ``(input_features, output_features)`` layout.
    ``ldl_factor`` is the normalized lower-triangular factor returned by
    EXL3's 16-row block LDL decomposition. Both arrays use float32 arithmetic.

    The implementation follows EXL3's descending buffered update order while
    retaining only compensation for rows that have not yet been quantized.
    It returns the reconstructed float32 weight and unpacked int16 trellis
    states in matrix and tensor-core tile order, respectively.
    """
    import mlx.core as mx

    weight = mx.array(weight)
    ldl_factor = mx.array(ldl_factor)
    if weight.dtype != mx.float32:
        raise ValueError("weight must have float32 dtype")
    if ldl_factor.dtype != mx.float32:
        raise ValueError("ldl_factor must have float32 dtype")
    if weight.ndim != 2 or any(dimension == 0 for dimension in weight.shape):
        raise ValueError("weight must be a nonempty rank-two array")
    if ldl_factor.ndim != 2 or ldl_factor.shape != (weight.shape[0], weight.shape[0]):
        raise ValueError("ldl_factor must be square with one row per input feature")

    rows, columns = weight.shape
    if rows % 16:
        raise ValueError("input features must be divisible by 16")
    if columns % 128:
        raise ValueError("output features must be divisible by 128")
    _validate_positive_multiple(buffer_rows, name="buffer_rows", multiple=16)
    if rows % buffer_rows:
        raise ValueError("input features must be divisible by buffer_rows")
    if columns % buffer_rows:
        raise ValueError("output features must be divisible by buffer_rows")

    # Compensation from already-quantized row buffers. After each outer
    # iteration this shrinks to the unquantized prefix, avoiding the source
    # implementation's writes to rows that can no longer be consumed.
    pending = mx.zeros(weight.shape, dtype=mx.float32)
    quantized_chunks = []
    encoded_chunks = []

    for stop in range(rows, 0, -buffer_rows):
        start = stop - buffer_rows
        source_chunk = weight[start:stop]
        chunk_compensation = pending[start:stop]
        factor_chunk = ldl_factor[start:stop]
        quantized_suffix = None
        encoded_blocks = []

        for block_stop in range(buffer_rows, 0, -16):
            block_start = block_stop - 16
            compensation = chunk_compensation[block_start:block_stop]
            if quantized_suffix is not None:
                suffix_error = source_chunk[block_stop:] - quantized_suffix
                coupling = factor_chunk[block_stop:, start + block_start : start + block_stop]
                compensation = compensation + _exl3_ldlq_product_mlx(
                    coupling, suffix_error
                )

            rows_to_quantize = source_chunk[block_start:block_stop] + compensation
            tiles = exl3_to_tensor_core_tiles_mlx(rows_to_quantize)
            quantized_tiles, encoded = exl3_quantize_tiles_mlx(
                tiles,
                bits=bits,
                codebook=codebook,
                workspace_bytes=workspace_bytes,
            )
            quantized_block = exl3_from_tensor_core_tiles_mlx(quantized_tiles)
            quantized_suffix = (
                quantized_block
                if quantized_suffix is None
                else mx.concatenate((quantized_block, quantized_suffix), axis=0)
            )
            encoded_blocks.append(encoded)
            mx.eval(quantized_suffix)

        encoded_chunk = mx.concatenate(tuple(reversed(encoded_blocks)), axis=0)
        mx.eval(quantized_suffix, encoded_chunk)
        quantized_chunks.append(quantized_suffix)
        encoded_chunks.append(encoded_chunk)

        if start:
            chunk_error = source_chunk - quantized_suffix
            pending = pending[:start] + _exl3_ldlq_product_mlx(
                factor_chunk[:, :start], chunk_error
            )
            mx.eval(pending)

    quantized_weight = mx.concatenate(tuple(reversed(quantized_chunks)), axis=0)
    encoded = mx.concatenate(tuple(reversed(encoded_chunks)), axis=0)
    mx.eval(quantized_weight, encoded)
    return quantized_weight, encoded


__all__ = ["exl3_ldlq_quantize_mlx"]
