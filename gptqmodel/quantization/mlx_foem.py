# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native MLX quantization for FOEM's alpha-zero group update."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _foem_group_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_foem_group_update",
        input_names=["weights", "raw", "hinv", "scales", "zeros", "beta"],
        output_names=["quantized", "errors"],
        source="""
            uint row = thread_position_in_grid.x;
            float values[G];
            float original[G];
            for (int k = 0; k < G; ++k) {
                values[k] = weights[row * G + k];
                original[k] = raw[row * G + k];
            }
            float scale = scales[row];
            float zero = zeros[row];
            float correction = beta;
            for (int k = 0; k < G; ++k) {
                float value = values[k];
                float code = metal::clamp(metal::rint(value / scale) + zero,
                                          0.0f, float(MAXQ));
                float q = scale * (code - zero);
                float error = ((value - q) - (value - original[k]) * correction)
                              / hinv[k * G + k];
                quantized[row * G + k] = q;
                errors[row * G + k] = error;
                for (int j = k + 1; j < G; ++j) {
                    values[j] -= error * hinv[k * G + j];
                }
                if (k + 1 < G) {
                    values[k + 1] -= correction * (values[k + 1] - original[k + 1]);
                }
            }
        """,
    )


def _foem_group_params_mlx(group, *, bits: int, sym: bool):
    """Match Quantizer.find_params for per-channel weights with MSE disabled."""
    import mlx.core as mx

    minimum = mx.minimum(mx.min(group, axis=1), 0)
    maximum = mx.maximum(mx.max(group, axis=1), 0)
    if sym:
        maximum = mx.maximum(mx.abs(minimum), maximum)
        minimum = mx.where(minimum < 0, -maximum, minimum)
    empty = (minimum == 0) & (maximum == 0)
    minimum = mx.where(empty, -1, minimum)
    maximum = mx.where(empty, 1, maximum)
    scale = (maximum - minimum) / (2**bits - 1)
    zero = mx.full(scale.shape, 2 ** (bits - 1), dtype=mx.float32) if sym else mx.round(-minimum / scale)
    return scale[:, None], zero[:, None]


def foem_quantize_weight_mlx(
    weight,
    inverse_hessian,
    *,
    bits: int = 4,
    group_size: int = 128,
    beta: float = 0.2,
    sym: bool = True,
):
    """Return FOEM's pseudo-quantized weight, scales, and zero points on MLX.

    The input follows PyTorch's ``(out_features, in_features)`` convention.
    This opt-in path implements FOEM with ``alpha=0``, MSE search disabled,
    no activation-order permutation, and ``blocksize=group_size``. The Hessian
    input is the upper Cholesky factor returned by the existing inverse path.
    """
    import mlx.core as mx

    from .mlx_native import _accurate_matmul_mlx

    if weight.ndim != 2 or weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError("weight must be a nonempty rank-2 matrix")
    rows, columns = weight.shape
    if bits not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError("FOEM bits must be between 2 and 8")
    if group_size not in (16, 32, 64, 128) or columns % group_size:
        raise ValueError("FOEM group_size must be 16, 32, 64, or 128 and divide input width")
    if inverse_hessian.shape != (columns, columns):
        raise ValueError("inverse_hessian must match the input width")
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("weight must have a supported floating dtype")
    if inverse_hessian.dtype != mx.float32:
        raise ValueError("inverse_hessian must have float32 dtype")
    if not isinstance(sym, bool) or not isinstance(beta, (float, int)) or not 0 <= beta <= 1:
        raise ValueError("sym must be bool and beta must be in [0, 1]")
    if not bool(mx.all(mx.isfinite(weight)).item()):
        raise ValueError("weight must be finite")
    if not bool(mx.all(mx.isfinite(inverse_hessian)).item()):
        raise ValueError("inverse_hessian must be finite")
    if not bool(mx.all(mx.diag(inverse_hessian) > 0).item()):
        raise ValueError("inverse_hessian diagonal must be positive")

    raw = weight.astype(mx.float32)
    beta_input = mx.array(beta, dtype=mx.float32)
    remaining = raw
    quantized_groups, scales, zeros = [], [], []
    kernel = _foem_group_kernel()
    for start in range(0, columns, group_size):
        end = start + group_size
        group = mx.contiguous(remaining[:, :group_size])
        original = mx.contiguous(raw[:, start:end])
        factor = mx.contiguous(inverse_hessian[start:end, start:end])
        scale, zero = _foem_group_params_mlx(group, bits=bits, sym=sym)
        q, error = kernel(
            inputs=[group, original, factor, scale, zero, beta_input],
            template=[("G", group_size), ("MAXQ", 2**bits - 1)],
            grid=(rows, 1, 1),
            threadgroup=(min(rows, 64), 1, 1),
            output_shapes=[(rows, group_size), (rows, group_size)],
            output_dtypes=[mx.float32, mx.float32],
        )
        quantized_groups.append(q)
        scales.append(scale)
        zeros.append(zero)
        if end < columns:
            remaining = remaining[:, group_size:] - _accurate_matmul_mlx(
                error, inverse_hessian[start:end, end:],
            )
            mx.eval(remaining)

    result = (
        mx.concatenate(quantized_groups, axis=1),
        mx.concatenate(scales, axis=1),
        mx.concatenate(zeros, axis=1),
    )
    mx.eval(*result)
    return result
