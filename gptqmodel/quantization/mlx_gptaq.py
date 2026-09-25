# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native MLX GPTAQ weight correction and group quantization."""

from functools import lru_cache
from math import isfinite


@lru_cache(maxsize=1)
def _gptaq_group_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_gptaq_group_update",
        input_names=["weights", "hinv", "correction", "scales", "zeros"],
        output_names=["quantized", "errors", "corrected"],
        source="""
            uint row = thread_position_in_grid.x;
            float values[G];
            for (int k = 0; k < G; ++k) {
                values[k] = weights[row * G + k];
            }
            float scale = scales[row];
            float zero = zeros[row];
            for (int k = 0; k < G; ++k) {
                float value = values[k];
                float code = metal::clamp(metal::rint(value / scale) + zero,
                                          0.0f, float(MAXQ));
                float q = scale * (code - zero);
                float error = (value - q) / hinv[k * G + k];
                quantized[row * G + k] = q;
                errors[row * G + k] = error;
                for (int j = k; j < G; ++j) {
                    values[j] -= error * hinv[k * G + j]
                               - value * correction[k * G + j];
                }
            }
            for (int k = 0; k < G; ++k) {
                corrected[row * G + k] = values[k];
            }
        """,
    )


def _group_params(group, bits, sym):
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
    zero = (
        mx.full(scale.shape, 2 ** (bits - 1), dtype=mx.float32)
        if sym
        else mx.round(-minimum / scale)
    )
    return scale[:, None], zero[:, None]


def gptaq_correction_mlx(delta_covariance, inverse_hessian, *, alpha=0.25):
    """Compute GPTAQ's strict-upper activation correction matrix on MLX."""
    import mlx.core as mx

    from .mlx_native import _accurate_matmul_mlx

    if (
        delta_covariance.ndim != 2
        or delta_covariance.shape[0] != delta_covariance.shape[1]
    ):
        raise ValueError("delta_covariance must be square")
    if inverse_hessian.shape != delta_covariance.shape:
        raise ValueError("inverse_hessian must match delta_covariance")
    if delta_covariance.dtype != mx.float32 or inverse_hessian.dtype != mx.float32:
        raise ValueError("correction inputs must be float32")
    if not isinstance(alpha, (float, int)) or not isfinite(alpha):
        raise ValueError("alpha must be a finite number")

    upper = mx.triu(_accurate_matmul_mlx(delta_covariance, inverse_hessian.T), k=1)
    scaled_upper = mx.array(alpha, dtype=mx.float32) * upper
    result = _accurate_matmul_mlx(scaled_upper, inverse_hessian)
    mx.eval(result)
    return result


def gptaq_quantize_weight_mlx(
    weight,
    inverse_hessian,
    correction,
    *,
    bits=4,
    group_size=128,
    sym=True,
):
    """Return GPTAQ pseudo-quantized weight, scales, and zero points.

    Inputs follow Torch's ``(out_features, in_features)`` layout. This path
    implements dynamic groups with MSE search and activation order disabled,
    and ``blocksize=group_size``. Build ``correction`` with
    :func:`gptaq_correction_mlx` from the calibration cross-covariance.
    """
    import mlx.core as mx

    from .mlx_native import _accurate_matmul_mlx

    if weight.ndim != 2 or not weight.shape[0] or not weight.shape[1]:
        raise ValueError("weight must be a nonempty rank-2 matrix")
    rows, columns = weight.shape
    if bits not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError("bits must be between 2 and 8")
    if group_size not in (16, 32, 64, 128) or columns % group_size:
        raise ValueError("group_size must be 16, 32, 64, or 128 and divide input width")
    if inverse_hessian.shape != (columns, columns) or correction.shape != (
        columns,
        columns,
    ):
        raise ValueError("inverse_hessian and correction must match input width")
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("weight must have a supported floating dtype")
    if inverse_hessian.dtype != mx.float32 or correction.dtype != mx.float32:
        raise ValueError("inverse_hessian and correction must be float32")
    if not isinstance(sym, bool):
        raise TypeError("sym must be bool")

    remaining = weight.astype(mx.float32)
    quantized, scales, zeros = [], [], []
    kernel = _gptaq_group_kernel()
    for start in range(0, columns, group_size):
        end = start + group_size
        group = mx.contiguous(remaining[:, :group_size])
        factor = mx.contiguous(inverse_hessian[start:end, start:end])
        correction_group = mx.contiguous(correction[start:end, start:end])
        scale, zero = _group_params(group, bits, sym)
        q, error, corrected = kernel(
            inputs=[group, factor, correction_group, scale, zero],
            template=[("G", group_size), ("MAXQ", 2**bits - 1)],
            grid=(rows, 1, 1),
            threadgroup=(min(rows, 64), 1, 1),
            output_shapes=[(rows, group_size)] * 3,
            output_dtypes=[mx.float32] * 3,
        )
        quantized.append(q)
        scales.append(scale)
        zeros.append(zero)
        if end < columns:
            remaining = (
                remaining[:, group_size:]
                - _accurate_matmul_mlx(error, inverse_hessian[start:end, end:])
                + _accurate_matmul_mlx(corrected, correction[start:end, end:])
            )
            mx.eval(remaining)

    result = (
        mx.concatenate(quantized, axis=1),
        mx.concatenate(scales, axis=1),
        mx.concatenate(zeros, axis=1),
    )
    mx.eval(*result)
    return result


__all__ = ["gptaq_correction_mlx", "gptaq_quantize_weight_mlx"]
