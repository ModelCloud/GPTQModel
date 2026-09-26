# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ method: Meituan, Ying Zhang et al., https://arxiv.org/abs/2406.09904

"""Opt-in native MLX QQQ weight quantization for Apple silicon."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _qqq_group_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_qqq_group_update",
        input_names=["weights", "hinv", "scales", "zeros"],
        output_names=["quantized", "errors"],
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
                                          SIGNED ? -7.0f : 0.0f,
                                          SIGNED ? 7.0f : 15.0f);
                float q = scale * (code - zero);
                float error = (value - q) / hinv[k * G + k];
                quantized[row * G + k] = q;
                errors[row * G + k] = error;
                for (int j = k + 1; j < G; ++j) {
                    values[j] -= error * hinv[k * G + j];
                }
            }
        """,
    )


@lru_cache(maxsize=1)
def _qqq_dynamic_group_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_qqq_dynamic_group_update",
        input_names=["weights", "hinv"],
        output_names=["quantized", "errors", "scales", "zeros"],
        source="""
            uint row = thread_position_in_grid.x;
            float values[G];
            float minimum = 0.0f;
            float maximum = 0.0f;
            for (int k = 0; k < G; ++k) {
                values[k] = weights[row * G + k];
                minimum = metal::min(minimum, values[k]);
                maximum = metal::max(maximum, values[k]);
            }
            maximum = metal::max(metal::abs(minimum), maximum);
            if (minimum < 0.0f) minimum = -maximum;
            if (minimum == 0.0f && maximum == 0.0f) {
                minimum = -1.0f;
                maximum = 1.0f;
            }
            float scale = (maximum - minimum) / 15.0f;
            float zero = 8.0f;
            scales[row] = scale;
            zeros[row] = zero;
            for (int k = 0; k < G; ++k) {
                float value = values[k];
                float code = metal::clamp(
                    metal::rint(value / scale) + zero, 0.0f, 15.0f);
                float q = scale * (code - zero);
                float error = (value - q) / hinv[k * G + k];
                quantized[row * G + k] = q;
                errors[row * G + k] = error;
                for (int j = k + 1; j < G; ++j) {
                    values[j] -= error * hinv[k * G + j];
                }
            }
        """,
    )


def _qqq_fixed_params(group):
    import mlx.core as mx

    minimum = mx.minimum(mx.min(group, axis=1), 0)
    maximum = mx.maximum(mx.max(group, axis=1), 0)
    maximum = mx.maximum(mx.abs(minimum), maximum)
    minimum = mx.where(minimum < 0, -maximum, minimum)
    empty = (minimum == 0) & (maximum == 0)
    minimum = mx.where(empty, -1, minimum)
    maximum = mx.where(empty, 1, maximum)
    return (maximum / 7)[:, None], mx.zeros((group.shape[0], 1))


def qqq_quantize_weight_mlx(weight, inverse_hessian, *, group_size=128):
    """Return QQQ pseudo-quantized weights, scales, zeros, and extra scales.

    ``group_size=-1`` uses signed 4-bit codes and a single fixed row scale.
    ``group_size=128`` uses dynamic unsigned groups with zero point eight.
    Inputs are preordered and use the same upper inverse-Hessian factor as
    QQQ's Torch quantizer; activation ordering and MSE search are caller tasks.
    The result is not packed into a QQQ inference layout.
    """
    import mlx.core as mx

    from .mlx_native import _accurate_matmul_mlx

    if weight.ndim != 2 or not weight.shape[0] or not weight.shape[1]:
        raise ValueError("weight must be a nonempty rank-2 matrix")
    rows, columns = weight.shape
    if group_size not in (-1, 128):
        raise ValueError("QQQ group_size must be -1 or 128")
    if group_size == 128 and columns % 128:
        raise ValueError("QQQ group_size 128 must divide input width")
    if inverse_hessian.shape != (columns, columns):
        raise ValueError("inverse_hessian must match input width")
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("weight must have a supported floating dtype")
    if inverse_hessian.dtype != mx.float32:
        raise ValueError("inverse_hessian must be float32")

    original = weight.astype(mx.float32)
    # QQQ's post-quantization int8 row scale is based on the original weight.
    maximum = mx.max(mx.abs(original), axis=1)
    scale_extra = mx.where(maximum == 0, 1, maximum)[:, None] / 127
    fixed_scale, fixed_zero = (
        _qqq_fixed_params(original) if group_size == -1 else (None, None)
    )
    remaining = original
    quantized, scales, zeros = [], [], []
    kernel = (
        _qqq_group_kernel() if group_size == -1 else _qqq_dynamic_group_kernel()
    )
    block = 128 if group_size == -1 else group_size
    for start in range(0, columns, block):
        end = min(start + block, columns)
        width = end - start
        group = mx.contiguous(remaining[:, :width])
        factor = mx.contiguous(inverse_hessian[start:end, start:end])
        if group_size == -1:
            q, error = kernel(
                inputs=[group, factor, fixed_scale, fixed_zero],
                template=[("G", width), ("SIGNED", True)],
                grid=(rows, 1, 1),
                threadgroup=(min(rows, 64), 1, 1),
                output_shapes=[(rows, width), (rows, width)],
                output_dtypes=[mx.float32, mx.float32],
            )
        else:
            q, error, scale, zero = kernel(
                inputs=[group, factor],
                template=[("G", width)],
                grid=(rows, 1, 1),
                threadgroup=(min(rows, 64), 1, 1),
                output_shapes=[(rows, width), (rows, width), (rows, 1), (rows, 1)],
                output_dtypes=[mx.float32] * 4,
            )
        quantized.append(q)
        if group_size != -1:
            scales.append(scale)
            zeros.append(zero)
        if end < columns:
            remaining = remaining[:, width:] - _accurate_matmul_mlx(
                error, inverse_hessian[start:end, end:]
            )
            mx.eval(remaining)
    result = (
        mx.concatenate(quantized, axis=1),
        fixed_scale if group_size == -1 else mx.concatenate(scales, axis=1),
        fixed_zero if group_size == -1 else mx.concatenate(zeros, axis=1),
        None if group_size == -1 else scale_extra,
    )
    mx.eval(*(value for value in result if value is not None))
    return result


__all__ = ["qqq_quantize_weight_mlx"]
