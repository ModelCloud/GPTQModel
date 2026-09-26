# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native MLX quantization for FOEM's alpha-zero group update."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _foem_group_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_foem_group_update",
        input_names=["weights", "raw", "hinv", "beta"],
        output_names=["quantized", "errors", "scales", "zeros"],
        source="""
            uint row = thread_position_in_grid.x;
            float values[G];
            float original[G];
            float minimum = 0.0f;
            float maximum = 0.0f;
            for (int k = 0; k < G; ++k) {
                values[k] = weights[row * G + k];
                original[k] = raw[row * G + k];
                minimum = metal::min(minimum, values[k]);
                maximum = metal::max(maximum, values[k]);
            }
            if (SYM) {
                maximum = metal::max(metal::abs(minimum), maximum);
                if (minimum < 0.0f) minimum = -maximum;
            }
            if (minimum == 0.0f && maximum == 0.0f) {
                minimum = -1.0f;
                maximum = 1.0f;
            }
            float scale = (maximum - minimum) / float(MAXQ);
            float zero = SYM ? float(1 << (BITS - 1))
                             : metal::rint(-minimum / scale);
            scales[row] = scale;
            zeros[row] = zero;
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
        q, error, scale, zero = kernel(
            inputs=[group, original, factor, beta_input],
            template=[
                ("G", group_size), ("BITS", bits),
                ("MAXQ", 2**bits - 1), ("SYM", int(sym)),
            ],
            grid=(rows, 1, 1),
            threadgroup=(min(rows, 64), 1, 1),
            output_shapes=[
                (rows, group_size), (rows, group_size), (rows, 1), (rows, 1),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
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
