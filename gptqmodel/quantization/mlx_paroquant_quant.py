# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native MLX pseudo-quantization for ParoQuant transformed weights."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _paroquant_quantize_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_paroquant_pseudo_quantize",
        input_names=["weights", "scales", "zero_points"],
        output_names=["dequantized"],
        source="""
            uint index = thread_position_in_grid.x;
            uint row = index / COLUMNS;
            uint column = index % COLUMNS;
            uint group = row * GROUPS + column / GROUP_SIZE;
            float scale = metal::clamp(scales[group], 1e-5f, 1e5f);
            float value = weights[index] / scale;
            float qmin = SYMMETRIC ? -float(1 << (BITS - 1)) : 0.0f;
            float qmax = SYMMETRIC
                ? float((1 << (BITS - 1)) - 1)
                : float((1 << BITS) - 1);
            float zero = 0.0f;
            if (!SYMMETRIC) {
                zero = metal::clamp(metal::rint(-zero_points[group]), qmin, qmax);
            }
            float code = metal::clamp(metal::rint(value) + zero, qmin, qmax);
            dequantized[index] = (code - zero) * scale;
        """,
    )


def paroquant_quantize_weight_mlx(
    weight,
    scales,
    *,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = True,
    zero_point_float=None,
):
    """Return ParoQuant's transformed-domain pseudo-quantized weight on MLX.

    ``weight`` is ``(out_features, in_features)``. ``scales`` and optional
    asymmetric ``zero_point_float`` use either ``(out_features, groups)`` or
    ParoQuant's parameter shape ``(out_features * groups, 1)`` and float32
    dtype. Rounding uses ties-to-even and matches the non-STE export path in
    ``pseudo_quantize_dequant``. The output has the input weight dtype.
    """
    import mlx.core as mx

    if weight.ndim != 2 or not weight.shape[0] or not weight.shape[1]:
        raise ValueError("weight must be a nonempty rank-2 matrix")
    if not isinstance(bits, int) or isinstance(bits, bool) or bits not in range(2, 9):
        raise ValueError("ParoQuant bits must be between 2 and 8")
    rows, columns = weight.shape
    if not isinstance(group_size, int) or isinstance(group_size, bool):
        raise TypeError("group_size must be an integer")
    if group_size == -1:
        group_size = columns
    if group_size <= 0 or group_size % 2 or columns % group_size:
        raise ValueError("group_size must be positive, even, and divide input width")
    groups = columns // group_size
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("weight must have float16, bfloat16, or float32 dtype")
    parameter_shape = (rows, groups)
    flat_parameter_shape = (rows * groups, 1)
    if (
        scales.shape not in (parameter_shape, flat_parameter_shape)
        or scales.dtype != mx.float32
    ):
        raise ValueError(
            "scales must be float32 with shape (out_features, groups) "
            "or (out_features * groups, 1)"
        )
    scales = scales.reshape(parameter_shape)
    if not isinstance(sym, bool):
        raise TypeError("sym must be bool")
    if sym:
        if zero_point_float is not None:
            raise ValueError(
                "zero_point_float must be omitted for symmetric quantization"
            )
        zero_points = mx.zeros((rows, groups), dtype=mx.float32)
    else:
        if (
            zero_point_float is None
            or zero_point_float.shape not in (parameter_shape, flat_parameter_shape)
            or zero_point_float.dtype != mx.float32
        ):
            raise ValueError(
                "asymmetric quantization requires float32 zero_point_float "
                "with shape (out_features, groups)"
            )
        zero_points = zero_point_float
        zero_points = zero_points.reshape(parameter_shape)

    valid = mx.all(mx.isfinite(weight)) & mx.all(mx.isfinite(scales))
    if not sym:
        valid = valid & mx.all(mx.isfinite(zero_points))
    mx.eval(valid)
    if not bool(valid.item()):
        raise ValueError("weight and quantization parameters must be finite")

    kernel = _paroquant_quantize_kernel()
    result = kernel(
        inputs=[
            mx.contiguous(weight.astype(mx.float32).reshape(-1)),
            mx.contiguous(scales.reshape(-1)),
            mx.contiguous(zero_points.reshape(-1)),
        ],
        template=[
            ("BITS", bits),
            ("COLUMNS", columns),
            ("GROUPS", groups),
            ("GROUP_SIZE", group_size),
            ("SYMMETRIC", sym),
        ],
        grid=(weight.size, 1, 1),
        threadgroup=(min(weight.size, 256), 1, 1),
        output_shapes=[weight.shape],
        output_dtypes=[mx.float32],
    )[0]
    result = result.astype(weight.dtype)
    mx.eval(result)
    return result


__all__ = ["paroquant_quantize_weight_mlx"]
