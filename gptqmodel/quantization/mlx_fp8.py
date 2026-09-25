# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native MLX quantization of dense weights into supported FP8 byte formats."""

from functools import lru_cache


_FORMATS = {
    "float8_e4m3fn": (4, 3, 7, 127, True),
    "float8_e5m2": (5, 2, 15, 124, True),
    "float8_e4m3fnuz": (4, 3, 8, 128, False),
    "float8_e5m2fnuz": (5, 2, 16, 128, False),
}


def _positive_values(fmt):
    _, mantissa_bits, bias, count, _ = _FORMATS[fmt]
    mantissa_mask = (1 << mantissa_bits) - 1
    values = []
    for code in range(count):
        exponent = code >> mantissa_bits
        mantissa = code & mantissa_mask
        if exponent == 0:
            values.append(mantissa * 2.0 ** (1 - bias - mantissa_bits))
        else:
            values.append((1 + mantissa / (1 << mantissa_bits)) * 2.0 ** (exponent - bias))
    return values


@lru_cache(maxsize=4)
def _thresholds(fmt):
    import mlx.core as mx

    values = _positive_values(fmt)
    return mx.array([(left + right) / 2 for left, right in zip(values, values[1:])], dtype=mx.float32)


@lru_cache(maxsize=1)
def _fp8_encode_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_fp8_encode",
        input_names=["raw", "thresholds", "scales"],
        output_names=["codes"],
        source="""
            uint index = thread_position_in_grid.x;
            if (index >= SIZE) return;
            uint row = index / COLS;
            uint scale_index = MODE == 0 ? 0 :
                (MODE == 1 ? row :
                (row / BLOCK_ROWS) * GROUP_COLS + (index % COLS) / BLOCK_COLS);
            uint bits = as_type<uint>(raw[index]);
            uint magnitude_bits = bits & 0x7fffffffu;
            if (metal::isinf(scales[scale_index])) {
                codes[index] = magnitude_bits == 0 ? uchar(NAN_CODE) :
                    uchar((COUNT - 1) | ((bits >> 31) << 7));
                return;
            }
            float value = raw[index] * scales[scale_index];
            value = metal::clamp(value, -float(MAXQ), float(MAXQ));
            if (magnitude_bits > 0 && magnitude_bits < 0x00800000u) {
                // Scale the stored float32 mantissa before restoring its
                // subnormal exponent; ordinary GPU multiplication flushes it.
                float unit = scales[scale_index] * 0x1.0p-126f;
                float product = float(magnitude_bits) * unit * 0x1.0p-23f;
                value = metal::clamp((bits >> 31) ? -product : product,
                                     -float(MAXQ), float(MAXQ));
            }
            float magnitude = metal::abs(value);
            uint low = 0;
            uint high = COUNT - 1;
            while (low < high) {
                uint midpoint = (low + high) / 2;
                float boundary = thresholds[midpoint];
                if (magnitude > boundary ||
                    (magnitude == boundary && (midpoint & 1))) {
                    low = midpoint + 1;
                } else {
                    high = midpoint;
                }
            }
            uint sign = as_type<uint>(value) >> 31;
            codes[index] = uchar(low | ((sign && (SIGNED_ZERO || low)) ? 128 : 0));
        """,
    )


def quantize_fp8_weight_mlx(
    weight,
    *,
    format="float8_e4m3fn",
    weight_scale_method="row",
    weight_block_size=None,
):
    """Return FP8 storage bytes and inverse scales for a 2D MLX weight.

    This opt-in API matches ``quantize_fp8_weight`` without transferring the
    weight to Torch. Bytes are ready for FP8 checkpoint storage; the returned
    scale tensor follows the existing inverse-scale convention.
    """
    import mlx.core as mx

    from .config import _normalize_fp8_fmt, _normalize_fp8_weight_block_size, _normalize_fp8_weight_scale_method

    if weight.ndim != 2 or not all(weight.shape):
        raise ValueError("FP8 weight must be a nonempty rank-2 matrix")
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("FP8 weight must have float16, bfloat16, or float32 dtype")
    fmt = _normalize_fp8_fmt(format)
    if fmt not in _FORMATS:
        raise ValueError(f"FP8 quantization does not support {fmt}")
    block_size = _normalize_fp8_weight_block_size(weight_block_size)
    method = _normalize_fp8_weight_scale_method(weight_scale_method, weight_block_size=block_size)
    if not bool(mx.all(mx.isfinite(weight)).item()):
        raise ValueError("FP8 weight must be finite")

    matrix = weight.astype(mx.float32)
    fp8_max = _positive_values(fmt)[-1]
    tiny = 2.0 ** -126
    magnitude_bits = matrix.view(mx.uint32) & 0x7fffffff

    def inverse_scale(maximum_bits):
        maximum = maximum_bits.view(mx.float32)
        return mx.where(maximum_bits > 0, fp8_max / mx.maximum(maximum, tiny), 1.0)

    if method == "tensor":
        scales = inverse_scale(mx.max(magnitude_bits))
    elif method == "row":
        scales = inverse_scale(mx.max(magnitude_bits, axis=1))
    else:
        block_rows, block_cols = block_size
        rows, cols = matrix.shape
        if rows % block_rows or cols % block_cols:
            raise ValueError("FP8 weight shape must be divisible by weight_block_size")
        bit_blocks = magnitude_bits.reshape(rows // block_rows, block_rows, cols // block_cols, block_cols)
        scales = inverse_scale(mx.max(bit_blocks, axis=(1, 3)))

    thresholds = _thresholds(fmt)
    _, _, _, count, signed_zero = _FORMATS[fmt]
    block_rows, block_cols = block_size if block_size else (1, 1)
    codes = _fp8_encode_kernel()(
        inputs=[mx.contiguous(matrix), thresholds, mx.contiguous(scales.reshape(-1))],
        template=[
            ("SIZE", weight.size), ("COUNT", count), ("SIGNED_ZERO", int(signed_zero)),
            ("MAXQ", int(fp8_max)),
            ("NAN_CODE", 127 if signed_zero else 128), ("COLS", weight.shape[1]),
            ("MODE", {"tensor": 0, "row": 1, "block": 2}[method]),
            ("BLOCK_ROWS", block_rows), ("BLOCK_COLS", block_cols),
            ("GROUP_COLS", weight.shape[1] // block_cols),
        ],
        grid=(weight.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[weight.shape],
        output_dtypes=[mx.uint8],
    )[0]
    mx.eval(codes, scales)
    return codes, scales
