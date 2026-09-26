# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 sign format: TurboDerp and ExLlamaV3 contributors.

"""Native MLX packing for compact EXL3 sign tensors."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _exl3_pack_signs_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_pack_signs",
        input_names=["signs"],
        output_names=["packed"],
        source=r"""
            uint index = thread_position_in_grid.x;
            if (index >= PACKED_WORDS) return;

            ushort value = 0;
            for (uint bit = 0; bit < 16; ++bit) {
                // EXL3's supported IEEE input types preserve their sign bit
                // when converted to half, including signed zero, infinities,
                // and NaNs. Avoid materializing that intermediate array.
                bool negative = metal::signbit(float(signs[index * 16 + bit]));
                value |= uint(negative) << bit;
            }
            packed[index] = as_type<short>(value);
        """,
    )


def exl3_pack_signs_mlx(signs):
    """Pack each group of 16 EXL3 signs into one little-endian int16 word.

    The input may have any nonempty shape and is flattened in row-major order.
    Float16, float32, and bfloat16 are supported. Each output bit records the
    sign bit of the corresponding value after EXL3's float16 conversion;
    signed zero and the signs of infinities and NaNs are preserved.
    """
    import mlx.core as mx

    signs = mx.array(signs)
    if signs.dtype not in (mx.float16, mx.float32, mx.bfloat16):
        raise ValueError("signs must have float16, float32, or bfloat16 dtype")
    if signs.size == 0:
        raise ValueError("signs must be nonempty")
    if signs.size % 16:
        raise ValueError("the flattened signs length must be divisible by 16")
    flattened = mx.contiguous(signs.reshape(-1))
    packed_words = signs.size // 16
    packed = _exl3_pack_signs_kernel()(
        inputs=[flattened],
        template=[("PACKED_WORDS", packed_words)],
        grid=(packed_words, 1, 1),
        threadgroup=(min(packed_words, 256), 1, 1),
        output_shapes=[(packed_words,)],
        output_dtypes=[mx.int16],
    )[0]
    mx.eval(packed)
    return packed


__all__ = ["exl3_pack_signs_mlx"]
