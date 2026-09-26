# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 trellis format: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx

"""Native MLX packing for EXL3 trellis states."""

from functools import lru_cache


@lru_cache(maxsize=8)
def _exl3_pack_trellis_kernel(bits: int):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_pack_trellis",
        input_names=["encoded"],
        output_names=["packed"],
        source="""
            uint index = thread_position_in_grid.x;
            uint physical_word = index % PACKED_WORDS;
            uint tile = index / PACKED_WORDS;

            // EXL3 swaps each adjacent uint16 pair when it stores the logical
            // big-endian bitstream in little-endian uint32 words.
            uint logical_word = physical_word ^ 1u;
            uint span = logical_word / BITS;
            uint word_in_span = logical_word % BITS;
            uint value = 0;

            for (uint output_bit = 0; output_bit < 16; ++output_bit) {
                uint stream_bit = word_in_span * 16 + output_bit;
                uint symbol = stream_bit / BITS;
                uint symbol_bit = stream_bit % BITS;
                uint state = uint(encoded[tile * 256 + span * 16 + symbol]);
                uint bit = (state >> (BITS - 1 - symbol_bit)) & 1u;
                value |= bit << (15 - output_bit);
            }
            packed[index] = as_type<short>(ushort(value));
        """,
    )


def exl3_pack_trellis_mlx(encoded, *, bits: int):
    """Pack EXL3's 256-state tiles into its checkpoint trellis layout.

    ``encoded`` must be a nonempty rank-3 MLX int16 array with shape
    ``(input_features // 16, output_features // 16, 256)``. Only each state's
    low ``bits`` bits are stored, matching EXL3's CUDA ``pack_trellis`` format.
    The returned int16 array has the same first two dimensions and
    ``16 * bits`` packed words per tile.

    This is a checkpoint-finalization primitive. Codebook search, Hessian
    processing, and the EXL3 optimization loop remain outside this function.
    """
    import mlx.core as mx

    encoded = mx.array(encoded)
    if not isinstance(bits, int) or isinstance(bits, bool) or bits not in range(1, 9):
        raise ValueError("EXL3 bits must be an integer between 1 and 8")
    if encoded.ndim != 3 or encoded.shape[-1] != 256:
        raise ValueError("encoded must have shape (tile_rows, tile_columns, 256)")
    if encoded.shape[0] == 0 or encoded.shape[1] == 0:
        raise ValueError("encoded must contain at least one tile")
    if encoded.dtype != mx.int16:
        raise ValueError("encoded must have int16 dtype")

    packed_words = 16 * bits
    output_shape = (*encoded.shape[:-1], packed_words)
    output_size = encoded.shape[0] * encoded.shape[1] * packed_words
    packed = _exl3_pack_trellis_kernel(bits)(
        inputs=[mx.contiguous(encoded)],
        template=[("BITS", bits), ("PACKED_WORDS", packed_words)],
        grid=(output_size, 1, 1),
        threadgroup=(min(output_size, 256), 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.int16],
    )[0]
    mx.eval(packed)
    return packed


__all__ = ["exl3_pack_trellis_mlx"]
