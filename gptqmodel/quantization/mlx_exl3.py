# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx

"""Native MLX quantization primitives for EXL3 states and trellises."""

from functools import lru_cache

_EXL3_CODEBOOKS = {"3inst": 0, "mcg": 1, "mul1": 2}


@lru_cache(maxsize=3)
def _exl3_decode_kernel(codebook_id: int):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_decode_states",
        input_names=["encoded"],
        output_names=["decoded"],
        source="""
            uint index = thread_position_in_grid.x;
            uint state = uint(as_type<ushort>(encoded[index]));
            half result;

            if (CODEBOOK == 0) {
                uint raw = state * 89226354u + 64248484u;
                raw = 0x3b603b60u ^ (raw & 0x8fff8fffu);
                half low = as_type<half>(ushort(raw & 0xffffu));
                half high = as_type<half>(ushort(raw >> 16));
                result = low + high;
            } else if (CODEBOOK == 1) {
                uint raw = state * 0xcbac1fedu;
                raw = 0x3b603b60u ^ (raw & 0x8fff8fffu);
                half low = as_type<half>(ushort(raw & 0xffffu));
                half high = as_type<half>(ushort(raw >> 16));
                result = low + high;
            } else {
                uint raw = state * 0x83dcd12du;
                uint byte_sum = (raw & 0xffu)
                    + ((raw >> 8) & 0xffu)
                    + ((raw >> 16) & 0xffu)
                    + ((raw >> 24) & 0xffu);
                half accumulator = as_type<half>(ushort(byte_sum + 0x6400u));
                half inverse = as_type<half>(ushort(0x1eeeu));
                half bias = as_type<half>(ushort(0xc931u));
                result = metal::fma(accumulator, inverse, bias);
            }
            decoded[index] = float(result);
        """,
    )


def exl3_decode_states_mlx(encoded, *, codebook: str = "mcg"):
    """Decode EXL3 uint16 state bit patterns into quantized FP32 values.

    ``encoded`` must be a nonempty MLX int16 array whose last dimension is
    256. Signed values are interpreted by their underlying uint16 bit pattern,
    exactly as EXL3's CUDA quantizer does. ``codebook`` may be ``"3inst"``,
    ``"mcg"``, or ``"mul1"``. The output is float32 and has the same shape.

    This reconstructs the codebook-selected values after path search; it does
    not perform the Viterbi search itself.
    """
    import mlx.core as mx

    encoded = mx.array(encoded)
    if not isinstance(codebook, str) or codebook.lower() not in _EXL3_CODEBOOKS:
        raise ValueError("EXL3 codebook must be '3inst', 'mcg', or 'mul1'")
    normalized = codebook.lower()
    if encoded.ndim < 2 or encoded.shape[-1] != 256:
        raise ValueError("encoded must have rank at least two with 256 states per tile")
    if any(dimension == 0 for dimension in encoded.shape):
        raise ValueError("encoded must be nonempty")
    if encoded.dtype != mx.int16:
        raise ValueError("encoded must have int16 dtype")

    decoded = _exl3_decode_kernel(_EXL3_CODEBOOKS[normalized])(
        inputs=[mx.contiguous(encoded)],
        template=[("CODEBOOK", _EXL3_CODEBOOKS[normalized])],
        grid=(encoded.size, 1, 1),
        threadgroup=(min(encoded.size, 256), 1, 1),
        output_shapes=[encoded.shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(decoded)
    return decoded


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


__all__ = ["exl3_decode_states_mlx", "exl3_pack_trellis_mlx"]
