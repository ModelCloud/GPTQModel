# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Native MLX quantization primitives for EXL3 states and trellises."""

from functools import lru_cache

_EXL3_CODEBOOKS = {"3inst": 0, "mcg": 1, "mul1": 2}

_EXL3_VITERBI_HEADER = r"""
inline half exl3_decode_state(uint state, uint codebook) {
    if (codebook == 0) {
        uint raw = state * 89226354u + 64248484u;
        raw = 0x3b603b60u ^ (raw & 0x8fff8fffu);
        half low = as_type<half>(ushort(raw & 0xffffu));
        half high = as_type<half>(ushort(raw >> 16));
        return low + high;
    }
    if (codebook == 1) {
        uint raw = state * 0xcbac1fedu;
        raw = 0x3b603b60u ^ (raw & 0x8fff8fffu);
        half low = as_type<half>(ushort(raw & 0xffffu));
        half high = as_type<half>(ushort(raw >> 16));
        return low + high;
    }

    uint raw = state * 0x83dcd12du;
    uint byte_sum = (raw & 0xffu)
        + ((raw >> 8) & 0xffu)
        + ((raw >> 16) & 0xffu)
        + ((raw >> 24) & 0xffu);
    half accumulator = as_type<half>(ushort(byte_sum + 0x6400u));
    half inverse = as_type<half>(ushort(0x1eeeu));
    half bias = as_type<half>(ushort(0xc931u));
    return metal::fma(accumulator, inverse, bias);
}
"""


@lru_cache(maxsize=24)
def _exl3_viterbi_kernel(bits: int, codebook_id: int):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_viterbi",
        input_names=["input_tiles"],
        output_names=["quantized", "indices", "temp_costs", "temp_edges"],
        header=_EXL3_VITERBI_HEADER,
        source=r"""
            uint lane = thread_position_in_threadgroup.x;
            uint slot = threadgroup_position_in_grid.x;

            threadgroup half tile_values[256];
            threadgroup half reduction_costs[THREADS];
            threadgroup ushort reduction_edges[THREADS];
            threadgroup ushort shared_edge;
            threadgroup half local_costs[LOCAL_COST_COUNT];

            device half* slot_costs = temp_costs;
            if (!LOCAL_COSTS) slot_costs += slot * 2 * EDGES;
            device short* slot_edges = temp_edges + slot * 256 * EDGES;

            for (uint tile = slot; tile < NUM_TILES; tile += ACTIVE_GROUPS) {
                if (lane < 256) {
                    tile_values[lane] = half(input_tiles[tile * 256 + lane]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // First pass starts halfway around the cyclic tile and finds
                // the state immediately before logical position zero.
                for (uint step = 0; step < 256; ++step) {
                    uint ri = (step + 128) & 255u;
                    uint write_bank = 1u - (step & 1u);
                    uint read_bank = step & 1u;
                    half weight = tile_values[ri];

                    for (uint out_edge = lane; out_edge < EDGES; out_edge += THREADS) {
                        uint state = out_edge;
                        uint in_edge = state >> BITS;
                        half delta = exl3_decode_state(state, CODEBOOK) - weight;
                        half best_cost;
                        if (step == 0) {
                            best_cost = delta * delta;
                        } else {
                            half previous_cost = LOCAL_COSTS
                                ? local_costs[read_bank * EDGES + in_edge]
                                : slot_costs[read_bank * EDGES + in_edge];
                            best_cost = metal::fma(delta, delta, previous_cost);
                        }
                        ushort best_edge = ushort(in_edge);

                        for (uint symbol = 1; symbol < MAX_SYMBOLS; ++symbol) {
                            state = (symbol << STATE_SHIFT) | out_edge;
                            in_edge = state >> BITS;
                            delta = exl3_decode_state(state, CODEBOOK) - weight;
                            half cost;
                            if (step == 0) {
                                cost = delta * delta;
                            } else {
                                half previous_cost = LOCAL_COSTS
                                    ? local_costs[read_bank * EDGES + in_edge]
                                    : slot_costs[read_bank * EDGES + in_edge];
                                cost = metal::fma(delta, delta, previous_cost);
                            }
                            if (cost < best_cost) {
                                best_cost = cost;
                                best_edge = ushort(in_edge);
                            }
                        }

                        if (LOCAL_COSTS) {
                            local_costs[write_bank * EDGES + out_edge] = best_cost;
                        } else {
                            slot_costs[write_bank * EDGES + out_edge] = best_cost;
                        }
                        slot_edges[ri * EDGES + out_edge] = as_type<short>(best_edge);
                    }
                    if (LOCAL_COSTS) {
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    } else {
                        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                    }
                }

                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

                half local_cost = as_type<half>(ushort(0x7c00u));
                ushort local_edge = ushort(0xffffu);
                for (uint edge = lane; edge < EDGES; edge += THREADS) {
                    half cost = LOCAL_COSTS ? local_costs[edge] : slot_costs[edge];
                    if (cost < local_cost || (cost == local_cost && edge < local_edge)) {
                        local_cost = cost;
                        local_edge = ushort(edge);
                    }
                }
                reduction_costs[lane] = local_cost;
                reduction_edges[lane] = local_edge;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint offset = THREADS >> 1; offset > 0; offset >>= 1) {
                    if (lane < offset) {
                        half other_cost = reduction_costs[lane + offset];
                        ushort other_edge = reduction_edges[lane + offset];
                        half own_cost = reduction_costs[lane];
                        ushort own_edge = reduction_edges[lane];
                        if (other_cost < own_cost || (other_cost == own_cost && other_edge < own_edge)) {
                            reduction_costs[lane] = other_cost;
                            reduction_edges[lane] = other_edge;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (lane == 0) {
                    ushort edge = reduction_edges[0];
                    for (int step = 255; step >= 0; --step) {
                        uint ri = (uint(step) + 128) & 255u;
                        edge = as_type<ushort>(slot_edges[ri * EDGES + uint(edge)]);
                        if (ri == 0) break;
                    }
                    shared_edge = edge;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Second pass fixes the incoming edge at position zero and
                // backtracks from that same edge to enforce tail biting.
                for (uint step = 0; step < 256; ++step) {
                    uint write_bank = 1u - (step & 1u);
                    uint read_bank = step & 1u;
                    half weight = tile_values[step];

                    for (uint out_edge = lane; out_edge < EDGES; out_edge += THREADS) {
                        uint state = out_edge;
                        uint in_edge = state >> BITS;
                        half delta = exl3_decode_state(state, CODEBOOK) - weight;
                        half best_cost;
                        if (step == 0) {
                            best_cost = in_edge == uint(shared_edge)
                                ? delta * delta
                                : as_type<half>(ushort(0x7c00u));
                        } else {
                            half previous_cost = LOCAL_COSTS
                                ? local_costs[read_bank * EDGES + in_edge]
                                : slot_costs[read_bank * EDGES + in_edge];
                            best_cost = metal::fma(delta, delta, previous_cost);
                        }
                        ushort best_edge = ushort(in_edge);

                        for (uint symbol = 1; symbol < MAX_SYMBOLS; ++symbol) {
                            state = (symbol << STATE_SHIFT) | out_edge;
                            in_edge = state >> BITS;
                            delta = exl3_decode_state(state, CODEBOOK) - weight;
                            half cost;
                            if (step == 0) {
                                cost = in_edge == uint(shared_edge)
                                    ? delta * delta
                                    : as_type<half>(ushort(0x7c00u));
                            } else {
                                half previous_cost = LOCAL_COSTS
                                    ? local_costs[read_bank * EDGES + in_edge]
                                    : slot_costs[read_bank * EDGES + in_edge];
                                cost = metal::fma(delta, delta, previous_cost);
                            }
                            if (cost < best_cost) {
                                best_cost = cost;
                                best_edge = ushort(in_edge);
                            }
                        }

                        if (LOCAL_COSTS) {
                            local_costs[write_bank * EDGES + out_edge] = best_cost;
                        } else {
                            slot_costs[write_bank * EDGES + out_edge] = best_cost;
                        }
                        slot_edges[step * EDGES + out_edge] = as_type<short>(best_edge);
                    }
                    if (LOCAL_COSTS) {
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    } else {
                        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                    }
                }

                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

                if (lane == 0) {
                    ushort edge = shared_edge;
                    for (int step = 255; step >= 0; --step) {
                        ushort previous = as_type<ushort>(
                            slot_edges[uint(step) * EDGES + uint(edge)]
                        );
                        ushort encoded = ushort((uint(previous) << BITS) | uint(edge));
                        edge = previous;
                        indices[tile * 256 + uint(step)] = as_type<short>(encoded);
                        quantized[tile * 256 + uint(step)] = float(
                            exl3_decode_state(uint(encoded), CODEBOOK)
                        );
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        """,
    )


def exl3_quantize_tiles_mlx(
    input_tiles,
    *,
    bits: int,
    codebook: str = "mcg",
    workspace_bytes: int = 256 << 20,
):
    """Quantize 256-value EXL3 tiles with native MLX Viterbi search.

    The final dimension of ``input_tiles`` must be 256. Values are converted
    from float32 to float16 before search, matching EXL3's CUDA quantizer. The
    returned tuple contains float32 codebook values and unpacked int16 states,
    both with the same shape as the input.

    ``workspace_bytes`` bounds reusable Viterbi backtrace storage and does not
    grow with the number of input tiles.
    """
    import mlx.core as mx

    input_tiles = mx.array(input_tiles)
    if not isinstance(bits, int) or isinstance(bits, bool) or bits not in range(1, 9):
        raise ValueError("EXL3 bits must be an integer between 1 and 8")
    if not isinstance(codebook, str) or codebook.lower() not in _EXL3_CODEBOOKS:
        raise ValueError("EXL3 codebook must be '3inst', 'mcg', or 'mul1'")
    if input_tiles.ndim < 2 or input_tiles.shape[-1] != 256:
        raise ValueError(
            "input_tiles must have rank at least two with 256 values per tile"
        )
    if any(dimension == 0 for dimension in input_tiles.shape):
        raise ValueError("input_tiles must be nonempty")
    if input_tiles.dtype != mx.float32:
        raise ValueError("input_tiles must have float32 dtype")
    if (
        not isinstance(workspace_bytes, int)
        or isinstance(workspace_bytes, bool)
        or workspace_bytes <= 0
    ):
        raise ValueError("workspace_bytes must be a positive integer")

    tile_count = input_tiles.size // 256
    edges = 1 << (16 - bits)
    local_costs = bits >= 4
    bytes_per_slot = edges * (256 * 2 + (0 if local_costs else 2 * 2))
    if workspace_bytes < bytes_per_slot:
        raise ValueError(
            f"workspace_bytes must be at least {bytes_per_slot} for {bits}-bit EXL3"
        )
    active_groups = min(tile_count, 256, workspace_bytes // bytes_per_slot)
    threads = min(edges, 512)
    output_shape = input_tiles.shape
    outputs = _exl3_viterbi_kernel(bits, _EXL3_CODEBOOKS[codebook.lower()])(
        inputs=[mx.contiguous(input_tiles)],
        template=[
            ("BITS", bits),
            ("CODEBOOK", _EXL3_CODEBOOKS[codebook.lower()]),
            ("STATE_SHIFT", 16 - bits),
            ("MAX_SYMBOLS", 1 << bits),
            ("EDGES", edges),
            ("THREADS", threads),
            ("NUM_TILES", tile_count),
            ("ACTIVE_GROUPS", active_groups),
            ("LOCAL_COSTS", local_costs),
            ("LOCAL_COST_COUNT", 2 * edges if local_costs else 1),
        ],
        grid=(active_groups * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            output_shape,
            output_shape,
            (1,) if local_costs else (active_groups, 2, edges),
            (active_groups, 256, edges),
        ],
        output_dtypes=[mx.float32, mx.int16, mx.float16, mx.int16],
    )
    quantized, indices = outputs[:2]
    mx.eval(quantized, indices)
    return quantized, indices


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


__all__ = ["exl3_decode_states_mlx", "exl3_pack_trellis_mlx", "exl3_quantize_tiles_mlx"]
