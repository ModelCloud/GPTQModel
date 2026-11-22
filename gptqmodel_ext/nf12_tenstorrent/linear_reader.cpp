// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// decode.h is prepended by the host.

struct GroupPairs {
    uint32_t pair0;
    uint32_t pair1;
    uint32_t pair2;
    uint32_t pair3;
};

inline uint32_t join_bf16_pair(uint32_t low, uint32_t high) {
    return (low & 0xff) | ((low & 0xff00) << 8) | high;
}

inline GroupPairs decode_regular_pairs(Group group, const uint32_t* exponent) {
    return {
        join_bf16_pair(group.low0, exponent[group.meta & 0xff]),
        join_bf16_pair(group.low0 >> 16, exponent[(group.meta >> 8) & 0xff]),
        join_bf16_pair(group.low1, exponent[(group.meta >> 16) & 0xff]),
        join_bf16_pair(group.low1 >> 16, exponent[group.meta >> 24]),
    };
}

inline GroupPairs decode_pairs(Group group, const uint32_t* exponent, uint32_t escape_page) {
    if (group.meta != 0) {
        return decode_regular_pairs(group, exponent);
    }
    const auto* entry = reinterpret_cast<volatile const uint32_t*>(escape_page + (group.low0 % 4) * 16);
    return {entry[0], entry[1], entry[2], entry[3]};
}

inline void store_rows(volatile uint32_t* output, uint32_t a, uint32_t b) {
    output[0] = (a & 0xffff) | (b << 16);
    output[8] = (a >> 16) | (b & 0xffff0000);
}

inline void store_group_pairs(volatile uint32_t* output, const GroupPairs& a, const GroupPairs& b) {
    store_rows(output, a.pair0, b.pair0);
    store_rows(output + 4, a.pair1, b.pair1);
    store_rows(output + 64, a.pair2, b.pair2);
    store_rows(output + 68, a.pair3, b.pair3);
}

template <typename Packed, typename Escapes>
inline void decode_tile(
    uint32_t nt, uint32_t kt, uint32_t dst, uint32_t scratch,
    uint32_t n_atoms, const uint32_t* exponent, const Packed& packed, const Escapes& escapes) {
    constexpr uint32_t k_tiles = NF12_CONFIG(1);
    auto* tile = reinterpret_cast<volatile uint32_t*>(dst);
    const auto* atom = reinterpret_cast<volatile const uint32_t*>(scratch);
    const uint32_t first_face = 2 * (kt % 2);
    const uint32_t escape_addr = scratch + 1536;
    for (uint32_t nh = 0; nh < 2; ++nh) {
        const uint32_t nb = nt * 2 + nh;
        if (nb >= n_atoms) {
            for (uint32_t i = 0; i < 128; ++i) {
                tile[nh * 128 + i] = 0;
                tile[(2 + nh) * 128 + i] = 0;
            }
            continue;
        }
        read_atom(nb * (k_tiles / 2) + kt / 2, packed, scratch);
        noc_async_read_barrier();
        uint8_t deferred[32];
        uint32_t pending = 0;
        for (uint32_t f = 0; f < 2; ++f) {
            auto* face = tile + (2 * f + nh) * 128;
            const uint32_t face_escapes = escape_addr + 32 * f * 64;
            // Two adjacent output features are four canonical lanes apart.
            // Write both in one word, in the [K, N] BF16 face order.
            for (uint32_t col_pair = 0; col_pair < 4; ++col_pair) {
                for (uint32_t row_pair = 0; row_pair < 4; ++row_pair) {
                    const uint32_t lane = col_pair * 8 + row_pair;
                    const auto a = load_group(atom, first_face + f, lane);
                    const auto b = load_group(atom, first_face + f, lane + 4);
                    if (a.meta != 0 && b.meta != 0) {
                        store_group_pairs(face + row_pair * 16 + col_pair,
                                          decode_regular_pairs(a, exponent), decode_regular_pairs(b, exponent));
                    } else {
                        fetch_escape(a, escapes, face_escapes + lane * 64);
                        fetch_escape(b, escapes, face_escapes + (lane + 4) * 64);
                        deferred[pending++] = f * 16 + col_pair * 4 + row_pair;
                    }
                }
            }
        }
        if (pending == 0) {
            continue;
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < pending; ++i) {
            const uint32_t f = deferred[i] / 16;
            const uint32_t col_pair = deferred[i] / 4 % 4;
            const uint32_t row_pair = deferred[i] % 4;
            const uint32_t lane = col_pair * 8 + row_pair;
            const uint32_t escape_page = escape_addr + (32 * f + lane) * 64;
            const auto a = load_group(atom, first_face + f, lane);
            const auto b = load_group(atom, first_face + f, lane + 4);
            store_group_pairs(tile + (2 * f + nh) * 128 + row_pair * 16 + col_pair,
                              decode_pairs(a, exponent, escape_page),
                              decode_pairs(b, exponent, escape_page + 4 * 64));
        }
    }
}

void kernel_main() {
    constexpr uint32_t k_tiles = NF12_CONFIG(1);
    constexpr uint32_t m_tiles = NF12_CONFIG(2);
    constexpr uint32_t m_block = NF12_CONFIG(3);
    constexpr bool dense = NF12_CONFIG(5) != 0;
    constexpr auto x_args = TensorAccessorArgs<NF12_ACCESSORS(6)>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto escape_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, NF12_ADDRESS(0), 2048);
    const auto w = TensorAccessor(w_args, NF12_ADDRESS(1), dense ? 2048 : NF12_PACKED_PAGE);
    const auto escapes = TensorAccessor(escape_args, NF12_ADDRESS(2), NF12_ESCAPE_PAGE);
    NF12_RANGE(NF12_CONFIG(6), 3, 4);
    const uint32_t scratch = get_write_ptr(24);
#ifdef NF12_PJRT
    noc_async_read(escapes.get_noc_addr(0), scratch, 64);
    noc_async_read_barrier();
    const auto* metadata = reinterpret_cast<volatile const uint32_t*>(scratch);
    const uint32_t base = metadata[0];
    const uint32_t n_atoms = metadata[2];
#else
    constexpr uint32_t base = NF12_CONFIG(4);
    constexpr uint32_t n_atoms = NF12_CONFIG(0);
#endif

    auto* exponent = reinterpret_cast<uint32_t*>(scratch + 1536 + 64 * 64);
    if constexpr (!dense) {
        for (uint32_t meta = 0; meta < 256; ++meta) {
            const uint32_t even = (meta & 0x87) + base;
            const uint32_t odd = (((meta << 1) & 0x80) | ((meta >> 3) & 7)) + base;
            exponent[meta] = (even << 8) | (odd << 24);
        }
    }

    for (uint32_t nt = start; nt < end; ++nt) {
        for (uint32_t mt = 0; mt < m_tiles; mt += m_block) {
            const uint32_t count = m_block < m_tiles - mt ? m_block : m_tiles - mt;
            for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                cb_reserve_back(0, 1);
                cb_reserve_back(1, m_block);
                uint32_t x_dst = get_write_ptr(1);
                for (uint32_t i = 0; i < count; ++i) {
                    noc_async_read_page((mt + i) * k_tiles + kt, x, x_dst + i * 2048);
                }
                noc_async_read_barrier();
                cb_push_back(1, m_block);
                if constexpr (dense) {
                    noc_async_read_page(nt * k_tiles + kt, w, get_write_ptr(0));
                    noc_async_read_barrier();
                } else {
                    decode_tile(nt, kt, get_write_ptr(0), scratch, n_atoms, exponent, w, escapes);
                }
                cb_push_back(0, 1);
            }
        }
    }
}
