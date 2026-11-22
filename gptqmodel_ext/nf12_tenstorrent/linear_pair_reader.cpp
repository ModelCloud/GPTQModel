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

inline void fill_expansion(uint32_t* exponent, uint32_t base) {
    for (uint32_t meta = 0; meta < 256; ++meta) {
        const uint32_t even = (meta & 0x87) + base;
        const uint32_t odd = (((meta << 1) & 0x80) | ((meta >> 3) & 7)) + base;
        exponent[meta] = (even << 8) | (odd << 24);
    }
}

constexpr uint32_t atom_bytes = 1536;
constexpr uint32_t atoms_per_pair = 2;
constexpr uint32_t escape_bytes_per_atom = 4 * 32 * 64;
constexpr uint32_t tile_bytes = 2048;

// PJRT can reuse a kernel for buffers with the same rounded allocation.
// Native TTNN specializes the atom count without an extra reader argument.
#ifdef NF12_PJRT
#define NF12_ATOM_COUNT_PARAMETER uint32_t n_atoms,
#define NF12_ATOM_COUNT_ARGUMENT n_atoms,
#else
#define NF12_ATOM_COUNT_PARAMETER
#define NF12_ATOM_COUNT_ARGUMENT
#endif

template <typename Packed>
inline void read_pair(
    const Packed& packed, NF12_ATOM_COUNT_PARAMETER uint32_t n_tile, uint32_t k_atom, uint32_t scratch) {
#ifndef NF12_PJRT
    constexpr uint32_t n_atoms = NF12_CONFIG(0);
#endif
    constexpr uint32_t k_atoms = NF12_CONFIG(1) / 2;
    for (uint32_t n_half = 0; n_half < atoms_per_pair; ++n_half) {
        const uint32_t n_atom = n_tile * atoms_per_pair + n_half;
        if (n_atom < n_atoms) {
            read_atom(n_atom * k_atoms + k_atom, packed, scratch + n_half * atom_bytes);
        }
    }
}

inline volatile uint32_t* tile_face(uint32_t dst, uint32_t n_half, uint32_t atom_face) {
    const uint32_t tile = dst + (atom_face / 2) * tile_bytes;
    return reinterpret_cast<volatile uint32_t*>(tile) + (2 * (atom_face % 2) + n_half) * 128;
}

template <typename Escapes>
inline void decode_pair(
    NF12_ATOM_COUNT_PARAMETER uint32_t n_tile, uint32_t dst, uint32_t scratch, uint32_t escape_scratch,
    const Escapes& escapes, const uint32_t* exponent) {
#ifndef NF12_PJRT
    constexpr uint32_t n_atoms = NF12_CONFIG(0);
#endif
    uint8_t deferred[atoms_per_pair * 4 * 16];
    uint32_t pending = 0;
    for (uint32_t n_half = 0; n_half < atoms_per_pair; ++n_half) {
        if (n_tile * atoms_per_pair + n_half >= n_atoms) {
            for (uint32_t f = 0; f < 4; ++f) {
                auto* output = tile_face(dst, n_half, f);
                for (uint32_t i = 0; i < 128; ++i) {
                    output[i] = 0;
                }
            }
            continue;
        }
        const auto* atom = reinterpret_cast<volatile const uint32_t*>(scratch + n_half * atom_bytes);
        for (uint32_t f = 0; f < 4; ++f) {
            auto* output = tile_face(dst, n_half, f);
            const uint32_t face_escapes = escape_scratch + n_half * escape_bytes_per_atom + 32 * f * 64;
            for (uint32_t col_pair = 0; col_pair < 4; ++col_pair) {
#pragma GCC unroll 4
                for (uint32_t row_pair = 0; row_pair < 4; ++row_pair) {
                    const uint32_t lane = col_pair * 8 + row_pair;
                    const auto a = load_group(atom, f, lane);
                    const auto b = load_group(atom, f, lane + 4);
                    if (a.meta != 0 && b.meta != 0) {
                        store_group_pairs(output + row_pair * 16 + col_pair,
                                          decode_regular_pairs(a, exponent), decode_regular_pairs(b, exponent));
                    } else {
                        fetch_escape(a, escapes, face_escapes + lane * 64);
                        fetch_escape(b, escapes, face_escapes + (lane + 4) * 64);
                        deferred[pending++] = (n_half * 4 + f) * 16 + col_pair * 4 + row_pair;
                    }
                }
            }
        }
    }
    if (pending == 0) {
        return;
    }
    noc_async_read_barrier();
    for (uint32_t i = 0; i < pending; ++i) {
        const uint32_t n_half = deferred[i] / 64;
        const uint32_t face = deferred[i] / 16 % 4;
        const uint32_t col_pair = deferred[i] / 4 % 4;
        const uint32_t row_pair = deferred[i] % 4;
        const uint32_t lane = col_pair * 8 + row_pair;
        const auto* atom = reinterpret_cast<volatile const uint32_t*>(scratch + n_half * atom_bytes);
        const uint32_t entry = escape_scratch + n_half * escape_bytes_per_atom + (32 * face + lane) * 64;
        auto* output = tile_face(dst, n_half, face) + row_pair * 16 + col_pair;
        store_group_pairs(output, decode_pairs(load_group(atom, face, lane), exponent, entry),
                          decode_pairs(load_group(atom, face, lane + 4), exponent, entry + 4 * 64));
    }
}

void kernel_main() {
    constexpr uint32_t k_tiles = NF12_CONFIG(1);
    static_assert(k_tiles % 2 == 0, "Canonical NF12 pads K to full 64-column atoms");
    constexpr uint32_t k_atoms = k_tiles / 2;
    constexpr uint32_t m_tiles = NF12_CONFIG(2);
    constexpr uint32_t m_block = NF12_CONFIG(3);
    constexpr auto x_args = TensorAccessorArgs<NF12_ACCESSORS(6)>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto escape_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, NF12_ADDRESS(0), tile_bytes);
    const auto w = TensorAccessor(w_args, NF12_ADDRESS(1), NF12_PACKED_PAGE);
    const auto escapes = TensorAccessor(escape_args, NF12_ADDRESS(2), NF12_ESCAPE_PAGE);
    NF12_RANGE(NF12_CONFIG(6), 3, 4);
    constexpr uint32_t pair_bytes = atoms_per_pair * atom_bytes;
    const uint32_t scratch = get_write_ptr(24);
    const uint32_t escape_scratch = scratch + 2 * pair_bytes;
    auto* exponent = reinterpret_cast<uint32_t*>(escape_scratch + atoms_per_pair * escape_bytes_per_atom);
#ifdef NF12_PJRT
    noc_async_read(escapes.get_noc_addr(0), scratch, 64);
    noc_async_read_barrier();
    const auto* metadata = reinterpret_cast<volatile const uint32_t*>(scratch);
    const uint32_t base = metadata[0];
    const uint32_t n_atoms = metadata[2];
#else
    constexpr uint32_t base = NF12_CONFIG(4);
#endif
    fill_expansion(exponent, base);

    for (uint32_t nt = start; nt < end; ++nt) {
        for (uint32_t mt = 0; mt < m_tiles; mt += m_block) {
            const uint32_t count = m_block < m_tiles - mt ? m_block : m_tiles - mt;
            read_pair(w, NF12_ATOM_COUNT_ARGUMENT nt, 0, scratch);
            for (uint32_t ka = 0; ka < k_atoms; ++ka) {
                cb_reserve_back(0, 2);
                cb_reserve_back(1, 2 * m_block);
                const uint32_t x_dst = get_write_ptr(1);
                const uint32_t pair_scratch = scratch + (ka % 2) * pair_bytes;
                for (uint32_t i = 0; i < count; ++i) {
                    const uint32_t x_tile = (mt + i) * k_tiles + 2 * ka;
                    noc_async_read_page(x_tile, x, x_dst + i * tile_bytes);
                    noc_async_read_page(x_tile + 1, x, x_dst + (m_block + i) * tile_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(1, 2 * m_block);
                if (ka + 1 < k_atoms) {
                    read_pair(w, NF12_ATOM_COUNT_ARGUMENT nt, ka + 1, scratch + ((ka + 1) % 2) * pair_bytes);
                }
                decode_pair(NF12_ATOM_COUNT_ARGUMENT nt, get_write_ptr(0), pair_scratch, escape_scratch, escapes, exponent);
                cb_push_back(0, 2);
            }
        }
    }
}
