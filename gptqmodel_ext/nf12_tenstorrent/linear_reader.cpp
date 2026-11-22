// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// decode.h is prepended by the host.

template <typename Packed, typename Escapes>
inline void decode_tile(
    uint32_t nt, uint32_t kt, uint32_t dst, uint32_t scratch, const Packed& packed, const Escapes& escapes) {
    constexpr uint32_t n_atoms = NF12_CONFIG(0);
    constexpr uint32_t k_tiles = NF12_CONFIG(1);
    constexpr uint32_t base = NF12_CONFIG(4);
    auto* tile = reinterpret_cast<volatile uint16_t*>(dst);
    const auto* atom = reinterpret_cast<volatile const uint32_t*>(scratch);
    const uint32_t first_face = 2 * (kt % 2);
    const uint32_t escape_addr = scratch + 1536;
    for (uint32_t nh = 0; nh < 2; ++nh) {
        const uint32_t nb = nt * 2 + nh;
        if (nb >= n_atoms) {
            for (uint32_t row = 0; row < 32; ++row) {
                for (uint32_t col = 0; col < 16; ++col) {
                    tile[tile_slot(row, nh * 16 + col)] = 0;
                }
            }
            continue;
        }
        read_atom(nb * (k_tiles / 2) + kt / 2, packed, scratch);
        noc_async_read_barrier();
        for (uint32_t f = 0; f < 2; ++f) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
                fetch_escape(load_group(atom, first_face + f, lane), escapes, escape_addr + (32 * f + lane) * 64);
            }
        }
        noc_async_read_barrier();
        for (uint32_t f = 0; f < 2; ++f) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const auto group = load_group(atom, first_face + f, lane);
                const uint32_t entry = escape_addr + (32 * f + lane) * 64;
                for (uint32_t slot = 0; slot < 8; ++slot) {
                    const uint32_t n = nh * 16 + lane / 4 + 8 * ((slot / 2) % 2);
                    const uint32_t k = f * 16 + (lane % 4) * 2 + 8 * (slot / 4) + slot % 2;
                    tile[tile_slot(k, n)] = decode_slot(group, slot, base, entry);
                }
            }
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
                    decode_tile(nt, kt, get_write_ptr(0), scratch, w, escapes);
                }
                cb_push_back(0, 1);
            }
        }
    }
}
