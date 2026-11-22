// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// decode.h is prepended by the host.

void kernel_main() {
    constexpr uint32_t n = NF12_CONFIG(0);
    constexpr uint32_t k = NF12_CONFIG(1);
    constexpr uint32_t count = NF12_CONFIG(2);
    constexpr uint32_t base = NF12_CONFIG(3);
    constexpr auto packed_args = TensorAccessorArgs<NF12_ACCESSORS(0)>();
    constexpr auto escape_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<escape_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    const auto packed = TensorAccessor(packed_args, NF12_ADDRESS(0), NF12_PACKED_PAGE);
    const auto escapes = TensorAccessor(escape_args, NF12_ADDRESS(1), NF12_ESCAPE_PAGE);
    const auto ids = TensorAccessor(ids_args, NF12_ADDRESS(2), 4096);
    const auto out = TensorAccessor(out_args, NF12_ADDRESS(3), 2048);
    constexpr uint32_t cols = k / 32;
    NF12_RANGE(((count + 31) / 32 * cols), 0, 0);

    const uint32_t atom_addr = get_write_ptr(24);
    const uint32_t id_addr = atom_addr + 1536;
    const uint32_t tile_addr = id_addr + 128;
    const uint32_t escape_addr = tile_addr + 2048;
    auto* tile = reinterpret_cast<volatile uint16_t*>(tile_addr);
    const auto* atom = reinterpret_cast<volatile const uint32_t*>(atom_addr);
    const auto* id_words = reinterpret_cast<volatile const uint32_t*>(id_addr);
    uint32_t cached_id_tile = 0xffffffff;
    uint32_t cached_atom = 0xffffffff;

    for (uint32_t task = start; task < end; ++task) {
        const uint32_t row_tile = task / cols;
        const uint32_t col_tile = task % cols;
        if (row_tile != cached_id_tile) {
            cached_id_tile = row_tile;
            noc_async_read(ids.get_noc_addr(row_tile), id_addr, 64);
            noc_async_read(ids.get_noc_addr(row_tile, 1024), id_addr + 64, 64);
            noc_async_read_barrier();
        }
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t id = id_words[r];
            if (row_tile * 32 + r >= count || id >= n) {
                for (uint32_t c = 0; c < 32; ++c) {
                    tile[tile_slot(r, c)] = 0;
                }
                continue;
            }
            const uint32_t atom_id = id / 16 * (cols / 2) + col_tile / 2;
            if (atom_id != cached_atom) {
                cached_atom = atom_id;
                read_atom(atom_id, packed, atom_addr);
                noc_async_read_barrier();
            }
            const uint32_t lane = id % 8 * 4;
            const uint32_t first_face = (col_tile % 2) * 2;
            const uint32_t slot = id % 16 >= 8 ? 2 : 0;
            for (uint32_t i = 0; i < 8; ++i) {
                fetch_escape(load_group(atom, first_face + i / 4, lane + i % 4), escapes, escape_addr + i * 64);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < 8; ++i) {
                const auto group = load_group(atom, first_face + i / 4, lane + i % 4);
                const uint32_t entry = escape_addr + i * 64;
                const uint32_t c = (i / 4) * 16 + 2 * (i % 4);
                tile[tile_slot(r, c)] = decode_slot(group, slot, base, entry);
                tile[tile_slot(r, c + 1)] = decode_slot(group, slot + 1, base, entry);
                tile[tile_slot(r, c + 8)] = decode_slot(group, slot + 4, base, entry);
                tile[tile_slot(r, c + 9)] = decode_slot(group, slot + 5, base, entry);
            }
        }
        noc_async_write_page(task, out, tile_addr);
        noc_async_write_barrier();
    }
}
