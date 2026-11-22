// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// decode.h is prepended by the host.

template <uint32_t bytes, typename Accessor>
inline void read_aligned(const Accessor& input, uint32_t offset, uint32_t address) {
    const uint32_t within = offset % 4096;
    const uint32_t first = within <= 4096 - bytes ? bytes : 4096 - within;
    noc_async_read(input.get_noc_addr(offset / 4096, within), address, first);
    if (first != bytes) {
        noc_async_read(input.get_noc_addr(offset / 4096 + 1), address + first, bytes - first);
    }
}

inline uint32_t expand_pair(uint32_t low, uint32_t meta, uint32_t base) {
    const uint32_t even = (meta & 0x87) + base;
    const uint32_t odd = (((meta << 1) & 0x80) | ((meta >> 3) & 7)) + base;
    return (low & 0xff) | (even << 8) | ((low & 0xff00) << 8) | (odd << 24);
}

template <typename Packed, typename Escapes, typename Ids, typename Out>
void gather_row_major(const Packed& packed, const Escapes& escapes, const Ids& ids, const Out& out,
                      uint32_t n, uint32_t base, uint32_t scratch) {
    constexpr uint32_t blocks = NF12_CONFIG(0) / 64;
    // Two adjacent 96-byte blocks naturally align on a 64-byte boundary for even row strides.
    constexpr uint32_t block_group = blocks % 2 == 0 ? 2 : 1;
    constexpr uint32_t read_bytes = block_group == 2 ? 192 : 128;
    constexpr uint32_t column_tasks = blocks / block_group;
    constexpr uint32_t output_tiles = 2 * block_group;
    constexpr uint32_t groups = 16 * block_group;
    NF12_RANGE(NF12_CONFIG(1) / output_tiles, 0, 0);
    const uint32_t id_addr = scratch + 32 * read_bytes;
    const uint32_t tile_addr = id_addr + 128;
    const uint32_t escape_addr = tile_addr + output_tiles * 2048;
    const auto* id_words = reinterpret_cast<volatile const uint32_t*>(id_addr);
    auto* tiles = reinterpret_cast<volatile uint32_t*>(tile_addr);
    uint32_t cached_id_tile = 0xffffffff;

    for (uint32_t task = start; task < end; ++task) {
        const uint32_t row_tile = task / column_tasks;
        if (row_tile != cached_id_tile) {
            cached_id_tile = row_tile;
            noc_async_read(ids.get_noc_addr(row_tile), id_addr, 64);
            noc_async_read(ids.get_noc_addr(row_tile, 1024), id_addr + 64, 64);
            noc_async_read_barrier();
        }
        const uint32_t block = task % column_tasks * block_group;
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t id = id_words[r];
            if (id < n && (r == 0 || id != id_words[r - 1])) {
                const uint32_t offset = (id * blocks + block) * 96;
                read_aligned<read_bytes>(packed, offset & ~63u, scratch + r * read_bytes);
            }
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t id = id_words[r];
            auto* row = tiles + tile_slot(r, 0) / 2;
            if (id >= n || (r != 0 && id == id_words[r - 1])) {
                const auto* previous = tiles + tile_slot((r + 31) % 32, 0) / 2;
                for (uint32_t face = 0; face < 2 * output_tiles; ++face) {
                    const uint32_t face_offset = (face / 2) * 512 + (face % 2) * 128;
                    for (uint32_t word = 0; word < 8; ++word) {
                        row[face_offset + word] = id >= n ? 0 : previous[face_offset + word];
                    }
                }
                continue;
            }

            const uint32_t offset = (id * blocks + block) * 96;
            const uint32_t record = scratch + r * read_bytes + (offset & 63u);
            uint32_t pending = 0;
            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t chunk = record + (group / 16) * 96;
                const auto* metas = reinterpret_cast<volatile const uint16_t*>(chunk);
                const auto* lows = reinterpret_cast<volatile const uint32_t*>(chunk + 32);
                const uint32_t meta = metas[group % 16];
                const uint32_t low = lows[group % 16];
                if (meta == 0) {
                    const uint32_t position = 64 + low / 8 * 64;
                    noc_async_read(escapes.get_noc_addr(position / 4096, position % 4096),
                                   escape_addr + group * 64, 64);
                    pending |= 1u << group;
                    continue;
                }
                const uint32_t dst = (group / 8) * 512 + (group / 4 % 2) * 128 + group % 4;
                row[dst] = expand_pair(low, meta, base);
                row[dst + 4] = expand_pair(low >> 16, meta >> 8, base);
            }
            if (pending != 0) {
                noc_async_read_barrier();
            }
            while (pending != 0) {
                const uint32_t group = __builtin_ctz(pending);
                pending &= pending - 1;
                const auto* escaped = reinterpret_cast<volatile const uint32_t*>(escape_addr + group * 64);
                const auto* lows = reinterpret_cast<volatile const uint32_t*>(record + (group / 16) * 96 + 32);
                const uint32_t entry = (lows[group % 16] % 8) * 2;
                const uint32_t dst = (group / 8) * 512 + (group / 4 % 2) * 128 + group % 4;
                row[dst] = escaped[entry];
                row[dst + 4] = escaped[entry + 1];
            }
        }
        for (uint32_t tile = 0; tile < output_tiles; ++tile) {
            noc_async_write_page(task * output_tiles + tile, out, tile_addr + tile * 2048);
        }
        noc_async_write_barrier();
    }
}

void kernel_main() {
    constexpr uint32_t cols = NF12_CONFIG(0) / 32;
    constexpr auto packed_args = TensorAccessorArgs<NF12_ACCESSORS(0)>();
    constexpr auto escape_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<escape_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    const auto packed = TensorAccessor(packed_args, NF12_ADDRESS(0), NF12_PACKED_PAGE);
    const auto escapes = TensorAccessor(escape_args, NF12_ADDRESS(1), NF12_ESCAPE_PAGE);
    const auto ids = TensorAccessor(ids_args, NF12_ADDRESS(2), 4096);
    const auto out = TensorAccessor(out_args, NF12_ADDRESS(3), 2048);
    NF12_RANGE(NF12_CONFIG(1), 0, 0);

    const uint32_t atom_addr = get_write_ptr(24);
    const uint32_t id_addr = atom_addr + 1536;
    const uint32_t tile_addr = id_addr + 128;
    const uint32_t escape_addr = tile_addr + 2048;
    noc_async_read(escapes.get_noc_addr(0), atom_addr, 64);
    noc_async_read_barrier();
    const auto* metadata = reinterpret_cast<volatile const uint32_t*>(atom_addr);
    const uint32_t base = metadata[0];
    const uint32_t n = metadata[1];
    if (metadata[3] == 1) {
        gather_row_major(packed, escapes, ids, out, n, base, atom_addr);
        return;
    }
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
            if (id >= n) {
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
