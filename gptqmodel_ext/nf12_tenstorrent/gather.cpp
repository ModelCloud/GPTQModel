// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// decode.h is prepended by the host.

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t blocks = get_compile_time_arg_val(1);
    constexpr uint32_t base = get_compile_time_arg_val(2);
    constexpr auto packed_args = TensorAccessorArgs<3>();
    constexpr auto escape_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<escape_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    const auto packed = TensorAccessor(packed_args, get_arg_val<uint32_t>(0), 1536);
    const auto escapes = TensorAccessor(escape_args, get_arg_val<uint32_t>(1), 64);
    const auto ids = TensorAccessor(ids_args, get_arg_val<uint32_t>(2), 64);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(3), blocks * 128);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t end = get_arg_val<uint32_t>(5);
    const uint32_t atom_addr = get_write_ptr(24);
    const uint32_t id_addr = atom_addr + 1536;
    const uint32_t row_addr = id_addr + 64;
    const uint32_t escape_addr = row_addr + 128;
    auto* row = reinterpret_cast<volatile uint16_t*>(row_addr);
    const auto* atom = reinterpret_cast<volatile const uint32_t*>(atom_addr);
    uint32_t cached_ids = 0xffffffff;

    for (uint32_t task = start; task < end; ++task) {
        const uint32_t dst = task / blocks;
        const uint32_t kb = task % blocks;
        if (dst / 16 != cached_ids) {
            cached_ids = dst / 16;
            noc_async_read_page(cached_ids, ids, id_addr);
            noc_async_read_barrier();
        }
        const uint32_t id = reinterpret_cast<volatile const uint32_t*>(id_addr)[dst % 16];
        if (id >= n) {
            for (uint32_t col = 0; col < 64; ++col) {
                row[col] = 0;
            }
        } else {
            noc_async_read_page(id / 16 * blocks + kb, packed, atom_addr);
            noc_async_read_barrier();
            const uint32_t lane_start = 4 * (id % 8);
            const uint32_t slot = id % 16 >= 8 ? 2 : 0;
            for (uint32_t face = 0; face < 4; ++face) {
                for (uint32_t t = 0; t < 4; ++t) {
                    const auto group = load_group(atom, face, lane_start + t);
                    fetch_escape(group, escapes, escape_addr + (face * 4 + t) * 64);
                }
            }
            noc_async_read_barrier();
            for (uint32_t face = 0; face < 4; ++face) {
                for (uint32_t t = 0; t < 4; ++t) {
                    const auto group = load_group(atom, face, lane_start + t);
                    const uint32_t entry = escape_addr + (face * 4 + t) * 64;
                    const uint32_t col = 16 * face + 2 * t;
                    row[col] = decode_slot(group, slot, base, entry);
                    row[col + 1] = decode_slot(group, slot + 1, base, entry);
                    row[col + 8] = decode_slot(group, slot + 4, base, entry);
                    row[col + 9] = decode_slot(group, slot + 5, base, entry);
                }
            }
        }
        noc_async_write(row_addr, out.get_noc_addr(dst, kb * 128), 128);
        noc_async_write_barrier();
    }
}
