// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "program.h"
#include "sfpu_escape_patch.h"

template <bool dense, bool single_row, typename Output, typename Apply>
void write_results(const Output& output, uint32_t start, uint32_t end, const Apply& apply) {
    constexpr auto decoded = tt::CBIndex::c_2;
    constexpr auto decoded_raw = tt::CBIndex::c_16;
    constexpr auto final_cb = tt::CBIndex::c_17;
    constexpr uint32_t k_tiles = NF12_CONFIG(0);

    for (uint32_t n = start; n < end; ++n) {
        if constexpr (!dense) {
            // The raw UInt16 producer and the BF16 consumer alias one L1 slab.
            cb_wait_front(decoded_raw, k_tiles);
            apply(n, get_read_ptr(decoded_raw));
#ifdef NF12_PJRT
            cb_reserve_back(decoded, 1);
            cb_push_back(decoded, 1);
#else
            cb_reserve_back(decoded, k_tiles);
            cb_push_back(decoded, k_tiles);
#endif
        }
        cb_wait_front(final_cb, 1);
        const uint32_t tile_address = get_read_ptr(final_cb);
        if constexpr (single_row) {
            const uint32_t row_address = get_write_ptr(25);
            auto* row = reinterpret_cast<volatile uint32_t*>(row_address);
            const auto* tile = reinterpret_cast<volatile const uint32_t*>(tile_address);
            // Copy the first row of each of the two upper BF16 faces as raw words.
            for (uint32_t word = 0; word < 8; ++word) {
                row[word] = tile[word];
                row[8 + word] = tile[128 + word];
            }
            noc_async_write_page(n, output, row_address);
        } else {
            noc_async_write_page(n, output, tile_address);
        }
        noc_async_write_barrier();
        cb_pop_front(final_cb, 1);
        if constexpr (!dense) {
            cb_pop_front(decoded_raw, k_tiles);
        }
    }
}

template <bool dense, bool has_escapes, bool single_row, uint32_t accessor_offset>
void write() {
    constexpr auto args = TensorAccessorArgs<accessor_offset>();
    const auto output = TensorAccessor(args, NF12_ADDRESS(0), single_row ? 64 : 2048);
    NF12_RANGE(NF12_CONFIG(3), 1, 2);

    if constexpr (has_escapes) {
        constexpr uint32_t k_tiles = NF12_CONFIG(0);
        constexpr auto header_args = TensorAccessorArgs<args.next_compile_time_args_offset()>();
#ifdef NF12_PJRT
        constexpr uint32_t blocks = NF12_CONFIG(3) * (k_tiles / 4);
        const auto headers = TensorAccessor(header_args, NF12_ADDRESS(1), 4096);
        EscapePatches<decltype(headers), decltype(headers)> patches{
            headers, headers, get_write_ptr(24), (blocks + 7) / 8};
#else
        constexpr auto escape_args = TensorAccessorArgs<header_args.next_compile_time_args_offset()>();
        const auto headers = TensorAccessor(header_args, get_arg_val<uint32_t>(3));
        const auto escapes = TensorAccessor(escape_args, get_arg_val<uint32_t>(4));
        EscapePatches<decltype(headers), decltype(escapes)> patches{headers, escapes, get_write_ptr(24)};
#endif
        write_results<dense, single_row>(output, start, end, [&](uint32_t n, uint32_t address) {
            for (uint32_t g = 0; g < k_tiles / 4; ++g) {
                patches.apply(n * (k_tiles / 4) + g, address + g * 4 * 2048);
            }
        });
    } else {
        write_results<dense, single_row>(output, start, end, [](uint32_t, uint32_t) {});
    }
}

void kernel_main() {
#ifdef NF12_PJRT
    constexpr bool single_row = false;
#else
    constexpr bool single_row = NF12_CONFIG(3) != 0;
#endif
    write<NF12_CONFIG(1) != 0, NF12_CONFIG(2) != 0, single_row, NF12_ACCESSORS(4)>();
}
