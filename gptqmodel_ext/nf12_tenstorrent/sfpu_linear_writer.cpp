// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "sfpu_escape_patch.h"

template <bool dense, typename Output, typename Apply>
void write_results(const Output& output, uint32_t start, uint32_t end, const Apply& apply) {
    constexpr auto decoded = tt::CBIndex::c_2;
    constexpr auto decoded_raw = tt::CBIndex::c_16;
    constexpr auto final_cb = tt::CBIndex::c_17;
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);

    for (uint32_t n = start; n < end; ++n) {
        if constexpr (!dense) {
            // The raw UInt16 producer and the BF16 consumer alias one L1 slab.
            cb_wait_front(decoded_raw, k_tiles);
            apply(n, get_read_ptr(decoded_raw));
            cb_reserve_back(decoded, k_tiles);
            cb_push_back(decoded, k_tiles);
        }
        cb_wait_front(final_cb, 1);
        noc_async_write_page(n, output, get_read_ptr(final_cb));
        noc_async_write_barrier();
        cb_pop_front(final_cb, 1);
        if constexpr (!dense) {
            cb_pop_front(decoded_raw, k_tiles);
        }
    }
}

template <bool dense, bool has_escapes, uint32_t accessor_offset>
void write() {
    constexpr auto args = TensorAccessorArgs<accessor_offset>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t end = get_arg_val<uint32_t>(2);

    if constexpr (has_escapes) {
        constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
        constexpr auto header_args = TensorAccessorArgs<args.next_compile_time_args_offset()>();
        constexpr auto escape_args = TensorAccessorArgs<header_args.next_compile_time_args_offset()>();
        const auto headers = TensorAccessor(header_args, get_arg_val<uint32_t>(3));
        const auto escapes = TensorAccessor(escape_args, get_arg_val<uint32_t>(4));
        EscapePatches<decltype(headers), decltype(escapes)> patches{headers, escapes, get_write_ptr(24)};
        write_results<dense>(output, start, end, [&](uint32_t n, uint32_t address) {
            for (uint32_t g = 0; g < k_tiles / 4; ++g) {
                patches.apply(n * (k_tiles / 4) + g, address + g * 4 * 2048);
            }
        });
    } else {
        write_results<dense>(output, start, end, [](uint32_t, uint32_t) {});
    }
}

void kernel_main() { write<get_compile_time_arg_val(1) != 0, get_compile_time_arg_val(2) != 0, 3>(); }
