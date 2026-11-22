// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "sfpu_escape_patch.h"

template <bool has_escapes, uint32_t accessor_offset>
void write_tiles() {
    constexpr auto dst = tt::CBIndex::c_16;
    constexpr auto args = TensorAccessorArgs<accessor_offset>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t end = get_arg_val<uint32_t>(2);

    const auto write_block = [&](uint32_t block, uint32_t address) {
        for (uint32_t i = 0; i < 4; ++i) {
            noc_async_write_page(block * 4 + i, output, address + i * 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(dst, 4);
    };
    if constexpr (!has_escapes) {
        for (uint32_t block = start; block < end; ++block) {
            cb_wait_front(dst, 4);
            write_block(block, get_read_ptr(dst));
        }
    } else {
        constexpr auto header_args = TensorAccessorArgs<args.next_compile_time_args_offset()>();
        constexpr auto escape_args = TensorAccessorArgs<header_args.next_compile_time_args_offset()>();
        const auto headers = TensorAccessor(header_args, get_arg_val<uint32_t>(3));
        const auto escapes = TensorAccessor(escape_args, get_arg_val<uint32_t>(4));
        EscapePatches<decltype(headers), decltype(escapes)> patches{headers, escapes, get_write_ptr(24)};
        for (uint32_t block = start; block < end; ++block) {
            cb_wait_front(dst, 4);
            const uint32_t address = get_read_ptr(dst);
            patches.apply(block, address);
            write_block(block, address);
        }
    }
}

void kernel_main() { write_tiles<get_compile_time_arg_val(0) != 0, 1>(); }
