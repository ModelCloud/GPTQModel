// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

struct PairHighTable {
    uint32_t values[256] = {};

    constexpr explicit PairHighTable(uint32_t base) {
        for (uint32_t meta = 0; meta < 256; ++meta) {
            const uint32_t even = (meta & 0x87) + base;
            const uint32_t odd = (((meta << 1) & 0x80) | ((meta >> 3) & 0x07)) + base;
            values[meta] = (even << 8) | (odd << 24);
        }
    }
};

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t stripes = get_compile_time_arg_val(1);
    constexpr uint32_t base = get_compile_time_arg_val(2);
    constexpr uint32_t count = get_compile_time_arg_val(3);
    constexpr uint32_t chunk = get_compile_time_arg_val(4);
    constexpr uint32_t logical_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t ids_per_task = get_compile_time_arg_val(6);
    constexpr uint32_t padded_count = ((count + ids_per_task - 1) / ids_per_task) * ids_per_task;
    static constexpr PairHighTable high(base);
    constexpr auto packed_args = TensorAccessorArgs<7>();
    constexpr auto escapes_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<escapes_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    const auto packed = TensorAccessor(packed_args, get_arg_val<uint32_t>(0), stripes * 192);
    const auto escapes = TensorAccessor(escapes_args, get_arg_val<uint32_t>(1), 64);
    const auto ids = TensorAccessor(ids_args, get_arg_val<uint32_t>(2), 64);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(3), stripes * 256);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t end = get_arg_val<uint32_t>(5);
    const uint32_t packed_addr = get_write_ptr(24);
    const uint32_t ids_addr = packed_addr + chunk * 192;
    const uint32_t decoded_addr = ids_addr + 64;
    const uint32_t escape_indices_addr = decoded_addr + chunk * 256;
    const uint32_t escape_addr = escape_indices_addr + chunk * 32 * 8;
    const auto* id_words = reinterpret_cast<volatile const uint32_t*>(ids_addr);
    const auto* in = reinterpret_cast<volatile const uint32_t*>(packed_addr);
    auto* decoded = reinterpret_cast<volatile uint32_t*>(decoded_addr);
    auto* escape_indices = reinterpret_cast<volatile uint32_t*>(escape_indices_addr);

    uint32_t cached_id_page = 0xffffffff;
    uint32_t cached_stripe = 0xffffffff;
    uint32_t cached_id = 0xffffffff;
    for (uint32_t task = start * ids_per_task; task < end * ids_per_task; ++task) {
        const uint32_t stripe = task / padded_count * chunk;
        const uint32_t width = stripe + chunk <= stripes ? chunk : stripes - stripe;
        const uint32_t dst_row = task % padded_count;
        if constexpr (count != padded_count) {
            if (dst_row >= count) {
                continue;
            }
        }
        if (dst_row / 16 != cached_id_page) {
            cached_id_page = dst_row / 16;
            noc_async_read_page(cached_id_page, ids, ids_addr);
            noc_async_read_barrier();
        }
        const uint32_t id = id_words[dst_row % 16];
        if (id != cached_id || stripe != cached_stripe) {
            noc_async_write_barrier();
            cached_id = id;
            cached_stripe = stripe;
            if (id >= n) {
                for (uint32_t i = 0; i < width * 64; ++i) {
                    decoded[i] = 0;
                }
            } else {
                noc_async_read(packed.get_noc_addr(id, stripe * 192), packed_addr, width * 192);
                noc_async_read_barrier();
                uint32_t escaped = 0;
                uint32_t escape_pages = 0;
                uint32_t last_escape_page = 0xffffffff;
                for (uint32_t block = 0; block < width * 2; ++block) {
                    auto* dst = decoded + block * 32;
                    if (stripe * 2 + block >= logical_blocks) {
                        for (uint32_t pair = 0; pair < 32; ++pair) {
                            dst[pair] = 0;
                        }
                        continue;
                    }
                    const auto* src = in + block * 24;
                    const auto* meta = reinterpret_cast<volatile const uint16_t*>(src);
                    for (uint32_t group = 0; group < 16; ++group) {
                        const uint32_t pair = group / 4 * 8 + group % 4;
                        const uint32_t m = meta[group];
                        const uint32_t lows = src[8 + group];
                        if (m == 0) {
                            const uint32_t page = lows / 8;
                            if (page != last_escape_page) {
                                last_escape_page = page;
                                noc_async_read_page(page, escapes, escape_addr + escape_pages * 64);
                                ++escape_pages;
                            }
                            escape_indices[escaped * 2] = block * 32 + pair;
                            escape_indices[escaped * 2 + 1] = (escape_pages - 1) * 16 + (lows % 8) * 2;
                            ++escaped;
                            continue;
                        }
                        const uint32_t low0 = (lows & 0xff) | ((lows & 0xff00) << 8);
                        const uint32_t low1 = ((lows >> 16) & 0xff) | ((lows >> 8) & 0xff0000);
                        dst[pair] = low0 | high.values[m & 0xff];
                        dst[pair + 4] = low1 | high.values[m >> 8];
                    }
                }
                if (escaped != 0) {
                    noc_async_read_barrier();
                    const auto* values = reinterpret_cast<volatile const uint32_t*>(escape_addr);
                    for (uint32_t i = 0; i < escaped; ++i) {
                        const uint32_t pair = escape_indices[2 * i];
                        const uint32_t entry = escape_indices[2 * i + 1];
                        decoded[pair] = values[entry];
                        decoded[pair + 4] = values[entry + 1];
                    }
                }
            }
        }
        noc_async_write(decoded_addr, out.get_noc_addr(dst_row, stripe * 256), width * 256);
    }
    noc_async_write_barrier();
}
