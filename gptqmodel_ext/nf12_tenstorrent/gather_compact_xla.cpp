// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "program.h"

inline uint32_t expand_pair(uint32_t low, uint32_t meta, uint32_t base) {
    const uint32_t even = (meta & 0x87) + base;
    const uint32_t odd = (((meta << 1) & 0x80) | ((meta >> 3) & 7)) + base;
    return (low & 0xff) | (even << 8) | ((low & 0xff00) << 8) | (odd << 24);
}

void kernel_main() {
    constexpr uint32_t k = NF12_CONFIG(0);
    constexpr auto weight_args = TensorAccessorArgs<NF12_ACCESSORS(0)>();
    constexpr auto escape_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<escape_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    const auto weight = TensorAccessor(weight_args, NF12_ADDRESS(0), 4096);
    const auto escapes = TensorAccessor(escape_args, NF12_ADDRESS(1), 4096);
    const auto ids = TensorAccessor(ids_args, NF12_ADDRESS(2), 4096);
    const auto out = TensorAccessor(out_args, NF12_ADDRESS(3), 2048);
    NF12_RANGE(k / 128, 0, 0);
    if (start == end) {
        return;
    }

    const uint32_t scratch = get_write_ptr(24);
    const uint32_t packed_addr = scratch + 128;
    const uint32_t output_addr = packed_addr + 192;
    const uint32_t escape_addr = output_addr + 256;
    const auto* header = reinterpret_cast<volatile const uint32_t*>(scratch);
    auto* decoded = reinterpret_cast<volatile uint32_t*>(output_addr);
    noc_async_read(escapes.get_noc_addr(0), scratch, 64);
    noc_async_read(ids.get_noc_addr(0), scratch + 64, 64);
    noc_async_read_barrier();
    const uint32_t base = header[0];
    const uint32_t n = header[1];
    const uint32_t id = header[16];

    for (uint32_t stripe = start; stripe < end; ++stripe) {
        if (id >= n) {
            for (uint32_t i = 0; i < 64; ++i) {
                decoded[i] = 0;
            }
        } else {
            const uint32_t address = (id * (k / 128) + stripe) * 192;
            const uint32_t within = address % 4096;
            const uint32_t first = within <= 4096 - 192 ? 192 : 4096 - within;
            noc_async_read(weight.get_noc_addr(address / 4096, within), packed_addr, first);
            if (first != 192) {
                noc_async_read(weight.get_noc_addr(address / 4096 + 1), packed_addr + first, 192 - first);
            }
            noc_async_read_barrier();
            uint32_t pending = 0;
            for (uint32_t group = 0; group < 32; ++group) {
                const uint32_t chunk = packed_addr + group / 16 * 96;
                const auto* metas = reinterpret_cast<volatile const uint16_t*>(chunk);
                const auto* lows = reinterpret_cast<volatile const uint32_t*>(chunk + 32);
                const uint32_t meta = metas[group % 16];
                const uint32_t low = lows[group % 16];
                if (meta == 0) {
                    const uint32_t pos = 64 + low / 8 * 64;
                    noc_async_read(escapes.get_noc_addr(pos / 4096, pos % 4096), escape_addr + group * 64, 64);
                    pending |= 1u << group;
                    continue;
                }
                const uint32_t dst = (group / 4 % 2) * 32 + (group / 8) * 8 + group % 4;
                decoded[dst] = expand_pair(low, meta, base);
                decoded[dst + 4] = expand_pair(low >> 16, meta >> 8, base);
            }
            if (pending != 0) {
                noc_async_read_barrier();
            }
            while (pending != 0) {
                const uint32_t group = __builtin_ctz(pending);
                pending &= pending - 1;
                const auto* values = reinterpret_cast<volatile const uint32_t*>(escape_addr + group * 64);
                const auto* lows = reinterpret_cast<volatile const uint32_t*>(packed_addr + group / 16 * 96 + 32);
                const uint32_t entry = (lows[group % 16] % 8) * 2;
                const uint32_t dst = (group / 4 % 2) * 32 + (group / 8) * 8 + group % 4;
                decoded[dst] = values[entry];
                decoded[dst + 4] = values[entry + 1];
            }
        }

        // A 128-value stripe occupies four rows in each of two BF16 faces.
        // Each writer owns disjoint 128-byte ranges in the compact carrier.
        const uint32_t offset = (stripe / 4 % 2) * 1024 + (stripe % 4) * 128;
        noc_async_write(output_addr, out.get_noc_addr(stripe / 8, offset), 128);
        noc_async_write(output_addr + 128, out.get_noc_addr(stripe / 8, offset + 512), 128);
        noc_async_write_barrier();
    }
}
