// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "program.h"

void kernel_main() {
    constexpr uint32_t k_tiles = NF12_CONFIG(1);
    constexpr uint32_t m_tiles = NF12_CONFIG(2);
    constexpr uint32_t m_block = NF12_CONFIG(3);
    constexpr uint32_t tile_bytes = 2048;
    constexpr auto x_args = TensorAccessorArgs<NF12_ACCESSORS(6)>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, NF12_ADDRESS(0), tile_bytes);
    const auto w = TensorAccessor(w_args, NF12_ADDRESS(1), tile_bytes);
    NF12_RANGE(NF12_CONFIG(6), 3, 4);

    for (uint32_t nt = start; nt < end; ++nt) {
        for (uint32_t mt = 0; mt < m_tiles; mt += m_block) {
            const uint32_t count = m_block < m_tiles - mt ? m_block : m_tiles - mt;
            for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                cb_reserve_back(0, 1);
                cb_reserve_back(1, m_block);
                const uint32_t x_dst = get_write_ptr(1);
                for (uint32_t i = 0; i < count; ++i) {
                    noc_async_read_page((mt + i) * k_tiles + kt, x, x_dst + i * tile_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(1, m_block);
                noc_async_read_page(nt * k_tiles + kt, w, get_write_ptr(0));
                noc_async_read_barrier();
                cb_push_back(0, 1);
            }
        }
    }
}
