// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "program.h"

void kernel_main() {
    constexpr uint32_t n_tiles = NF12_CONFIG(0);
    constexpr uint32_t m_tiles = NF12_CONFIG(1);
    constexpr uint32_t tile_bytes = NF12_CONFIG(2);
    #ifdef NF12_PJRT
    constexpr auto x_args = TensorAccessorArgs<NF12_ACCESSORS(3)>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto escape_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto args = TensorAccessorArgs<escape_args.next_compile_time_args_offset()>();
#else
    constexpr auto args = TensorAccessorArgs<3>();
#endif
    const auto out = TensorAccessor(args, NF12_ADDRESS(0), tile_bytes);
    NF12_RANGE(n_tiles, 1, 2);
    for (uint32_t nt = start; nt < end; ++nt) {
        for (uint32_t mt = 0; mt < m_tiles; ++mt) {
            cb_wait_front(16, 1);
            noc_async_write_page(mt * n_tiles + nt, out, get_read_ptr(16));
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
}
