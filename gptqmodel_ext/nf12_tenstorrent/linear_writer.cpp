// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t n_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr auto args = TensorAccessorArgs<3>();
    const auto out = TensorAccessor(args, get_arg_val<uint32_t>(0), tile_bytes);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t end = get_arg_val<uint32_t>(2);
    for (uint32_t nt = start; nt < end; ++nt) {
        for (uint32_t mt = 0; mt < m_tiles; ++mt) {
            cb_wait_front(16, 1);
            noc_async_write_page(mt * n_tiles + nt, out, get_read_ptr(16));
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
}
