// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto src = tt::CBIndex::c_0;
    constexpr uint32_t inputs = get_compile_time_arg_val(0) ? 4 : 3;
    constexpr auto args = TensorAccessorArgs<1>();
    const auto packed = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t end = get_arg_val<uint32_t>(2);

    for (uint32_t block = start; block < end; ++block) {
        cb_reserve_back(src, inputs);
        const uint32_t address = get_write_ptr(src);
        for (uint32_t i = 0; i < inputs; ++i) {
            noc_async_read_page(block * inputs + i, packed, address + i * 2048);
        }
        noc_async_read_barrier();
        cb_push_back(src, inputs);
    }
}
