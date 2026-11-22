// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto packed_cb = tt::CBIndex::c_0;
    constexpr auto activation_cb = tt::CBIndex::c_1;
    constexpr auto decoded_cb = tt::CBIndex::c_2;
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr bool dense = get_compile_time_arg_val(1);
    constexpr uint32_t prefetch_groups = get_compile_time_arg_val(2);
    constexpr auto weights_args = TensorAccessorArgs<3>();
    constexpr auto input_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    const auto weights = TensorAccessor(weights_args, get_arg_val<uint32_t>(0));
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(1));
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t end = get_arg_val<uint32_t>(3);

    cb_reserve_back(activation_cb, k_tiles);
    const uint32_t activation_address = get_write_ptr(activation_cb);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        noc_async_read_page(k, input, activation_address + k * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(activation_cb, k_tiles);

    constexpr uint32_t block_tiles = dense ? 4 : 3;
    constexpr uint32_t prefetch_tiles = block_tiles * prefetch_groups;
    constexpr auto weight_cb = dense ? decoded_cb : packed_cb;
    for (uint32_t n = start; n < end; ++n) {
        for (uint32_t group = 0; group < k_tiles / 4; group += prefetch_groups) {
            cb_reserve_back(weight_cb, prefetch_tiles);
            const uint32_t packed_address = get_write_ptr(weight_cb);
            const uint32_t first_page = (n * (k_tiles / 4) + group) * block_tiles;
            for (uint32_t i = 0; i < prefetch_tiles; ++i) {
                noc_async_read_page(first_page + i, weights, packed_address + i * 2048);
            }
            noc_async_read_barrier();
            cb_push_back(weight_cb, prefetch_tiles);
        }
    }
}
