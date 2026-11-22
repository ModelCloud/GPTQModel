// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "program.h"

void kernel_main() {
    constexpr auto packed_cb = tt::CBIndex::c_0;
    constexpr auto activation_cb = tt::CBIndex::c_1;
    constexpr auto decoded_cb = tt::CBIndex::c_2;
    constexpr uint32_t k_tiles = NF12_CONFIG(0);
    constexpr bool dense = NF12_CONFIG(1);
    constexpr uint32_t prefetch_groups = NF12_CONFIG(2);
    constexpr auto weights_args = TensorAccessorArgs<NF12_ACCESSORS(3)>();
    constexpr auto input_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    const auto weights = TensorAccessor(weights_args, NF12_ADDRESS(0),
#ifdef NF12_PJRT
                                        dense ? 2048 : 4096
#else
                                        2048
#endif
    );
    const auto input = TensorAccessor(input_args, NF12_ADDRESS(1), 2048);
    NF12_RANGE(NF12_CONFIG(3), 2, 3);
#ifdef NF12_PJRT
    constexpr uint32_t m_tiles = NF12_CONFIG(4);
    constexpr uint32_t m_group = NF12_CONFIG(5);
#else
    constexpr uint32_t m_tiles = 1;
    constexpr uint32_t m_group = 1;
#endif

    constexpr uint32_t block_tiles = dense ? 4 : 3;
    constexpr uint32_t prefetch_tiles = block_tiles * prefetch_groups;
    constexpr auto weight_cb = dense ? decoded_cb : packed_cb;
    for (uint32_t m = 0; m < m_tiles; m += m_group) {
        const uint32_t live = m + m_group <= m_tiles ? m_group : m_tiles - m;
        cb_reserve_back(activation_cb, m_group * k_tiles);
        const uint32_t activation_address = get_write_ptr(activation_cb);
        for (uint32_t tile = 0; tile < live * k_tiles; ++tile) {
            noc_async_read_page(m * k_tiles + tile, input, activation_address + tile * 2048);
        }
        noc_async_read_barrier();
        // Keep the circular buffer wrapped at its fixed capacity on a short last group.
        cb_push_back(activation_cb, m_group * k_tiles);

        for (uint32_t n = start; n < end; ++n) {
            for (uint32_t group = 0; group < k_tiles / 4; group += prefetch_groups) {
                cb_reserve_back(weight_cb, prefetch_tiles);
                const uint32_t packed_address = get_write_ptr(weight_cb);
                const uint32_t first_page = (n * (k_tiles / 4) + group) * block_tiles;
                for (uint32_t i = 0; i < prefetch_tiles; ++i) {
                    const uint32_t page = first_page + i;
                    const uint32_t destination = packed_address + i * 2048;
#ifdef NF12_PJRT
                    if constexpr (!dense) {
                        // PJRT carries two raw UInt16 planes per physical INT32 tile.
                        noc_async_read(weights.get_noc_addr(page / 2, page % 2 * 2048), destination, 2048);
                    } else
#endif
                    {
                        noc_async_read_page(page, weights, destination);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(weight_cb, prefetch_tiles);
            }
        }
    }
}
