// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

#include "program.h"

void kernel_main() {
    constexpr uint32_t k_tiles = NF12_CONFIG(0);
    constexpr uint32_t m_tiles = NF12_CONFIG(1);
    constexpr uint32_t m_block = NF12_CONFIG(2);
    #ifdef NF12_PJRT
    NF12_RANGE(NF12_CONFIG(3), 0, 0);
    const uint32_t n_count = end - start;
#else
    const uint32_t n_count = get_arg_val<uint32_t>(0);
#endif
    compute_kernel_hw_startup<SrcOrder::Reverse>(1, 0, 16);
    matmul_init(1, 0);
    for (uint32_t nt = 0; nt < n_count; ++nt) {
        for (uint32_t mt = 0; mt < m_tiles; mt += m_block) {
            const uint32_t count = m_block < m_tiles - mt ? m_block : m_tiles - mt;
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                cb_wait_front(0, 1);
                cb_wait_front(1, m_block);
                for (uint32_t i = 0; i < count; ++i) {
                    matmul_tiles(1, 0, i, 0, i);
                }
                cb_pop_front(1, m_block);
                cb_pop_front(0, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < count; ++i) {
                cb_reserve_back(16, 1);
                pack_tile(i, 16);
                cb_push_back(16, 1);
            }
            tile_regs_release();
        }
    }
}
