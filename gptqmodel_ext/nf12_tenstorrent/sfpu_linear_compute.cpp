// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#include "sfpu_decode_math.h"

void kernel_main() {
    constexpr auto packed = tt::CBIndex::c_0;
    constexpr auto activation = tt::CBIndex::c_1;
    constexpr auto decoded = tt::CBIndex::c_2;
    constexpr auto decoded_raw = tt::CBIndex::c_16;
    constexpr auto output = tt::CBIndex::c_17;
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr bool dense = get_compile_time_arg_val(1);
    constexpr uint32_t base = get_compile_time_arg_val(2);
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(activation, decoded, output);
    cb_wait_front(activation, k_tiles);
    if constexpr (dense) {
        matmul_init(activation, decoded);
    }
    for (uint32_t n = 0; n < n_tiles; ++n) {
        if constexpr (!dense) {
            disable_fp32_dest_acc();
            reconfig_data_format<SrcOrder::Regular, false>(packed, packed);
            pack_reconfig_data_format<false, false>(decoded_raw);
            copy_init<false>(packed);
            for (uint32_t group = 0; group < k_tiles / 4; ++group) {
                cb_wait_front(packed, 3);
                tile_regs_acquire();
                for (uint32_t i = 0; i < 3; ++i) {
                    copy_tile<false>(packed, i, i);
                }
                MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, decode_face, 0, VectorMode::RC, base));
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(decoded_raw, 4);
                for (uint32_t i = 0; i < 4; ++i) {
                    pack_tile<false, false>(i, decoded_raw);
                }
                cb_push_back(decoded_raw, 4);
                tile_regs_release();
                cb_pop_front(packed, 3);
            }
            enable_fp32_dest_acc();
            reconfig_data_format<SrcOrder::Reverse, true>(activation, decoded);
            pack_reconfig_data_format<false, true>(output);
            matmul_init(activation, decoded);
        }
        cb_wait_front(decoded, k_tiles);
        tile_regs_acquire();
        for (uint32_t k = 0; k < k_tiles; ++k) {
            matmul_tiles(activation, decoded, k, k, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(output, 1);
        pack_tile(0, output);
        cb_push_back(output, 1);
        tile_regs_release();
        cb_pop_front(decoded, k_tiles);
    }
    cb_pop_front(activation, k_tiles);
}
