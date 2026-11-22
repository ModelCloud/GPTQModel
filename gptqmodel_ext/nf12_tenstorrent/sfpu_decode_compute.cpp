// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

#include "sfpu_decode_math.h"

void kernel_main() {
    constexpr auto src = tt::CBIndex::c_0;
    constexpr auto dst = tt::CBIndex::c_16;
    constexpr bool dense = get_compile_time_arg_val(0);
    constexpr uint32_t inputs = dense ? 4 : 3;
    constexpr uint32_t base = get_compile_time_arg_val(1);
    const uint32_t blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(src, dst);
    if constexpr (!dense) {
        MATH(SFPU_UNARY_INIT(unused));
    }
    copy_init(src);
    for (uint32_t block = 0; block < blocks; ++block) {
        cb_wait_front(src, inputs);
        tile_regs_acquire();
        for (uint32_t i = 0; i < inputs; ++i) {
            copy_tile(src, i, i);
        }
        if constexpr (!dense) {
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, decode_face, 0, VectorMode::RC, base));
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(dst, 4);
        for (uint32_t i = 0; i < 4; ++i) {
            pack_tile(i, dst);
        }
        cb_push_back(dst, 4);
        cb_pop_front(src, inputs);
        tile_regs_release();
    }
}
