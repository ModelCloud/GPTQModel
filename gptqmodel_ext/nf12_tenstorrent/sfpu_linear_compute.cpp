// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#include "sfpu_decode_math.h"
#include "program.h"

#ifdef NF12_PJRT
// The bundled TT-XLA headers expose explicit accumulation width on the LLK
// calls. Their public compute wrappers specialize on the kernel's FP32 mode.
inline void raw_decode_init(uint32_t packed, uint32_t decoded) {
    UNPACK((llk_unpack_reconfig_data_format<false, detail::dim_stride_of(false)>(packed, packed)));
    MATH((llk_math_reconfig_data_format<false>(packed, packed)));
    PACK((llk_pack_reconfig_data_format<false>(decoded)));
    state_configure(packed, __builtin_LINE());
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        0, false, packed)));
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, false, BroadcastType::NONE>(packed)));
}

inline void raw_copy(uint32_t packed, uint32_t index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(packed, index)));
    MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, false, BroadcastType::NONE, UnpackToDestEn>(
        index, packed)));
}
#endif

void kernel_main() {
    constexpr auto packed = tt::CBIndex::c_0;
    constexpr auto activation = tt::CBIndex::c_1;
    constexpr auto decoded = tt::CBIndex::c_2;
    constexpr auto decoded_raw = tt::CBIndex::c_16;
    constexpr auto output = tt::CBIndex::c_17;
    constexpr uint32_t k_tiles = NF12_CONFIG(0);
    constexpr bool dense = NF12_CONFIG(1);
    constexpr uint32_t base = NF12_CONFIG(2);
#ifdef NF12_PJRT
    NF12_RANGE(NF12_CONFIG(3), 0, 0);
    const uint32_t n_tiles = end - start;
    constexpr uint32_t m_tiles = NF12_CONFIG(4);
    constexpr uint32_t m_group = NF12_CONFIG(5);
    constexpr uint32_t dest_tiles = 1;
#else
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t m_tiles = NF12_CONFIG(3);
    constexpr uint32_t m_group = m_tiles;
    constexpr uint32_t dest_tiles = m_group < 4 ? m_group : 4;
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(activation, decoded, output);
    for (uint32_t m = 0; m < m_tiles; m += m_group) {
        const uint32_t live = m + m_group <= m_tiles ? m_group : m_tiles - m;
        cb_wait_front(activation, m_group * k_tiles);
        if constexpr (dense) {
            matmul_init(activation, decoded);
        }
        for (uint32_t n = 0; n < n_tiles; ++n) {
            if constexpr (!dense) {
                disable_fp32_dest_acc();
#ifdef NF12_PJRT
                raw_decode_init(packed, decoded_raw);
#else
                reconfig_data_format<SrcOrder::Regular, false>(packed, packed);
                pack_reconfig_data_format<false, false>(decoded_raw);
                copy_init<false>(packed);
#endif
                for (uint32_t group = 0; group < k_tiles / 4; ++group) {
                    cb_wait_front(packed, 3);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < 3; ++i) {
#ifdef NF12_PJRT
                        raw_copy(packed, i);
#else
                        copy_tile<false>(packed, i, i);
#endif
                    }
                    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                        DST_SYNC_MODE, DST_ACCUM_MODE, decode_face, 0, VectorMode::RC, base));
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(decoded_raw, 4);
                    for (uint32_t i = 0; i < 4; ++i) {
#ifdef NF12_PJRT
                        PACK((llk_pack<false, false, PackMode::Default>(i, decoded_raw, 0)));
#else
                        pack_tile<false, false>(i, decoded_raw);
#endif
                    }
                    cb_push_back(decoded_raw, 4);
                    tile_regs_release();
                    cb_pop_front(packed, 3);
                }
                enable_fp32_dest_acc();
#ifdef NF12_PJRT
                reconfig_data_format<SrcOrder::Reverse>(activation, decoded);
                pack_reconfig_data_format(output);
#else
                reconfig_data_format<SrcOrder::Reverse, true>(activation, decoded);
                pack_reconfig_data_format<false, true>(output);
#endif
                matmul_init(activation, decoded);
            }
#ifdef NF12_PJRT
            // Artifact ABI v2 cannot alias two formats to one L1 allocation. Use
            // the BF16 CB as a ready token, then bind its unpack address to the
            // exact UInt16 slab only for the matmul. Both page sizes are 2 KiB.
            cb_wait_front(decoded, dense ? k_tiles : 1);
            UNPACK(uint32_t ready_ptr = 0;);
            if constexpr (!dense) {
                UNPACK(ready_ptr = get_local_cb_interface(decoded).fifo_rd_ptr;
                       get_local_cb_interface(decoded).fifo_rd_ptr = get_local_cb_interface(decoded_raw).fifo_rd_ptr;);
            }
#else
            cb_wait_front(decoded, k_tiles);
#endif
            // Keep the decoded weights in L1 and advance each destination in K order.
            for (uint32_t local_m = 0; local_m < live; local_m += dest_tiles) {
                const uint32_t count = live - local_m < dest_tiles ? live - local_m : dest_tiles;
                tile_regs_acquire();
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    for (uint32_t i = 0; i < count; ++i) {
                        matmul_tiles(activation, decoded, (local_m + i) * k_tiles + k, k, i);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < count; ++i) {
                    cb_reserve_back(output, 1);
                    pack_tile(i, output);
                    cb_push_back(output, 1);
                }
                tile_regs_release();
            }
#ifdef NF12_PJRT
            if constexpr (!dense) {
                UNPACK(get_local_cb_interface(decoded).fifo_rd_ptr = ready_ptr;);
            }
            cb_pop_front(decoded, dense ? k_tiles : 1);
#else
            cb_pop_front(decoded, k_tiles);
#endif
        }
        cb_pop_front(activation, m_group * k_tiles);
    }
}
