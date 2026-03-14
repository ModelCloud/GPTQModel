#include <cuda_fp16.h>

#include <c10/util/Exception.h>
#include <cstdint>
#include <map>

#include "../util.h"
#include "exl3_kernel_map.cuh"
#include "comp_units/exl3_comp_unit_1.cuh"
#include "comp_units/exl3_comp_unit_2.cuh"
#include "comp_units/exl3_comp_unit_3.cuh"
#include "comp_units/exl3_comp_unit_4.cuh"
#include "comp_units/exl3_comp_unit_5.cuh"
#include "comp_units/exl3_comp_unit_6.cuh"
#include "comp_units/exl3_comp_unit_7.cuh"
#include "comp_units/exl3_comp_unit_8.cuh"

#include "exl3_kernel_map_samples.cuh"

namespace {

std::map<uint64_t, TResult> tuning_cache = {};
TResult forced_result;

int exl3_gemm_tilesize_k[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim[] = {EXL3_GEMM_BLOCKDIM};

fp_exl3_gemm_kernel get_gemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb)
{
    int kernel_idx = shape_idx + (EXL3_GEMM_NUM_SHAPES + 1) * cb;

    if (c_fp32)
    {
        switch (K)
        {
            case 1: return tfp_exl3_gemm_kernel_fp32_b1[kernel_idx];
            case 2: return tfp_exl3_gemm_kernel_fp32_b2[kernel_idx];
            case 3: return tfp_exl3_gemm_kernel_fp32_b3[kernel_idx];
            case 4: return tfp_exl3_gemm_kernel_fp32_b4[kernel_idx];
            case 5: return tfp_exl3_gemm_kernel_fp32_b5[kernel_idx];
            case 6: return tfp_exl3_gemm_kernel_fp32_b6[kernel_idx];
            case 7: return tfp_exl3_gemm_kernel_fp32_b7[kernel_idx];
            case 8: return tfp_exl3_gemm_kernel_fp32_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
    else
    {
        switch (K)
        {
            case 1: return tfp_exl3_gemm_kernel_fp16_b1[kernel_idx];
            case 2: return tfp_exl3_gemm_kernel_fp16_b2[kernel_idx];
            case 3: return tfp_exl3_gemm_kernel_fp16_b3[kernel_idx];
            case 4: return tfp_exl3_gemm_kernel_fp16_b4[kernel_idx];
            case 5: return tfp_exl3_gemm_kernel_fp16_b5[kernel_idx];
            case 6: return tfp_exl3_gemm_kernel_fp16_b6[kernel_idx];
            case 7: return tfp_exl3_gemm_kernel_fp16_b7[kernel_idx];
            case 8: return tfp_exl3_gemm_kernel_fp16_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }

    return nullptr;
}

}  // namespace

TResult* select_exl3_gemm_kernel_tuned
(
    int cc,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int force_shape_idx,
    int force_num_sms,
    int cb
)
{
    if (force_shape_idx > 0)
    {
        TORCH_CHECK(force_num_sms, "Must supply force_shape_idx and force_num_sms together");
        forced_result.kernel = get_gemm_kernel_ptr(K, force_shape_idx, c_fp32, cb);
        forced_result.shape_idx = force_shape_idx;
        forced_result.num_sms = force_num_sms;
        forced_result.block_dim = exl3_gemm_blockdim[force_shape_idx];
        return &forced_result;
    }
    TORCH_CHECK(!force_num_sms, "Must supply force_shape_idx and force_num_sms together.");

    // Cache by the dimensions that drive sample lookup plus cb/c_fp32 because they change kernel tables.
    uint64_t key = (((uint64_t) size_k) << 40) |
                   (((uint64_t) size_n) << 16) |
                   (((uint64_t) cc)     <<  8) |
                   (((uint64_t) K)      <<  4) |
                   (((uint64_t) cb)     <<  1) |
                   (c_fp32 ? 0x01ull : 0x00ull);

    auto lookup = tuning_cache.find(key);
    if (lookup == tuning_cache.end())
    {
        bool mod512 = (size_n % 512 == 0);
        bool mod256 = (size_n % 256 == 0);
        bool mod128 = (size_n % 128 == 0);
        TORCH_CHECK(mod128, "size_n must be a multiple of 128");

        TSample* cand = mod512 ? samples_512 : (mod256 ? samples_256 : samples_128);
        TSample* best = nullptr;
        int64_t best_dist = 1ll << 62;

        for (; cand->K; cand++)
        {
            if (cand->K != K) continue;
            if (cand->cc != cc) continue;

            int64_t distk = (int64_t) (size_k - cand->k);
            int64_t distn = (int64_t) (size_n - cand->n);
            int64_t dist = distk * distk + distn * distn;
            if (dist < best_dist)
            {
                best_dist = dist;
                best = cand;
            }
        }
        TORCH_CHECK(best, "Failed to find valid kernel for shape");

        int tilesize_k = exl3_gemm_tilesize_k[best->shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[best->shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n;
        int num_sms = MAX(MIN(max_slices, best->num_sms), 1);

        tuning_cache[key] = TResult {
            get_gemm_kernel_ptr(K, best->shape_idx, c_fp32, cb),
            best->shape_idx,
            num_sms,
            exl3_gemm_blockdim[best->shape_idx]
        };
    }

    lookup = tuning_cache.find(key);
    return &(lookup->second);
}
