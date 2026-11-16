#include <cuda_fp16.h>

#include <c10/util/Exception.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <shared_mutex>

#include "../util.h"
#include "exl3_devctx.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_kernel_map_packed.cuh"
#include "comp_units/exl3_comp_unit_1.cuh"
#include "comp_units/exl3_comp_unit_2.cuh"
#include "comp_units/exl3_comp_unit_3.cuh"
#include "comp_units/exl3_comp_unit_4.cuh"
#include "comp_units/exl3_comp_unit_5.cuh"
#include "comp_units/exl3_comp_unit_6.cuh"
#include "comp_units/exl3_comp_unit_7.cuh"
#include "comp_units/exl3_comp_unit_8.cuh"

namespace {

struct TPackedTable
{
    const int* n_axis;
    int n_count;
    const uint16_t* payload;
};

// The tuned-kernel cache is process-wide and is filled on first use, so GIL=0
// callers can reach it from several threads at once.  An unsynchronized
// find/insert pair on a std::map is a data race that can corrupt the container.
// The steady state only reads, so a shared lock keeps concurrent dispatches
// parallel and the exclusive lock is taken only to publish a new key.  std::map
// nodes are stable and nothing erases, so the pointer returned to the caller
// stays valid after the lock is released.
std::map<uint64_t, TResult> tuning_cache = {};
std::shared_mutex tuning_cache_mutex;

// The forced-shape result is written per call, so it cannot be shared between
// threads: a second caller would overwrite the record the first is still
// reading.  Thread-local storage gives each caller its own record and keeps the
// existing pointer-returning contract.
thread_local TResult forced_result;

int exl3_gemm_tilesize_k[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim[] = {EXL3_GEMM_BLOCKDIM};

constexpr TPackedTable packed_table_128 = {
    exl3_packed::n_axis_128,
    exl3_packed::n_axis_len_128,
    exl3_packed::samples_128
};

constexpr TPackedTable packed_table_256 = {
    exl3_packed::n_axis_256,
    exl3_packed::n_axis_len_256,
    exl3_packed::samples_256
};

constexpr TPackedTable packed_table_512 = {
    exl3_packed::n_axis_512,
    exl3_packed::n_axis_len_512,
    exl3_packed::samples_512
};

int map_cc_to_index(int cc)
{
    switch (cc)
    {
        case CC_AMPERE: return 0;
        case CC_ADA: return 1;
        case CC_HOPPER: return 2;
        default: return -1;
    }
}

int nearest_axis_index(const int* axis, int axis_len, int value)
{
    int best_idx = 0;
    int64_t best_dist = axis[0] > value ? axis[0] - (int64_t) value : (int64_t) value - axis[0];

    for (int idx = 1; idx < axis_len; ++idx)
    {
        int64_t dist = axis[idx] > value ? axis[idx] - (int64_t) value : (int64_t) value - axis[idx];
        if (dist < best_dist)
        {
            best_dist = dist;
            best_idx = idx;
        }
    }

    return best_idx;
}

const TPackedTable& select_packed_table(int size_n)
{
    bool mod512 = (size_n % 512 == 0);
    bool mod256 = (size_n % 256 == 0);
    bool mod128 = (size_n % 128 == 0);
    TORCH_CHECK(mod128, "size_n must be a multiple of 128");

    if (mod512) return packed_table_512;
    if (mod256) return packed_table_256;
    return packed_table_128;
}

uint16_t lookup_packed_sample(const TPackedTable& table, int cc_idx, int bits, int k_idx, int n_idx)
{
    int flat_idx =
        ((((cc_idx * exl3_packed::bit_count) + (bits - 1)) * exl3_packed::k_axis_len) + k_idx) * table.n_count + n_idx;
    return table.payload[flat_idx];
}

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

    // Steady state: a shared lock, so dispatches for keys that are already
    // cached still run in parallel.  std::map nodes are stable and nothing
    // erases, so the returned pointer stays valid after the lock is released.
    {
        std::shared_lock<std::shared_mutex> guard(tuning_cache_mutex);
        auto lookup = tuning_cache.find(key);
        if (lookup != tuning_cache.end())
        {
            return &(lookup->second);
        }
    }

    // First use of this key: build the record and publish it under the exclusive
    // lock, so no concurrent reader can observe a half-written entry.  Re-check
    // first, because another thread may have inserted while this one waited.
    std::unique_lock<std::shared_mutex> guard(tuning_cache_mutex);
    auto lookup = tuning_cache.find(key);
    if (lookup == tuning_cache.end())
    {
        TORCH_CHECK(K >= 1 && K <= exl3_packed::bit_count, "Failed to find valid kernel for shape");
        int cc_idx = map_cc_to_index(cc);
        TORCH_CHECK(cc_idx >= 0, "Failed to find valid kernel for shape");

        const TPackedTable& table = select_packed_table(size_n);
        int k_idx = nearest_axis_index(exl3_packed::k_axis, exl3_packed::k_axis_len, size_k);
        int n_idx = nearest_axis_index(table.n_axis, table.n_count, size_n);
        uint16_t packed = lookup_packed_sample(table, cc_idx, K, k_idx, n_idx);
        int shape_idx = packed >> 8;
        int tuned_num_sms = packed & 0xff;
        TORCH_CHECK(shape_idx, "Failed to find valid kernel for shape");

        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n;
        int num_sms = MAX(MIN(max_slices, tuned_num_sms), 1);

        lookup = tuning_cache.emplace(key, TResult {
            get_gemm_kernel_ptr(K, shape_idx, c_fp32, cb),
            shape_idx,
            num_sms,
            exl3_gemm_blockdim[shape_idx]
        }).first;
    }

    return &(lookup->second);
}
