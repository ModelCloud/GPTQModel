/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#ifndef MARLIN_GEMM_EXPORT_NAME
  #define MARLIN_GEMM_EXPORT_NAME gptq_marlin_gemm
#endif

#ifndef MARLIN_ENABLE_FP16
  #define MARLIN_ENABLE_FP16 1
#endif

#ifndef MARLIN_ENABLE_BF16
  #define MARLIN_ENABLE_BF16 1
#endif

#if !MARLIN_ENABLE_FP16 && !MARLIN_ENABLE_BF16
  #error "At least one Marlin compute dtype must be enabled."
#endif

#include <ATen/ATen.h>
#include <mutex>
#include <vector>

#ifndef MARLIN_SHARED_MEM_GUARD_BYTES
#  define MARLIN_SHARED_MEM_GUARD_BYTES 0 // 512
#endif

#ifndef MARLIN_MULTI_BLOCK_SHARED_MEM_GUARD_BYTES
#  define MARLIN_MULTI_BLOCK_SHARED_MEM_GUARD_BYTES 0 // 1024
#endif

template <typename T, T v>
struct is_valid_value { static constexpr bool value = true; };

#if defined(__has_include)
#  if __has_include(<ATen/native/quantized/Float8_e8m0fnu.h>)
#    define HAS_FLOAT8_E8M0FNU 1
#  else
#    define HAS_FLOAT8_E8M0FNU 0
#  endif
#else
#  define HAS_FLOAT8_E8M0FNU 0
#endif

#include "kernel.h"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace marlin {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int lda, int block_rows) {}

}  // namespace marlin

torch::Tensor MARLIN_GEMM_EXPORT_NAME(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    vllm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "marlin_gemm(..) requires CUDA_ARCH >= 7.5");
  return torch::empty({1, 1});
}

#else

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int lda, int block_rows) {
  auto start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = finish_row - start_row;

  int input_row_stride = lda * sizeof(half) / 16;
  int output_row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int input_offset = row * input_row_stride;
    int output_offset = row * output_row_stride;

    half const* a_row_half =
        reinterpret_cast<half const*>(a_int4_ptr + input_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + output_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t full_sm80_small_batch_thread_configs[] = {
    // Near-full GA100 boards benefit from lighter tiles to create more work
    // across the observed 124 SMs without relying on 2 blocks/SM.
    {64, 128, 128},
    {128, 128, 256},
    {128, 64, 128}};

thread_config_t full_sm80_large_batch_thread_configs[] = {
    {64, 128, 128},
    {64, 256, 256},
    {128, 64, 128}};

thread_config_t full_sm80_heavy_batch_thread_configs[] = {
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

bool marlin_prefers_full_sm80(int major_capability, int minor_capability,
                              int sms) {
  return major_capability == 8 && minor_capability == 0 && sms >= 124;
}

int marlin_full_sm80_exact_thread_m_blocks(int prob_m) {
  switch (prob_m) {
    case 96:
      return 2;
    case 160:
      return 2;
    default:
      return -1;
  }
}

struct marlin_device_info_t {
  int sms = 0;
  int max_shared_mem = 0;
  int major_capability = 0;
  int minor_capability = 0;
};

marlin_device_info_t query_marlin_device_info(int device) {
  marlin_device_info_t info;
  cudaDeviceGetAttribute(&info.sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&info.max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cudaDeviceGetAttribute(&info.major_capability,
                         cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&info.minor_capability,
                         cudaDevAttrComputeCapabilityMinor, device);
  return info;
}

marlin_device_info_t get_marlin_device_info(int device) {
  static std::mutex mutex;
  static std::vector<marlin_device_info_t> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (device >= static_cast<int>(cache.size())) {
    cache.resize(device + 1);
  }
  marlin_device_info_t& info = cache[device];
  if (info.sms == 0) {
    info = query_marlin_device_info(device);
  }
  return info;
}

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * stages * 2;          // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, bool has_act_order, bool is_k_full,
                          int has_zp, int is_zp_float, int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * 2;
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size =
      tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int has_zp, int is_zp_float, int stages,
                     int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Our stage-2 dense 4-bit kernels need enough B-stage capacity to cover the
  // output tile. Larger M tiles with thread_k == 64 are not emitted.
  if (stages == 2 && num_bits == 4 &&
      thread_m_blocks * 2 > th_config.thread_k / 16) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_act_order, is_k_full, has_zp, is_zp_float, stages);
  return cache_size + MARLIN_SHARED_MEM_GUARD_BYTES <= max_shared_mem;
}

  #define _GET_IF(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,   \
                  M_BLOCK_SIZE_8, STAGES, GROUP_BLOCKS, NUM_THREADS,           \
                  IS_ZP_FLOAT)                                                 \
    else if (q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS &&         \
             thread_n_blocks == THREAD_N_BLOCKS &&                             \
             thread_k_blocks == THREAD_K_BLOCKS &&                             \
             m_block_size_8 == M_BLOCK_SIZE_8 &&                               \
             stages == STAGES && group_blocks == GROUP_BLOCKS &&               \
             num_threads == NUM_THREADS &&                                     \
             is_zp_float == IS_ZP_FLOAT) {                                     \
      constexpr bool kIsStage2FourBitTile =                                    \
          STAGES == 2 &&                                                       \
          (W_TYPE == vllm::kU4 || W_TYPE == vllm::kU4B8 ||                     \
           W_TYPE == vllm::kFE2M1f);                                           \
      constexpr bool kIsSupportedStage2Tile =                                  \
          !kIsStage2FourBitTile ||                                             \
          THREAD_M_BLOCKS * 2 <= THREAD_K_BLOCKS;                              \
      if constexpr (kIsSupportedStage2Tile) {                                  \
        constexpr auto S_TYPE =                                                \
            W_TYPE == vllm::kFE2M1f                                            \
                ? (GROUP_BLOCKS == 1 ? vllm::kFE4M3fn : vllm::kFE8M0fnu)       \
                : ((W_TYPE == vllm::kFE4M3fn && GROUP_BLOCKS == 2)             \
                       ? vllm::kFE8M0fnu                                       \
                       : (std::is_same<scalar_t, half>::value                  \
                              ? vllm::kFloat16                                 \
                              : vllm::kBFloat16));                             \
        kernel = Marlin<scalar_t, W_TYPE.id(), S_TYPE.id(), NUM_THREADS,       \
                        THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,     \
                        M_BLOCK_SIZE_8, STAGES, GROUP_BLOCKS, IS_ZP_FLOAT>;    \
      }                                                                        \
    }

  // COMMON: cases for (group_blocks in [-1, 2, 4, 8] and is_zp_float == false)
  //         this is the most common cases
  // BIGGROUP: cases for big group size (group_blocks in [-1, 8])
  // FZP: cases for float-zero-point (is_zp_float = true)
  // ACT: cases for act order case (group_blocks == 0)
  // FP4: cases for nvfp4(e2m1) (group_blocks == 1)
  #define COMMON_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS, STAGES) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, -1, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, 2, NUM_THREADS,    \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, 4, NUM_THREADS,    \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, 8, NUM_THREADS,    \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, -1, NUM_THREADS,  \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, 4, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, 8, NUM_THREADS,   \
            false)

  #define COMMON_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS,       \
                             STAGES)                                         \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, -1, NUM_THREADS,  \
            false)                                                           \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, 4, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, 8, NUM_THREADS,   \
            false)                                                           \
                                                                            \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, -1, NUM_THREADS,  \
            false)                                                           \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, 4, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, 8, NUM_THREADS,   \
            false)                                                           \
                                                                            \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, -1, NUM_THREADS,  \
            false)                                                           \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, 4, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, 8, NUM_THREADS,   \
            false)

  #define COMMON_GET_IF(W_TYPE, STAGES)              \
    COMMON_GET_IF_M1(W_TYPE, 8, 8, 256, STAGES)      \
    COMMON_GET_IF_M1(W_TYPE, 8, 4, 128, STAGES)      \
    COMMON_GET_IF_M1(W_TYPE, 4, 8, 128, STAGES)      \
    COMMON_GET_IF_M234(W_TYPE, 16, 4, 256, STAGES)   \
    COMMON_GET_IF_M234(W_TYPE, 8, 4, 128, STAGES)    \
    COMMON_GET_IF_M234(W_TYPE, 4, 8, 128, STAGES)

  #define BIGGROUP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS,       \
                             STAGES)                                         \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, -1, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, 8, NUM_THREADS,    \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, -1, NUM_THREADS,  \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, 8, NUM_THREADS,   \
            false)

  #define BIGGROUP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS,     \
                               STAGES)                                       \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, -1, NUM_THREADS,  \
            false)                                                           \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, 8, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, -1, NUM_THREADS,  \
            false)                                                           \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, 8, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, -1, NUM_THREADS,  \
            false)                                                           \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, 8, NUM_THREADS,   \
            false)

  #define BIGGROUP_GET_IF(W_TYPE, STAGES)            \
    BIGGROUP_GET_IF_M1(W_TYPE, 8, 8, 256, STAGES)    \
    BIGGROUP_GET_IF_M1(W_TYPE, 8, 4, 128, STAGES)    \
    BIGGROUP_GET_IF_M1(W_TYPE, 4, 8, 128, STAGES)    \
    BIGGROUP_GET_IF_M234(W_TYPE, 16, 4, 256, STAGES) \
    BIGGROUP_GET_IF_M234(W_TYPE, 8, 4, 128, STAGES)  \
    BIGGROUP_GET_IF_M234(W_TYPE, 4, 8, 128, STAGES)

  #define NVFP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS, STAGES) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, 1, NUM_THREADS,    \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, 1, NUM_THREADS,   \
            false)

  #define NVFP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS,        \
                            STAGES)                                          \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, 1, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, 1, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, 1, NUM_THREADS,   \
            false)

  #define NVFP4_GET_IF(W_TYPE, STAGES)             \
    NVFP4_GET_IF_M1(W_TYPE, 8, 8, 256, STAGES)     \
    NVFP4_GET_IF_M1(W_TYPE, 8, 4, 128, STAGES)     \
    NVFP4_GET_IF_M1(W_TYPE, 4, 8, 128, STAGES)     \
    NVFP4_GET_IF_M234(W_TYPE, 16, 4, 256, STAGES)  \
    NVFP4_GET_IF_M234(W_TYPE, 8, 4, 128, STAGES)   \
    NVFP4_GET_IF_M234(W_TYPE, 4, 8, 128, STAGES)

  #define MXFP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS, STAGES) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, 2, NUM_THREADS,    \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)

  #define MXFP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS,        \
                            STAGES)                                          \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)

  #define MXFP4_GET_IF(W_TYPE, STAGES)             \
    MXFP4_GET_IF_M1(W_TYPE, 8, 8, 256, STAGES)     \
    MXFP4_GET_IF_M1(W_TYPE, 8, 4, 128, STAGES)     \
    MXFP4_GET_IF_M1(W_TYPE, 4, 8, 128, STAGES)     \
    MXFP4_GET_IF_M234(W_TYPE, 16, 4, 256, STAGES)  \
    MXFP4_GET_IF_M234(W_TYPE, 8, 4, 128, STAGES)   \
    MXFP4_GET_IF_M234(W_TYPE, 4, 8, 128, STAGES)

  #define MXFP8_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS, STAGES) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, 2, NUM_THREADS,    \
            false)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)

  #define MXFP8_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS,        \
                            STAGES)                                          \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)                                                           \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, 2, NUM_THREADS,   \
            false)

  #define MXFP8_GET_IF(W_TYPE, STAGES)             \
    MXFP8_GET_IF_M1(W_TYPE, 8, 8, 256, STAGES)     \
    MXFP8_GET_IF_M1(W_TYPE, 8, 4, 128, STAGES)     \
    MXFP8_GET_IF_M1(W_TYPE, 4, 8, 128, STAGES)     \
    MXFP8_GET_IF_M234(W_TYPE, 16, 4, 256, STAGES)  \
    MXFP8_GET_IF_M234(W_TYPE, 8, 4, 128, STAGES)   \
    MXFP8_GET_IF_M234(W_TYPE, 4, 8, 128, STAGES)

  // We currently have 4-bit models only with group_blocks == 4
  #define FZP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS, STAGES)   \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, 4, NUM_THREADS,   \
            true)                                                           \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, 4, NUM_THREADS,  \
            true)

  #define FZP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS, STAGES) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, 4, NUM_THREADS,  \
            true)                                                           \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, 4, NUM_THREADS,  \
            true)                                                           \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, 4, NUM_THREADS,  \
            true)

  #define FZP_GET_IF(W_TYPE, STAGES)               \
    FZP_GET_IF_M1(W_TYPE, 8, 8, 256, STAGES)       \
    FZP_GET_IF_M1(W_TYPE, 8, 4, 128, STAGES)       \
    FZP_GET_IF_M1(W_TYPE, 4, 8, 128, STAGES)       \
    FZP_GET_IF_M234(W_TYPE, 16, 4, 256, STAGES)    \
    FZP_GET_IF_M234(W_TYPE, 8, 4, 128, STAGES)     \
    FZP_GET_IF_M234(W_TYPE, 4, 8, 128, STAGES)

  // We currently have 4-bit models only with group_blocks == 4
  #define ACT_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS, STAGES)   \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, STAGES, 0, NUM_THREADS,   \
            false)                                                          \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, STAGES, 0, NUM_THREADS,  \
            false)

  #define ACT_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS, STAGES) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, STAGES, 0, NUM_THREADS,  \
            false)                                                          \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, STAGES, 0, NUM_THREADS,  \
            false)                                                          \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, STAGES, 0, NUM_THREADS,  \
            false)

  #define ACT_GET_IF(W_TYPE, STAGES)               \
    ACT_GET_IF_M1(W_TYPE, 8, 8, 256, STAGES)       \
    ACT_GET_IF_M1(W_TYPE, 8, 4, 128, STAGES)       \
    ACT_GET_IF_M1(W_TYPE, 4, 8, 128, STAGES)       \
    ACT_GET_IF_M234(W_TYPE, 16, 4, 256, STAGES)    \
    ACT_GET_IF_M234(W_TYPE, 8, 4, 128, STAGES)     \
    ACT_GET_IF_M234(W_TYPE, 4, 8, 128, STAGES)

template <typename scalar_t>
MarlinFuncPtr get_marlin_kernel(const vllm::ScalarType q_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_act_order, bool has_zp,
                                int group_blocks, int num_threads,
                                bool is_zp_float, int stages) {
  int num_bits = q_type.size_bits();
  auto kernel = MarlinDefault;
  if constexpr (std::is_same<scalar_t, half>::value) {
    if (stages == 2) {
      if (false) {
      }
      COMMON_GET_IF(vllm::kU4, 2)
      COMMON_GET_IF(vllm::kU4B8, 2)
      COMMON_GET_IF(vllm::kU8B128, 2)

      NVFP4_GET_IF(vllm::kFE2M1f, 2)

      BIGGROUP_GET_IF(vllm::kFE4M3fn, 2)

      ACT_GET_IF(vllm::kU4B8, 2)
      ACT_GET_IF(vllm::kU8B128, 2)
    }
  }

  if (stages == pipe_stages) {
    if (false) {
    }
    COMMON_GET_IF(vllm::kU4, pipe_stages)
    COMMON_GET_IF(vllm::kU4B8, pipe_stages)
    COMMON_GET_IF(vllm::kU8B128, pipe_stages)

    NVFP4_GET_IF(vllm::kFE2M1f, pipe_stages)

    BIGGROUP_GET_IF(vllm::kFE4M3fn, pipe_stages)

    ACT_GET_IF(vllm::kU4B8, pipe_stages)
    ACT_GET_IF(vllm::kU8B128, pipe_stages)
  }

  if constexpr (std::is_same<scalar_t, half>::value) {
    if (stages == 2) {
      if (false) {
      }
      FZP_GET_IF(vllm::kU4, 2)
    }
    if (stages == pipe_stages) {
      if (false) {
      }
      FZP_GET_IF(vllm::kU4, pipe_stages)
    }
  }
  if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    if (stages == pipe_stages) {
      if (false) {
      }
      MXFP8_GET_IF(vllm::kFE4M3fn, pipe_stages)
      MXFP4_GET_IF(vllm::kFE2M1f, pipe_stages)
    }
  }

  return kernel;
}

template <typename scalar_t>
exec_config_t determine_exec_config(const vllm::ScalarType& q_type, int prob_m,
                                    int prob_n, int prob_k, int thread_m_blocks,
                                    bool m_block_size_8, int num_bits,
                                    int group_size, bool has_act_order,
                                    bool is_k_full, bool has_zp,
                                    bool is_zp_float, int stages,
                                    int max_shared_mem, int major_capability,
                                    int minor_capability, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  bool full_sm80 =
      marlin_prefers_full_sm80(major_capability, minor_capability, sms);
  bool heavy_batch = full_sm80 && prob_m >= 128;
  thread_config_t const* thread_configs =
      prob_m > 16
          ? (heavy_batch ? full_sm80_heavy_batch_thread_configs
             : (full_sm80 ? full_sm80_large_batch_thread_configs
                          : large_batch_thread_configs))
          : (full_sm80 ? full_sm80_small_batch_thread_configs
                       : small_batch_thread_configs);
  int thread_configs_size =
      prob_m > 16
          ? (heavy_batch
                 ? static_cast<int>(sizeof(full_sm80_heavy_batch_thread_configs) /
                                    sizeof(thread_config_t))
             : (full_sm80
                    ? static_cast<int>(sizeof(full_sm80_large_batch_thread_configs) /
                                       sizeof(thread_config_t))
                    : static_cast<int>(sizeof(large_batch_thread_configs) /
                                       sizeof(thread_config_t))))
          : (full_sm80
                 ? static_cast<int>(sizeof(full_sm80_small_batch_thread_configs) /
                                    sizeof(thread_config_t))
                 : static_cast<int>(sizeof(small_batch_thread_configs) /
                                    sizeof(thread_config_t)));

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_act_order, is_k_full, has_zp,
                         is_zp_float, stages, max_shared_mem)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
        group_size, has_act_order, is_k_full, has_zp, is_zp_float, stages);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : group_size / 16;
    }

    auto kernel = get_marlin_kernel<scalar_t>(
        q_type, thread_m_blocks, th_config.thread_n / 16,
        th_config.thread_k / 16, m_block_size_8, has_act_order, has_zp,
        group_blocks, th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    // int m_tiles = div_ceil(prob_m, thread_m_blocks * 16);
    // int n_tiles = prob_n / th_config.thread_n;
    // int k_tiles = prob_k / th_config.thread_k;

    return {1, th_config};
  }

  return exec_cfg;
}

template <typename scalar_t>
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* s, void* s2, void* zp, void* g_idx, void* perm,
               void* a_tmp, int prob_m, int prob_n, int prob_k, int lda,
               void* workspace, vllm::ScalarType const& q_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  if (has_zp) {
    TORCH_CHECK(
        q_type == vllm::kU4 || q_type == vllm::kU8,
        "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());
  } else {
    TORCH_CHECK(
        q_type == vllm::kU4B8 || q_type == vllm::kU8B128 ||
            q_type == vllm::kFE4M3fn || q_type == vllm::kFE2M1f,
        "q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when "
        "has_zp = False. Got = ",
        q_type.str());
  }

  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    } else {
      TORCH_CHECK(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int num_bits = q_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr = (const int4*)b_bias;
  const int4* s_ptr = (const int4*)s;
  const float* s2_ptr = (const float*)s2;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;

  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int permute_blocks = min(sms, max(prob_m, 1));
    int block_rows = div_ceil(prob_m, permute_blocks);
    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    permute_cols_kernel<<<permute_blocks, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, lda, block_rows);
    // clang-format on
    A_ptr = a_tmp_ptr;
    lda = prob_k;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  marlin_device_info_t device_info = get_marlin_device_info(dev);
  int max_shared_mem = device_info.max_shared_mem;
  TORCH_CHECK(max_shared_mem > 0);

  int major_capability = device_info.major_capability;
  int minor_capability = device_info.minor_capability;
  TORCH_CHECK(major_capability > 7 ||
                  (major_capability == 7 && minor_capability >= 5),
              "marlin kernel only supports Turing or newer GPUs.");

  int stages = pipe_stages;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    if constexpr (!std::is_same<scalar_t, half>::value) {
      TORCH_CHECK(false, "Turing only supports float16 dense Marlin kernels.");
    }
  }

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  bool full_sm80 =
      marlin_prefers_full_sm80(major_capability, minor_capability, sms);
  bool disable_full_sm80_exact_split = false;
  while (rest_m) {
    int thread_k = thread_k_init;
    int thread_n = thread_n_init;
    bool manual_override = thread_k != -1 && thread_n != -1;
    int attempt_thread_m_blocks = max_thread_m_blocks;
    bool force_exact_split = false;
    // On the local 124-SM sm_80 boards, a few exact-M shapes consistently beat
    // the generic split path when we keep all M tiles in one launch. This
    // avoids tiny remainder launches such as 128+32.
    if (!manual_override && full_sm80 && !disable_full_sm80_exact_split) {
      int exact_thread_m_blocks = marlin_full_sm80_exact_thread_m_blocks(rest_m);
      if (exact_thread_m_blocks > 0 &&
          rest_m % (exact_thread_m_blocks * 16) == 0 &&
          attempt_thread_m_blocks > exact_thread_m_blocks) {
        attempt_thread_m_blocks = exact_thread_m_blocks;
        force_exact_split = true;
      }
    }

    int par_count = rest_m / (attempt_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (attempt_thread_m_blocks * 16)) : rest_m;
    if (force_exact_split) prob_m_split = rest_m;

    int thread_m_blocks =
        min(div_ceil(prob_m_split, 16), attempt_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                  " is not divisible by thread_n = ", thread_n);
      TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                  " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config<scalar_t>(
          q_type, prob_m_split, prob_n, prob_k, thread_m_blocks, m_block_size_8,
          num_bits, group_size, has_act_order, is_k_full, has_zp, is_zp_float,
          stages, max_shared_mem, major_capability, minor_capability, sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_k == -1 && force_exact_split) {
        disable_full_sm80_exact_split = true;
        max_thread_m_blocks = 4;
        continue;
      }
      if (thread_tfg.thread_k == -1 && attempt_thread_m_blocks > 1) {
        max_thread_m_blocks = attempt_thread_m_blocks - 1;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new =
          max_shared_mem / exec_cfg.blocks_per_sm -
          MARLIN_MULTI_BLOCK_SHARED_MEM_GUARD_BYTES;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    TORCH_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_act_order, is_k_full,
                        has_zp, is_zp_float, stages, max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel<scalar_t>(
        q_type, thread_m_blocks, thread_n_blocks, thread_k_blocks,
        m_block_size_8, has_act_order, has_zp, group_blocks, num_threads,
        is_zp_float, stages);

    if (kernel == MarlinDefault) {
      TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                  ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                  ", num_groups = ", num_groups, ", group_size = ", group_size,
                  ", prob_m_split = ", prob_m_split,
                  ", thread_m_blocks = ", thread_m_blocks,
                  ", thread_n_blocks = ", thread_n_blocks,
                  ", thread_k_blocks = ", thread_k_blocks,
                  ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, s_ptr, s2_ptr, zp_ptr,
        g_idx_ptr, num_groups,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
    // clang-format on

    A_ptr += prob_m_split * (lda / 8);
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}  // namespace marlin

torch::Tensor MARLIN_GEMM_EXPORT_NAME(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    vllm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  vllm::ScalarType const b_q_type = vllm::ScalarType::from_id(b_q_type_id);
  int pack_factor = 32 / b_q_type.size_bits();

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k,
              ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(
      b_q_weight.size(1) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(1) = ", b_q_weight.size(1),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.stride(1) == 1, "A.stride(1) is not 1");
  // We use int4 (16 bytes) to load A, so A must aligned to 16 bytes
  TORCH_CHECK(a.stride(0) % 8 == 0, "A.stride(0) must divisible by 8");
  TORCH_CHECK(((uint64_t)a.data_ptr()) % 16 == 0, "A must aligned to 16 bytes");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel
  int sms = marlin::get_marlin_device_info(a.get_device()).sms;

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    TORCH_CHECK(c.size(0) == size_m, "Shape mismatch: c.size(0) = ", c.size(0),
                ", size_m = ", size_m);
    TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                ", size_n = ", size_n);
  } else {
    c = torch::empty({size_m, size_n}, options);
  }
  if (size_m == 0) return c;

  // Alloc C tmp buffer that is going to be used for the global reduce
  torch::Tensor c_tmp;
  auto options_fp32 =
      torch::TensorOptions().dtype(at::kFloat).device(a.device());
  if (use_fp32_reduce) {
    int max_m_block_size = (size_m + 16 - 1) / 16 * 16;
    max_m_block_size = min(max_m_block_size, 64);
    int max_c_tmp_size =
        sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
    c_tmp = torch::empty({max_c_tmp_size}, options_fp32);
  } else {
    c_tmp = torch::empty({0}, options_fp32);
  }

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.sizes().size();
  TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  torch::Tensor g_idx, perm, a_tmp;
  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm = perm_or_none.value();

    TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
    TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");
    TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

    // Verify g_idx and perm
    TORCH_CHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                    (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                " and perm.size(-1) = ", perm.size(-1),
                ", where size_k = ", size_k);
  } else {
    g_idx = torch::empty({0}, options);
    perm = torch::empty({0}, options);
    a_tmp = torch::empty({0}, options);
  }
  bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

  if (has_act_order) {
    a_tmp = torch::empty({size_m, size_k}, options);
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    a_tmp = torch::empty({0}, options);
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  torch::Tensor global_scale;
  if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    TORCH_CHECK(b_q_type == vllm::kFE2M1f && group_size == 16,
                "global_scale can only be used for nvfp4 format.");
  } else {
    global_scale = torch::empty({0}, options_fp32);
    TORCH_CHECK(!(b_q_type == vllm::kFE2M1f && group_size == 16),
                "the global_scale parameter must be passed for nvfp4 format.");
  }

  bool has_bias = b_bias_or_none.has_value();
  torch::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    TORCH_CHECK(b_bias.device().is_cuda(), "b_bias is not on GPU");
    TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");
    TORCH_CHECK(b_bias.size(0) == size_n, "b_bias.size(0) != size_n");
    TORCH_CHECK(b_bias.stride(0) == 1, "b_bias.stride(0) != 1");
  } else {
    b_bias = torch::empty({0}, options);
  }

  torch::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = torch::empty({0}, options);
  }
  bool has_zp = b_zeros.size(-1) > 0;
  if (has_zp) {
    TORCH_CHECK(
        b_q_type == vllm::kU4 || b_q_type == vllm::kU8,
        "b_q_type must be u4 or u8 when has_zp = True. Got = ", b_q_type.str());
  } else {
    TORCH_CHECK(b_q_type == vllm::kU4B8 || b_q_type == vllm::kU8B128 ||
                    b_q_type == vllm::kFE4M3fn || b_q_type == vllm::kFE2M1f,
                "b_q_type must be uint4b8, uint8b128, float8_e4m3fn or "
                "float4_e2m1f when "
                "has_zp = False. Got = ",
                b_q_type.str());
  }

  if (has_zp && is_zp_float) {
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half,
                "Computation type must be float16 (half) when using float zero "
                "points.");
  }

  // Verify b_zeros
  if (has_zp) {
    int rank = b_zeros.sizes().size();
    TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");
    if (is_zp_float) {
      TORCH_CHECK(b_zeros.size(1) == size_n,
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not size_n = ", size_n);
      TORCH_CHECK(num_groups == b_zeros.size(0),
                  "b_zeros dim 0 = ", b_zeros.size(0),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(num_groups != -1, "num_groups must be != -1");
    } else {
      TORCH_CHECK(b_zeros.size(0) == num_groups,
                  "b_zeros dim 0 = ", b_zeros.size(0),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(b_zeros.size(1) == size_n / pack_factor,
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not size_n / pack_factor = ", size_n / pack_factor);
    }
  }

  // Verify workspace size
  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
              "size_n = ", size_n, ", is not divisible by min_thread_n = ",
              MARLIN_NAMESPACE_NAME::min_thread_n);

  int min_workspace_size = sms;
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();
  TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
              "scalar type of global_scale must be float");
  #if MARLIN_ENABLE_FP16
  if (a.scalar_type() == at::ScalarType::Half) {
    void* scales_ptr;
    if (b_q_type == vllm::kFE2M1f) {
      if (group_size == 16)
        scales_ptr = b_scales.data_ptr<at::Float8_e4m3fn>();
      else if (group_size == 32) {
        #if HAS_FLOAT8_E8M0FNU
          scales_ptr = b_scales.data_ptr<at::Float8_e8m0fnu>();
        #else
          TORCH_CHECK(false,
            "float8_e8m0fnu is not available in this build of PyTorch.");
        #endif
      } else
        TORCH_CHECK(false,
                    "float4_e2m1f only supports group_size == 16 (NVFP4) ",
                    "and group_size == 32 (MXFP4)");
#if HAS_FLOAT8_E8M0FNU
    } else if (b_q_type == vllm::kFE4M3fn &&
               b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
      TORCH_CHECK(false, "float8_e4m3fn with float8_e8m0fnu scales requires "
                         "bfloat16 compute (MXFP8).");
#endif
    } else {
      scales_ptr = b_scales.data_ptr<at::Half>();
    }

    marlin::marlin_mm<half>(
        a.data_ptr<at::Half>(), b_q_weight.data_ptr(), c.data_ptr<at::Half>(),
        c_tmp.data_ptr<float>(), b_bias.data_ptr<at::Half>(), scales_ptr,
        global_scale.data_ptr<float>(), b_zeros.data_ptr(), g_idx.data_ptr(),
        perm.data_ptr(), a_tmp.data_ptr<at::Half>(), size_m, size_n, size_k,
        a.stride(0), workspace.data_ptr(), b_q_type, has_bias, has_act_order,
        is_k_full, has_zp, num_groups, group_size, dev,
        at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms,
        use_atomic_add, use_fp32_reduce, is_zp_float);
    return c;
  }
  #endif

  #if MARLIN_ENABLE_BF16
  if (a.scalar_type() == at::ScalarType::BFloat16) {
    void* scales_ptr;
    if (b_q_type == vllm::kFE2M1f) {
      if (group_size == 16)
        scales_ptr = b_scales.data_ptr<at::Float8_e4m3fn>();
      else if (group_size == 32) {
        #if HAS_FLOAT8_E8M0FNU
          scales_ptr = b_scales.data_ptr<at::Float8_e8m0fnu>();
        #else
          TORCH_CHECK(false,
            "float8_e8m0fnu is not available in this build of PyTorch.");
        #endif
     } else
        TORCH_CHECK(false,
                    "float4_e2m1f only supports group_size == 16 (NVFP4) ",
                    "and group_size == 32 (MXFP4)");
#if HAS_FLOAT8_E8M0FNU
    } else if (b_q_type == vllm::kFE4M3fn &&
               b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
      TORCH_CHECK(group_size == 32,
                  "float8_e4m3fn only supports group_size == 32 (MXFP8) when "
                  "using float8_e8m0fnu scales.");
      scales_ptr = b_scales.data_ptr<at::Float8_e8m0fnu>();
#endif
    } else {
      scales_ptr = b_scales.data_ptr<at::BFloat16>();
    }

    marlin::marlin_mm<nv_bfloat16>(
        a.data_ptr<at::BFloat16>(), b_q_weight.data_ptr(),
        c.data_ptr<at::BFloat16>(), c_tmp.data_ptr<float>(),
        b_bias.data_ptr<at::BFloat16>(), scales_ptr,
        global_scale.data_ptr<float>(), b_zeros.data_ptr(),
        g_idx.data_ptr(), perm.data_ptr(), a_tmp.data_ptr<at::BFloat16>(),
        size_m, size_n, size_k, a.stride(0), workspace.data_ptr(), b_q_type,
        has_bias, has_act_order, is_k_full, has_zp, num_groups, group_size, dev,
        at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms,
        use_atomic_add, use_fp32_reduce, is_zp_float);
    return c;
  }
  #endif

  #if MARLIN_ENABLE_FP16 && MARLIN_ENABLE_BF16
  TORCH_CHECK(false, "gpt_marlin_gemm only supports bfloat16 and float16");
  #elif MARLIN_ENABLE_FP16
  TORCH_CHECK(false, "gpt_marlin_gemm_fp16 only supports float16");
  #else
  TORCH_CHECK(false, "gpt_marlin_gemm_bf16 only supports bfloat16");
  #endif

  return c;
}

#endif
