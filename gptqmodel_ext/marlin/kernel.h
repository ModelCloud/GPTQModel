
#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#define MARLIN_KERNEL_PARAMS                                                   \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                      \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                          \
      const int4 *__restrict__ b_bias_ptr,                                     \
      const int4 *__restrict__ scales_ptr,                                     \
      const float *__restrict__ global_scale_ptr,                              \
      const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx,          \
      int num_groups, int prob_m, int prob_n, int prob_k, int lda, int *locks, \
      bool has_bias, bool use_atomic_add, bool use_fp32_reduce,                \
      int max_shared_mem

#define MARLIN_EORA_KERNEL_PARAMS_FOR(eora_scalar_t)                           \
  MARLIN_KERNEL_PARAMS, const eora_scalar_t *__restrict__ eora_x,             \
      const eora_scalar_t *__restrict__ eora_down_weight,                     \
      const eora_scalar_t *__restrict__ eora_up_weight,                       \
      eora_scalar_t *__restrict__ eora_out, float *__restrict__ eora_workspace

#define MARLIN_EORA_KERNEL_PARAMS MARLIN_EORA_KERNEL_PARAMS_FOR(scalar_t)

namespace MARLIN_NAMESPACE_NAME {
template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const vllm::ScalarTypeId s_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

// Single-row rank-32 W4A16 attention specialization. The Marlin and LoRA
// phases share one cooperative launch and reuse Marlin's dead reduction
// scratch after the base output is complete.
template <typename scalar_t,
          const vllm::ScalarTypeId w_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,
          const int stages,
          const int group_blocks,
          const bool is_zp_float>
__global__ void MarlinEoraRank32(MARLIN_EORA_KERNEL_PARAMS);

// Single-row rank-64 W4A16 attention specialization. The Marlin and LoRA
// phases share one cooperative launch and reuse Marlin's dead reduction
// scratch after the base output is complete.
template <typename scalar_t,
          const vllm::ScalarTypeId w_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,
          const int stages,
          const int group_blocks,
          const bool is_zp_float>
__global__ void MarlinEoraRank64(MARLIN_EORA_KERNEL_PARAMS);

// Single-row rank-96 W4A16 attention specialization. The Marlin and LoRA
// phases share one cooperative launch and reuse Marlin's dead reduction
// scratch after the base output is complete.
template <typename scalar_t,
          const vllm::ScalarTypeId w_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,
          const int stages,
          const int group_blocks,
          const bool is_zp_float>
__global__ void MarlinEoraRank96(MARLIN_EORA_KERNEL_PARAMS);

// Single-row rank-128 W4A16 attention specialization. The Marlin and LoRA
// phases share one cooperative launch and reuse Marlin's dead reduction
// scratch after the base output is complete.
template <typename scalar_t,
          const vllm::ScalarTypeId w_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,
          const int stages,
          const int group_blocks,
          const bool is_zp_float>
__global__ void MarlinEoraRank128(MARLIN_EORA_KERNEL_PARAMS);

// Single-row rank-192 W4A16 attention specialization. It preserves the same
// one-wave cooperative Marlin schedule while using six LoRA-up warps.
template <typename scalar_t,
          const vllm::ScalarTypeId w_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,
          const int stages,
          const int group_blocks,
          const bool is_zp_float>
__global__ void MarlinEoraRank192(MARLIN_EORA_KERNEL_PARAMS);

// Single-row rank-256 W4A16 attention specialization. It preserves the same
// one-wave cooperative Marlin schedule while doubling adapter rank work.
template <typename scalar_t,
          const vllm::ScalarTypeId w_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,
          const int stages,
          const int group_blocks,
          const bool is_zp_float>
__global__ void MarlinEoraRank256(MARLIN_EORA_KERNEL_PARAMS);

// Large-M W4A16 specialization. Each CTA owns one output tile and traverses
// the full K dimension, avoiding the cross-CTA reduction used by decode Marlin.
template <typename scalar_t,
          const vllm::ScalarTypeId w_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,
          const int stages,
          const int group_blocks,
          const bool is_zp_float>
__global__ void MarlinPrefill(MARLIN_KERNEL_PARAMS);

}
