/*
 * SPDX-FileCopyrightText: 2026 ModelCloud.ai
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compact INT4 GPTQ-GEMM CUDA path for symmetric GPTQ weights. This is a
 * direct, conservative Ampere Tensor Core scaffold intended for explicit
 * benchmarking against the stable GPTQ kernels before auto-selection.
 */

#include "gptq_gemm_kernel.cuh"

struct __align__(16) GptqGemmSmem {
    half     A[GPTQ_GEMM_M_PER_WARP * GPTQ_GEMM_K_PER_WARP];
    half     S[GPTQ_GEMM_N_PER_WARP];
    uint32_t Bfrag[GPTQ_GEMM_BFRAG_WORDS_PER_BUF];
};

__device__ __forceinline__
half zero_half() {
    return __float2half(0.0f);
}

__device__ __forceinline__
uint8_t load_b_pair_byte(const uint8_t* __restrict__ B_packed,
                         int K, int N,
                         int k_even, int n_col) {
    if (n_col >= N || k_even >= K) {
        return 0x88u;
    }

    const int packed_row = k_even >> 1;
    uint8_t byte = B_packed[packed_row * N + n_col];
    if (k_even + 1 >= K) {
        byte = static_cast<uint8_t>((byte & 0x0Fu) | 0x80u);
    }
    return byte;
}

__device__ __forceinline__
uint16_t pack_lane_bfrag(const uint8_t* __restrict__ B_packed,
                         int N, int K,
                         int k_base, int n_base,
                         int j, int lane) {
    const int group_id = lane >> 2;
    const int tid4 = lane & 3;
    const int n_col = n_base + j * 8 + group_id;
    if (n_col >= N) {
        return 0x8888u;
    }

    const int k01 = k_base + 2 * tid4;
    const int k89 = k01 + 8;
    const uint8_t byte01 = load_b_pair_byte(B_packed, K, N, k01, n_col);
    const uint8_t byte89 = load_b_pair_byte(B_packed, K, N, k89, n_col);
    return static_cast<uint16_t>(byte01)
         | (static_cast<uint16_t>(byte89) << 8);
}

__device__ __forceinline__
void stage_a_tile(GptqGemmSmem* __restrict__ smem,
                  const half* __restrict__ A,
                  int M, int K,
                  int m_base, int k_base) {
    for (int idx = threadIdx.x; idx < GPTQ_GEMM_M_PER_WARP * GPTQ_GEMM_K_PER_WARP; idx += blockDim.x) {
        const int row = idx / GPTQ_GEMM_K_PER_WARP;
        const int col = idx % GPTQ_GEMM_K_PER_WARP;
        const int global_m = m_base + row;
        const int global_k = k_base + col;
        smem->A[idx] = (global_m < M && global_k < K)
                     ? A[global_m * K + global_k]
                     : zero_half();
    }
}

__device__ __forceinline__
void stage_scale_row(GptqGemmSmem* __restrict__ smem,
                     const half* __restrict__ S,
                     int N,
                     int k_base,
                     int group_size,
                     int n_base) {
    const int group_idx = k_base / group_size;
    for (int idx = threadIdx.x; idx < GPTQ_GEMM_N_PER_WARP; idx += blockDim.x) {
        const int global_n = n_base + idx;
        smem->S[idx] = (global_n < N)
                     ? S[group_idx * N + global_n]
                     : zero_half();
    }
}

__device__ __forceinline__
void stage_bfrag_tiles(GptqGemmSmem* __restrict__ smem,
                       const uint8_t* __restrict__ B_packed,
                       int N, int K,
                       int k_base, int n_base) {
    for (int idx = threadIdx.x; idx < GPTQ_GEMM_BFRAG_WORDS_PER_BUF; idx += blockDim.x) {
        const int j = idx / GPTQ_GEMM_BFRAG_WORDS_PER_TILE;
        const int lane_pair = idx % GPTQ_GEMM_BFRAG_WORDS_PER_TILE;
        const int even_lane = lane_pair * 2;
        const uint16_t even_p16 = pack_lane_bfrag(B_packed, N, K, k_base, n_base, j, even_lane);
        const uint16_t odd_p16 = pack_lane_bfrag(B_packed, N, K, k_base, n_base, j, even_lane + 1);
        smem->Bfrag[idx] = static_cast<uint32_t>(even_p16)
                         | (static_cast<uint32_t>(odd_p16) << 16);
    }
}

__device__ __forceinline__
void do_mma_inner_loop(const GptqGemmSmem* __restrict__ smem,
                       float RC[GPTQ_GEMM_J_TILES][4]) {
    const int lane = threadIdx.x & (GPTQ_GEMM_WARP_SIZE - 1);
    const int group_id = lane >> 2;
    const half zero_point = __float2half(8.0f);

    uint32_t RA[4];
    load_a_fragment_rowmajor(smem->A, lane, RA);

    #pragma unroll
    for (int j = 0; j < GPTQ_GEMM_J_TILES; ++j) {
        const half scale = smem->S[j * 8 + group_id];
        const uint16_t packed_16 = fetch_bfrag_packed16(smem->Bfrag, 0, 0, j, lane);

        uint32_t RB[2];
        decode_bfrag_to_rb(packed_16, scale, zero_point, RB);
        mma_f32_m16n8k16(RA, RB, RC[j]);
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__global__ void gptq_gemm_kernel(
    const half*    __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const half*    __restrict__ S,
    half*          __restrict__ C,
    int M, int N, int K, int group_size)
{
    extern __shared__ uint8_t raw_smem[];
    GptqGemmSmem* smem = reinterpret_cast<GptqGemmSmem*>(raw_smem);

    const int warp_m = blockIdx.x;
    const int warp_n = blockIdx.y;

    const int m_base = warp_m * GPTQ_GEMM_M_PER_WARP;
    const int n_base = warp_n * GPTQ_GEMM_N_PER_WARP;
    const int lane = threadIdx.x & (GPTQ_GEMM_WARP_SIZE - 1);

    float RC[GPTQ_GEMM_J_TILES][4];
    #pragma unroll
    for (int j = 0; j < GPTQ_GEMM_J_TILES; ++j) {
        RC[j][0] = 0.0f;
        RC[j][1] = 0.0f;
        RC[j][2] = 0.0f;
        RC[j][3] = 0.0f;
    }

    const int num_k_tiles = (K + GPTQ_GEMM_K_PER_WARP - 1) / GPTQ_GEMM_K_PER_WARP;
    for (int t = 0; t < num_k_tiles; ++t) {
        const int k_base = t * GPTQ_GEMM_K_PER_WARP;

        stage_a_tile(smem, A, M, K, m_base, k_base);
        stage_scale_row(smem, S, N, k_base, group_size, n_base);
        stage_bfrag_tiles(smem, B_packed, N, K, k_base, n_base);
        __syncthreads();

        do_mma_inner_loop(smem, RC);
        __syncthreads();
    }

    const int row_base = lane >> 2;
    const int col_pair = 2 * (lane & 3);
    #pragma unroll
    for (int j = 0; j < GPTQ_GEMM_J_TILES; ++j) {
        const int m0 = m_base + row_base;
        const int m1 = m_base + row_base + 8;
        const int n0 = n_base + j * 8 + col_pair + 0;
        const int n1 = n_base + j * 8 + col_pair + 1;

        if (m0 < M) {
            if (n0 < N) C[m0 * N + n0] = __float2half_rn(RC[j][0]);
            if (n1 < N) C[m0 * N + n1] = __float2half_rn(RC[j][1]);
        }
        if (m1 < M) {
            if (n0 < N) C[m1 * N + n0] = __float2half_rn(RC[j][2]);
            if (n1 < N) C[m1 * N + n1] = __float2half_rn(RC[j][3]);
        }
    }
}
#else
__global__ void gptq_gemm_kernel(
    const half*, const uint8_t*, const half*, half*, int, int, int, int) {}
#endif

cudaError_t gptq_gemm(
    const half*    A,
    const uint8_t* B_packed,
    const half*    S,
    half*          C,
    int M, int N, int K, int group_size,
    cudaStream_t stream)
{
    if (group_size <= 0) {
        group_size = K;
    }
    if ((group_size % GPTQ_GEMM_K_PER_WARP) != 0) {
        return cudaErrorInvalidValue;
    }

    dim3 grid(
        (M + GPTQ_GEMM_M_PER_WARP - 1) / GPTQ_GEMM_M_PER_WARP,
        (N + GPTQ_GEMM_N_PER_WARP - 1) / GPTQ_GEMM_N_PER_WARP,
        1);
    dim3 block(GPTQ_GEMM_WARP_SIZE, 1, 1);
    const size_t smem_bytes = sizeof(GptqGemmSmem);

    gptq_gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        A, B_packed, S, C, M, N, K, group_size);
    return cudaGetLastError();
}
