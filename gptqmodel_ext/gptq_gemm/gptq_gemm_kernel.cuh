/*
 * SPDX-FileCopyrightText: 2026 ModelCloud.ai
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from the public groxaxo/GPTQ-Pro Ampere scaffold and integrated as
 * GPTQ-GEMM for GPTQModel's managed torch.ops JIT path.
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

static constexpr int GPTQ_GEMM_KS_TILES  = 1;
static constexpr int GPTQ_GEMM_J_TILES   = 8;
static constexpr int GPTQ_GEMM_WARP_SIZE = 32;

static constexpr int GPTQ_GEMM_M_PER_WARP = 16;
static constexpr int GPTQ_GEMM_N_PER_WARP = GPTQ_GEMM_J_TILES * 8;
static constexpr int GPTQ_GEMM_K_PER_WARP = GPTQ_GEMM_KS_TILES * 16;

static constexpr int GPTQ_GEMM_BFRAG_WORDS_PER_TILE = GPTQ_GEMM_WARP_SIZE / 2;
static constexpr int GPTQ_GEMM_BFRAG_WORDS_PER_BUF =
    GPTQ_GEMM_KS_TILES * GPTQ_GEMM_J_TILES * GPTQ_GEMM_BFRAG_WORDS_PER_TILE;

union Half2Reg {
    half2    h2;
    uint32_t u32;
    uint16_t u16[2];
};

__device__ __forceinline__
uint32_t pack_half2_reg(half lo, half hi) {
    Half2Reg reg;
    reg.h2 = __halves2half2(lo, hi);
    return reg.u32;
}

__device__ __forceinline__
uint32_t bfrag_smem_addr(const uint32_t* __restrict__ smem_bfrag_base,
                         int buf, int ks, int j, int lane) {
    const int tile_idx  = ks * GPTQ_GEMM_J_TILES + j;
    const int buf_words = GPTQ_GEMM_BFRAG_WORDS_PER_BUF;
    const int word_idx  = buf * buf_words
                        + tile_idx * GPTQ_GEMM_BFRAG_WORDS_PER_TILE
                        + (lane >> 1);
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_bfrag_base))
         + static_cast<uint32_t>(word_idx * sizeof(uint32_t));
}

__device__ __forceinline__
uint32_t ld_shared_u32(uint32_t smem_addr) {
    uint32_t val;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(smem_addr));
    return val;
}

__device__ __forceinline__
uint16_t fetch_bfrag_packed16(const uint32_t* __restrict__ smem_bfrag,
                              int buf, int ks, int j, int lane) {
    const uint32_t addr = bfrag_smem_addr(smem_bfrag, buf, ks, j, lane);
    const uint32_t word = ld_shared_u32(addr);
    return static_cast<uint16_t>((lane & 1) ? (word >> 16) : (word & 0xFFFFu));
}

__device__ __forceinline__
void decode_bfrag_to_rb(uint16_t packed_16,
                        half scale, half zero_point,
                        uint32_t (&RB)[2]) {
    const uint32_t p  = static_cast<uint32_t>(packed_16);
    const uint32_t w0 = (p >>  0) & 0xFu;
    const uint32_t w1 = (p >>  4) & 0xFu;
    const uint32_t w2 = (p >>  8) & 0xFu;
    const uint32_t w3 = (p >> 12) & 0xFu;

    const half2 vals01 = __halves2half2(__int2half_rn(static_cast<int>(w0)),
                                        __int2half_rn(static_cast<int>(w1)));
    const half2 vals23 = __halves2half2(__int2half_rn(static_cast<int>(w2)),
                                        __int2half_rn(static_cast<int>(w3)));
    const half2 zp_h2 = __halves2half2(zero_point, zero_point);
    const half2 sc_h2 = __halves2half2(scale, scale);

    Half2Reg rb0, rb1;
    rb0.h2 = __hmul2(sc_h2, __hsub2(vals01, zp_h2));
    rb1.h2 = __hmul2(sc_h2, __hsub2(vals23, zp_h2));
    RB[0] = rb0.u32;
    RB[1] = rb1.u32;
}

__device__ __forceinline__
void load_a_fragment_rowmajor(const half* __restrict__ smem_a,
                              int lane,
                              uint32_t (&RA)[4]) {
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;
    const int a_col_lo = 2 * thread_id;
    const int a_col_hi = a_col_lo + 8;

    RA[0] = pack_half2_reg(
        smem_a[(group_id + 0) * GPTQ_GEMM_K_PER_WARP + a_col_lo + 0],
        smem_a[(group_id + 0) * GPTQ_GEMM_K_PER_WARP + a_col_lo + 1]);
    RA[1] = pack_half2_reg(
        smem_a[(group_id + 8) * GPTQ_GEMM_K_PER_WARP + a_col_lo + 0],
        smem_a[(group_id + 8) * GPTQ_GEMM_K_PER_WARP + a_col_lo + 1]);
    RA[2] = pack_half2_reg(
        smem_a[(group_id + 0) * GPTQ_GEMM_K_PER_WARP + a_col_hi + 0],
        smem_a[(group_id + 0) * GPTQ_GEMM_K_PER_WARP + a_col_hi + 1]);
    RA[3] = pack_half2_reg(
        smem_a[(group_id + 8) * GPTQ_GEMM_K_PER_WARP + a_col_hi + 0],
        smem_a[(group_id + 8) * GPTQ_GEMM_K_PER_WARP + a_col_hi + 1]);
}

__device__ __forceinline__
void mma_f32_m16n8k16(const uint32_t RA[4],
                      const uint32_t RB[2],
                      float          RC[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(RC[0]), "+f"(RC[1]), "+f"(RC[2]), "+f"(RC[3])
        :  "r"(RA[0]),  "r"(RA[1]),  "r"(RA[2]),  "r"(RA[3]),
           "r"(RB[0]),  "r"(RB[1]));
}

cudaError_t gptq_gemm(
    const half*    A,
    const uint8_t* B_packed,
    const half*    S,
    half*          C,
    int M, int N, int K, int group_size,
    cudaStream_t stream);
