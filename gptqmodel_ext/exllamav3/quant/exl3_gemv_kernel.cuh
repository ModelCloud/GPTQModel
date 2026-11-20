#pragma once
// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.

// Largest batch the GEMV path accepts, on both backends.
#define EXL3_GEMV_MAX_M 8

#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)

// Small-m GEMV path for the EXL3 GEMM, QTIP-style structure (see Cornell-RelaxML/qtip,
// qtip-kernels/src/inference.cu) on the unmodified EXL3 format:
//
// - warps split k and never synchronize during the main loop: no block-wide pipeline barriers;
//   B streams straight to registers with ld.global.cs (evict-first; B is single-use) behind a
//   register prefetch ring
// - the two-word bit windows of the trellis stream are resolved in-warp: with SMEM_STAGE = false
//   via lane shuffles (the extraction helpers in exl3_dq.cuh read exactly two words per lane, at
//   lane-computable indices), with SMEM_STAGE = true by staging the tile words through
//   warp-private shared memory and calling the standard dq_dispatch
// - one m16n8k16 MMA pair per 16x16 weight tile with fp16 accumulation, folded to fp32 on a fixed
//   cadence; per-block cross-warp reduction over the k splits through shared memory
//
// Same launch signature as exl3_gemm_kernel so kernel args and graph parameter patching are
// interchangeable. Cooperative launch: one grid.sync after the input Hadamard stage and one before
// the output stage, no other cross-block coordination. 2, 3 and 4 bpw.
//
// CFG 0 ("narrow", 512 threads, 2 n-tiles/warp, 16 k-splits) wins at attention-projection sizes;
// CFG 1 ("wide", 256 threads, 4 n-tiles/warp, 8 k-splits) wins at large-n FFN sizes. MMODE 0 is
// the m == 1 fast path, MMODE 1 covers 2 <= m <= 8 with row-guarded fragment loads.

#include <cooperative_groups.h>
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"

namespace exl3_gemv_ns {

// mma.m16n8k16 with the A operand supplied as two FragB halves, fp16 accumulate.
// On Turing this splits into the two k=8 halves it is already expressed as (a01 pairs with
// b[0], a23 with b[1]), accumulating through the same c. Same operands and products; one
// extra rounding at the k=8 boundary. See the longer note in ptx.cuh.
__device__ __forceinline__ void mma_ab_h(const FragB& a01, const FragB& a23, const FragB& b, FragC_h& c)
{
#if defined(USE_ROCM)
    FragA a;
    a.elems[0] = a01.elems[0];
    a.elems[1] = a01.elems[1];
    a.elems[2] = a23.elems[0];
    a.elems[3] = a23.elems[1];
    ptx_mma_m16n8k16(a, b, c);
#else
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
#if EXL3_SM75
    asm
    (
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(a0[0]), "r"(a0[1]),
           "r"(bb[0])
    );
    asm
    (
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(a1[0]), "r"(a1[1]),
           "r"(bb[1])
    );
#else
    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
           "r"(bb[0]), "r"(bb[1])
    );
#endif
#endif
}

// mul1 codebook pair decode via dp4a byte sum (bit-identical to the vabsdiff4 form)
__device__ __forceinline__ half2 decode_pair_cb2_dp4a_(uint32_t x0, uint32_t x1)
{
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    half_uint16 h0((uint16_t) sum0);
    half_uint16 h1((uint16_t) sum1);
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

template <int cb>
__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    if constexpr (cb == 2)
    {
        f0[0] = decode_pair_cb2_dp4a_(w0, w1);
        f0[1] = decode_pair_cb2_dp4a_(w2, w3);
        f1[0] = decode_pair_cb2_dp4a_(w4, w5);
        f1[1] = decode_pair_cb2_dp4a_(w6, w7);
    }
    else
    {
        f0[0] = decode_3inst_2<cb>(w0, w1);
        f0[1] = decode_3inst_2<cb>(w2, w3);
        f1[0] = decode_3inst_2<cb>(w4, w5);
        f1[1] = decode_3inst_2<cb>(w6, w7);
    }
}

// Window extraction from two already-loaded words, same order as dq8_aligned_4bits. w[i] is
// the trellis window of weight i of the lane's eight (see the column/k mapping below), which
// the INT8 kernel needs without the codebook decode.
__device__ __forceinline__ void ext8_regs_4bits(uint32_t a, uint32_t b, uint32_t (&w)[8])
{
    uint32_t s;
    FSHF_IMM(s, b, a, 20);
    w[7] = b & 0xffff;
    BFE16_IMM(w[6], b, 4);
    BFE16_IMM(w[5], b, 8);
    BFE16_IMM(w[4], b, 12);
    BFE16_IMM(w[3], b, 16);
    w[2] = s & 0xffff;
    BFE16_IMM(w[1], s, 4);
    BFE16_IMM(w[0], s, 8);
}

template <int cb>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB& f0, FragB& f1)
{
    uint32_t w[8];
    ext8_regs_4bits(a, b, w);
    decode8<cb>(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], f0, f1);
}

// Register form of dq8_aligned_2bits: the two words and the funnel shift are lane-dependent
__device__ __forceinline__ void ext8_regs_2bits(uint32_t a, uint32_t b, int t_offset, uint32_t (&w)[8])
{
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w[7] = b & 0xffff;
    BFE16_IMM(w[6], b, 2);
    BFE16_IMM(w[5], b, 4);
    BFE16_IMM(w[4], b, 6);
    BFE16_IMM(w[3], b, 8);
    BFE16_IMM(w[2], b, 10);
    BFE16_IMM(w[1], b, 12);
    BFE16_IMM(w[0], b, 14);
}

template <int cb>
__device__ __forceinline__ void dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset, FragB& f0, FragB& f1)
{
    uint32_t w[8];
    ext8_regs_2bits(a, b, t_offset, w);
    decode8<cb>(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], f0, f1);
}

// Register form of dq8<3, cb, 4> with the per-lane funnel alignment precomputed
__device__ __forceinline__ void ext8_regs_3bits(uint32_t a, uint32_t b, int s2, uint32_t (&w)[8])
{
    w[7] = fshift(b, a, s2);
    w[6] = w[7] >> 3;
    w[5] = w[6] >> 3;
    w[4] = w[5] >> 3;
    w[3] = fshift(b, a, s2 + 12);
    w[2] = w[3] >> 3;
    w[1] = w[2] >> 3;
    w[0] = w[1] >> 3;
    w[7] &= 0xffff; w[6] &= 0xffff; w[5] &= 0xffff; w[4] &= 0xffff;
    w[3] &= 0xffff; w[2] &= 0xffff; w[1] &= 0xffff; w[0] &= 0xffff;
}

template <int cb>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1)
{
    uint32_t w[8];
    ext8_regs_3bits(a, b, s2, w);
    decode8<cb>(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], f0, f1);
}

// Register form of dq8_half<KA, cb> (exl3_dq.cuh): the two window groups' word pairs and funnel alignments are
// lane constants, so the caller resolves the four words by shuffle or from the staged tile
template <int KA, int cb>
__device__ __forceinline__ void dq8_regs_half(uint32_t a7, uint32_t b7, int s7, uint32_t a3, uint32_t b3, int s3,
                                              FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b7, a7, s7);
    w6 = w7 >> (KA + 1);
    w5 = w6 >> KA;
    w4 = w5 >> (KA + 1);
    w3 = fshift(b3, a3, s3);
    w2 = w3 >> (KA + 1);
    w1 = w2 >> KA;
    w0 = w1 >> (KA + 1);
    decode8<cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

}  // namespace exl3_gemv_ns

// HALF: bitrate is bits + 0.5 (bits = 1, 2, 3; mul1 codebook): 12 / 20 / 28 uint32 per tile. 1.5 bpw packs two tiles
// per 24-word warp load like 2 bpw does per 32-word load; 2.5 and 3.5 bpw load one tile per warp load with the
// trailing lanes idle like 3 bpw
// The half-integer instances carry six lane constants for window extraction; without a minimum-blocks bound
// ptxas spends 81-85 registers on them on sm_86/sm_89 (one 512-thread block per SM instead of two, measured
// 18-28% slower at attention-projection shapes on the 3090). They are packed into one register (see x_pack) and
// the bound keeps the compiler at 64. The integer narrow instances share this body and, unbounded, get 71-80
// registers too (CUDA 12.8 and 13.1, sm_89), so they exist in both forms: BOUND (two blocks per SM) for grids larger
// than one unbounded wave, unbounded for grids that fit in one, where the extra registers are a little faster.
// The integer wide instances take a two-block minimum in their only form: unbounded, 3 bpw with the mcg codebook
// goes past the register count that lets two 256-thread blocks share an SM, and a wide grid is always larger
// than one block per SM. The other wide instances already fit two blocks and compile the same either way
template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE, bool HALF = false, bool BOUND = HALF>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256, CFG == 0 ? (BOUND ? 2 : 1) : (HALF ? 1 : 2))
void exl3_gemv_kernel(EXL3_GEMM_ARGS)
{
    constexpr bool Q4NX = cb == EXL3_Q4NX_CODEBOOK;
    constexpr int STEP_BITS = 4 * bits + 2 * HALF;
    if constexpr (Q4NX)
        static_assert(q4nx_supports_step_bits(STEP_BITS) && c_fp32 && !SMEM_STAGE);
    else
        static_assert(HALF ? (bits >= 1 && bits <= 3 && cb == 2) : (bits == 2 || bits == 3 || bits == 4));
    constexpr int WK   = CFG == 0 ? 16 : 8;     // k-split (warps per block)
    constexpr int WNT  = CFG == 0 ? 2 : 4;      // adjacent n-tiles per warp
    constexpr int PF   = CFG == 0 ? 4 : 2;      // prefetch ring depth
    constexpr int FOLD = CFG == 0 ? 4 : 2;      // fp16->fp32 fold cadence (divides PF)
    constexpr int THREADS = WK * 32;
    constexpr int ROWS = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    constexpr int COLS = WNT * 16;

    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;              // uint32 per 16x16 tile
    constexpr bool TWO_PER_LOAD = !Q4NX && (HALF ? bits == 1 : bits == 2);   // two tiles per warp load
    constexpr int LOADS = TWO_PER_LOAD ? WNT / 2 : WNT;                       // warp loads per k-slice
    constexpr int LSTRIDE = TWO_PER_LOAD ? 2 * TWORDS : (TWORDS < 32 ? TWORDS : 32);   // uint32 per load (lanes < LSTRIDE load)
    static_assert(!TWO_PER_LOAD || WNT % 2 == 0, "two tiles per warp load needs an even tile count per warp");

    auto grid = cooperative_groups::this_grid();

    // Input scales and Hadamard transform, same as exl3_gemm_kernel
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;   // uint32 per k-slice row
    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // A fragment row indices for this lane
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

    // Per-lane extraction constants (see dq8_aligned_2bits / dq8<3, cb, 4> / dq8_half in exl3_dq.cuh)
    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    // Half-integer rates: word indices (< TWORDS <= 28) and funnel shifts (< 32) of both window groups, six 5-bit
    // fields in one register, extracted per use (bfe)
    [[maybe_unused]] uint32_t x_pack = 0;
    if constexpr (!Q4NX && HALF)
    {
        constexpr int bits2 = 2 * bits + 1;
        constexpr int gspan = 18 + 3 * bits;
        const int t_offset = lane << 3;
        const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        const uint32_t hi7 = ((e7 - 1) / 32) % TWORDS, lo7 = ((e7 - gspan) / 32) % TWORDS, s7 = ((e7 - 1) / 32 + 1) * 32 - e7;
        const uint32_t hi3 = ((e3 - 1) / 32) % TWORDS, lo3 = ((e3 - gspan) / 32) % TWORDS, s3 = ((e3 - 1) / 32 + 1) * 32 - e3;
        x_pack = lo7 | (hi7 << 5) | (s7 << 10) | (lo3 << 15) | (hi3 << 20) | (s3 << 25);
    }
    #define XP(i) ((x_pack >> (5 * (i))) & 31u)   // 0: lo7, 1: hi7, 2: s7, 3: lo3, 4: hi3, 5: s3
    else if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    else if constexpr (!Q4NX && bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    // The packed register for each lane holds the word at that lane's
    // index. Shuffle in the two words of the lane's zero-prefixed window.
    [[maybe_unused]] const int q4nx_start = (2 * lane + 2) * STEP_BITS - 32;
    [[maybe_unused]] const int q4nx_first = max(q4nx_start, 0) / 32;
    [[maybe_unused]] const int q4nx_second = (q4nx_start + 32) / 32;

    __shared__ float sh_red[WK][ROWS][COLS];
    [[maybe_unused]] __shared__ uint32_t sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];

    for (int group = blockIdx.x; group < num_groups; group += gridDim.x)
    {
        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        // Prefetch ring (indices must be compile-time or pf lands in local memory)
        auto ld_b = [&] (int i, int l) -> uint32_t
        {
            if constexpr (LSTRIDE < 32)
                return lane < LSTRIDE ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        [[maybe_unused]] FragC q4nx_acc[WNT][2] = {};
        [[maybe_unused]] FragC_h ch[WNT][2] = {};
        [[maybe_unused]] float2 acc0[WNT][2] = {};

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (SMEM_STAGE)
            {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (LSTRIDE == 32 || lane < LSTRIDE)
                        sh_stage[warp][l * LSTRIDE + lane] = bw[l];
                __syncwarp();
            }

            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (Q4NX)
                {
                    const uint32_t before = __shfl_sync(0xffffffffu, bw[t], q4nx_first);
                    const uint32_t after = __shfl_sync(0xffffffffu, bw[t], q4nx_second);
                    const uint32_t window = __funnelshift_l(after, q4nx_start < 0 ? 0 : before, q4nx_start & 31);
                    dq_q4nx_window<STEP_BITS>(window, f0, f1);
                }
                else if constexpr (SMEM_STAGE)
                {
                    const uint32_t* tp = &sh_stage[warp][t * TWORDS];
                    if constexpr (HALF)
                        exl3_gemv_ns::dq8_regs_half<bits, cb>(tp[XP(0)], tp[XP(1)], XP(2), tp[XP(3)], tp[XP(4)], XP(5), f0, f1);
                    else if constexpr (bits == 4)
                        exl3_gemv_ns::dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
                    else if constexpr (bits == 2)
                        exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                    else
                        exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
                }
                else if constexpr (HALF)
                {
                    // 1.5 bpw: tile t lives in lanes (t & 1) * 12 .. +11 of load t / 2; 2.5 / 3.5 bpw: one tile per load
                    const uint32_t w = TWO_PER_LOAD ? bw[t >> 1] : bw[t];
                    const int base = TWO_PER_LOAD ? (t & 1) * TWORDS : 0;
                    uint32_t a7 = __shfl_sync(0xffffffffu, w, base + XP(0));
                    uint32_t b7 = __shfl_sync(0xffffffffu, w, base + XP(1));
                    uint32_t a3 = __shfl_sync(0xffffffffu, w, base + XP(3));
                    uint32_t b3 = __shfl_sync(0xffffffffu, w, base + XP(4));
                    exl3_gemv_ns::dq8_regs_half<bits, cb>(a7, b7, XP(2), a3, b3, XP(5), f0, f1);
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    // Two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                if constexpr (Q4NX)
                {
                    FragA a{{a01[0], a01[1], a23[0], a23[1]}};
                    ptx_mma_m16n8k16(a, f0, q4nx_acc[t][0]);
                    ptx_mma_m16n8k16(a, f1, q4nx_acc[t][1]);
                }
                else
                {
                    exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                    exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
                }
            }

            if constexpr (!Q4NX)
            {
                if ((d + 1) % FOLD == 0 || i + 1 == myn)
                {
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t)
                        #pragma unroll
                        for (int f = 0; f < 2; ++f)
                        {
                            acc0[t][f].x += __low2float(ch[t][f][0]);
                            acc0[t][f].y += __high2float(ch[t][f][0]);
                            ch[t][f][0] = hzero;
                        }
                }
            }
        }
        }

        // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
        // tile*16 + frag*8 + 2*(l%4) (+1)
        {
            const int c0 = 2 * (lane & 3);
            const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
            const int sr0 = MMODE == 0 ? 0 : r0;
            if (store0)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        const int col = t * 16 + f * 8 + c0;
                        sh_red[warp][sr0][col + 0] = Q4NX ? q4nx_acc[t][f][0] : acc0[t][f].x;
                        sh_red[warp][sr0][col + 1] = Q4NX ? q4nx_acc[t][f][1] : acc0[t][f].y;
                    }
            }
        }
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
        {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
            else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }

    #undef XP

    // Output scales and Hadamard transform, same semantics as the inner GEMM epilogue
    {
        grid.sync();

        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner<false, true>
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f * (Q4NX ? EXL3_Q4NX_ALPHA : 1.0f)
                );
            else
                had_hf_r_128_inner<false, true>
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
    }
}

#else  // __HIP_PLATFORM_AMD__ || USE_ROCM

// ROCm small-m GEMV.
//
// Same shape contract and output as the CUDA path above (EXL3_GEMM_ARGS, so the host
// argument array is interchangeable), but the execution model differs because RDNA has no
// tensor cores and no cooperative launch:
//
// - no mma: each lane decodes the eight weights of its own two columns of a 16x16 trellis
//   tile (the tensor-core B fragment layout, see reconstruct.cu) and folds them into FP32
//   with v_dot2_f32_f16, two k values per instruction. The four lanes that split a tile's
//   k rows are reduced with one butterfly, so the trellis is decoded once per tile and
//   never materialized.
// - no grid.sync: the host runs three launches (input Hadamard, GEMV, output Hadamard).
//   The input Hadamard writes the caller's A_had scratch exactly as the CUDA kernel's first
//   stage does, the GEMV reads A_had, and the output Hadamard reproduces the inner GEMM
//   epilogue (svh and 1/sqrt(128), times EXL3_Q4NX_ALPHA for Q4NX).
//
// Warps split k and never synchronize during the main loop, as on CUDA: B streams straight
// to registers, one 128 B word per lane per k-slice, and the words the window extraction
// needs are exchanged with ds_bpermute.

#include <cuda_runtime.h>
#include "../util.cuh"
#include "exl3_dq.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"

namespace exl3_gemv_hip_ns {

// ds_bpermute is the wave32 lane exchange; the HIP shuffle wrappers take a 64-bit mask on
// this platform, and a plain byte permute is what the hardware does anyway.
__device__ __forceinline__ uint32_t shfl32(uint32_t value, int src_lane)
{
    return (uint32_t) __builtin_amdgcn_ds_bpermute(src_lane << 2, (int) value);
}

// v_dot2_f32_f16: two FP16 products folded into an FP32 accumulator. The products of two
// FP16 values are exact in FP32, so this matches "decode to FP32, multiply, accumulate"
// bit for bit; only the accumulation order differs from the tensor-core path.
//
// The ROCm 10.1 clang has no __builtin_amdgcn_fdot2_f32_f16 (only the bf16 form), so the
// instruction is spelled out for the RDNA3/RDNA4 targets that implement it, and the
// portable form keeps every other target assembling.
__device__ __forceinline__ float dot2(half2 a, half2 b, float c)
{
#if defined(__GFX11__) || defined(__GFX12__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1202__)
    float d;
    asm("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
#else
    const float2 fa = __half22float2(a);
    const float2 fb = __half22float2(b);
    return fmaf(fa.x, fb.x, fmaf(fa.y, fb.y, c));
#endif
}

// Same eight-window decode as the CUDA path's decode8. decode_3inst_2 is bit-identical to
// the dp4a form of cb 2, and gfx12 has no dp4a, so one decoder serves every codebook.
template <int cb>
__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    f0[0] = decode_3inst_2<cb>(w0, w1);
    f0[1] = decode_3inst_2<cb>(w2, w3);
    f1[0] = decode_3inst_2<cb>(w4, w5);
    f1[1] = decode_3inst_2<cb>(w6, w7);
}

// Window extraction from two already-loaded words, same order as dq8_aligned_4bits. w[i] is
// the trellis window of weight i of the lane's eight (see the column/k mapping below), which
// the INT8 kernel needs without the codebook decode.
__device__ __forceinline__ void ext8_regs_4bits(uint32_t a, uint32_t b, uint32_t (&w)[8])
{
    uint32_t s;
    FSHF_IMM(s, b, a, 20);
    w[7] = b & 0xffff;
    BFE16_IMM(w[6], b, 4);
    BFE16_IMM(w[5], b, 8);
    BFE16_IMM(w[4], b, 12);
    BFE16_IMM(w[3], b, 16);
    w[2] = s & 0xffff;
    BFE16_IMM(w[1], s, 4);
    BFE16_IMM(w[0], s, 8);
}

template <int cb>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB& f0, FragB& f1)
{
    uint32_t w[8];
    ext8_regs_4bits(a, b, w);
    decode8<cb>(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], f0, f1);
}

// Register form of dq8_aligned_2bits: the two words and the funnel shift are lane-dependent
__device__ __forceinline__ void ext8_regs_2bits(uint32_t a, uint32_t b, int t_offset, uint32_t (&w)[8])
{
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w[7] = b & 0xffff;
    BFE16_IMM(w[6], b, 2);
    BFE16_IMM(w[5], b, 4);
    BFE16_IMM(w[4], b, 6);
    BFE16_IMM(w[3], b, 8);
    BFE16_IMM(w[2], b, 10);
    BFE16_IMM(w[1], b, 12);
    BFE16_IMM(w[0], b, 14);
}

template <int cb>
__device__ __forceinline__ void dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset, FragB& f0, FragB& f1)
{
    uint32_t w[8];
    ext8_regs_2bits(a, b, t_offset, w);
    decode8<cb>(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], f0, f1);
}

// Register form of dq8<3, cb, 4> with the per-lane funnel alignment precomputed
__device__ __forceinline__ void ext8_regs_3bits(uint32_t a, uint32_t b, int s2, uint32_t (&w)[8])
{
    w[7] = fshift(b, a, s2);
    w[6] = w[7] >> 3;
    w[5] = w[6] >> 3;
    w[4] = w[5] >> 3;
    w[3] = fshift(b, a, s2 + 12);
    w[2] = w[3] >> 3;
    w[1] = w[2] >> 3;
    w[0] = w[1] >> 3;
    w[7] &= 0xffff; w[6] &= 0xffff; w[5] &= 0xffff; w[4] &= 0xffff;
    w[3] &= 0xffff; w[2] &= 0xffff; w[1] &= 0xffff; w[0] &= 0xffff;
}

template <int cb>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1)
{
    uint32_t w[8];
    ext8_regs_3bits(a, b, s2, w);
    decode8<cb>(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], f0, f1);
}

// Register form of dq8_half<KA, cb> (exl3_dq.cuh); the four words and both funnel alignments
// are lane constants resolved by the caller.
template <int KA>
__device__ __forceinline__ void ext8_regs_half(uint32_t a7, uint32_t b7, int s7, uint32_t a3, uint32_t b3, int s3,
                                               uint32_t (&w)[8])
{
    w[7] = fshift(b7, a7, s7);
    w[6] = w[7] >> (KA + 1);
    w[5] = w[6] >> KA;
    w[4] = w[5] >> (KA + 1);
    w[3] = fshift(b3, a3, s3);
    w[2] = w[3] >> (KA + 1);
    w[1] = w[2] >> KA;
    w[0] = w[1] >> (KA + 1);
    w[7] &= 0xffff; w[6] &= 0xffff; w[5] &= 0xffff; w[4] &= 0xffff;
    w[3] &= 0xffff; w[2] &= 0xffff; w[1] &= 0xffff; w[0] &= 0xffff;
}

template <int KA, int cb>
__device__ __forceinline__ void dq8_regs_half(uint32_t a7, uint32_t b7, int s7, uint32_t a3, uint32_t b3, int s3,
                                              FragB& f0, FragB& f1)
{
    uint32_t w[8];
    ext8_regs_half<KA>(a7, b7, s7, a3, b3, s3, w);
    decode8<cb>(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], f0, f1);
}

}  // namespace exl3_gemv_hip_ns

// One warp per 16-column tile and k split; a block covers WT tiles (16 * WT columns) for
// every row, with WK warps splitting k. Lane l decodes the weights of columns l / 4 and
// l / 4 + 8 at k rows 2 * (l % 4), +1, +8, +9, so the two half2 words of the decoded
// fragment pair with two A half2 loads. MMODE 0 is the m == 1 fast path, MMODE 1 covers
// 2 <= m <= EXL3_GEMV_MAX_M with row-guarded loads.
template <int bits, int cb, bool c_fp32, int MMODE, int WK, int WT, bool HALF = false>
__global__ __launch_bounds__(32 * WK * WT)
void exl3_gemv_hip_kernel(EXL3_GEMM_ARGS)
{
    constexpr bool Q4NX = cb == EXL3_Q4NX_CODEBOOK;
    constexpr int STEP_BITS = 4 * bits + 2 * HALF;                 // bits per Q4NX state group
    constexpr int TWORDS = Q4NX ? 2 * STEP_BITS                    // uint32 per 16x16 tile
                                : (HALF ? 4 * (2 * bits + 1) : 8 * bits);
    constexpr int ROWS = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    constexpr int COLS = 16 * WT;
    constexpr int THREADS = 32 * WK * WT;
    constexpr int PF = 2;                                          // prefetch ring depth
    static_assert(TWORDS <= 32, "one warp load covers a trellis tile");
    if constexpr (Q4NX)
        static_assert(q4nx_supports_step_bits(STEP_BITS) && c_fp32 && (HALF || bits == 3));
    else if constexpr (HALF)
        static_assert(bits >= 1 && bits <= 3 && cb == 2, "half-integer rates are mul1-only");
    else
        static_assert(bits == 2 || bits == 3 || bits == 4);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int ksplit = warp % WK;
    const int tile = warp / WK;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int chunk = (kslices + WK - 1) / WK;
    const int ks0 = ksplit * chunk;
    const int ks1 = min(ks0 + chunk, kslices);
    const int rows_out = MMODE == 0 ? 1 : min(size_m, EXL3_GEMV_MAX_M);

    // Fragment lane mapping (see reconstruct.cu): column l / 4, k rows 2 * (l % 4) + {0,1,8,9}
    const int q = lane & 3;
    const int c0 = lane >> 2;

    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const int ntile = blockIdx.x * WT + tile;
    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + (size_t) ntile * TWORDS;

    // Per-lane extraction constants, same derivation as the CUDA kernel
    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    int x_lo7 = 0, x_hi7 = 0, x_s7 = 0, x_lo3 = 0, x_hi3 = 0, x_s3 = 0;
    int q4nx_start = 0, q4nx_first = 0, q4nx_second = 0;
    if constexpr (Q4NX)
    {
        q4nx_start = (2 * lane + 2) * STEP_BITS - 32;
        q4nx_first = max(q4nx_start, 0) / 32;
        q4nx_second = (q4nx_start + 32) / 32;
    }
    else if constexpr (HALF)
    {
        constexpr int bits2 = 2 * bits + 1;
        constexpr int gspan = 18 + 3 * bits;
        const int t_offset = lane << 3;
        const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        x_hi7 = ((e7 - 1) / 32) % TWORDS; x_lo7 = ((e7 - gspan) / 32) % TWORDS; x_s7 = ((e7 - 1) / 32 + 1) * 32 - e7;
        x_hi3 = ((e3 - 1) / 32) % TWORDS; x_lo3 = ((e3 - gspan) / 32) % TWORDS; x_s3 = ((e3 - 1) / 32 + 1) * 32 - e3;
    }
    else if constexpr (bits == 2)
    {
        const int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    else if constexpr (bits == 3)
    {
        const int t_offset = lane << 3;
        const int b1 = (t_offset + 257) * 3;
        const int b2 = b1 + 21;
        const int i0 = (b1 - 16) / 32;
        const int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    // Each lane holds the word at its own index; the window extraction shuffles the two
    // (or four) words the lane's zero-prefixed window needs. Lanes past the tile's width
    // idle their load.
    auto load = [&] (int ks) -> uint32_t
    {
        if (ks >= ks1) return 0u;
        const uint32_t* tp = bp + (size_t) (ks - ks0) * slice_stride;
        return lane < TWORDS ? tp[lane] : 0u;
    };

    uint32_t pf[PF];
    #pragma unroll
    for (int d = 0; d < PF; ++d) pf[d] = load(ks0 + d);

    float acc[ROWS][2];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
    }

    const half2* A2 = reinterpret_cast<const half2*>(A_had);
    const int a_row_stride = size_k >> 1;   // half2 per row

    for (int ks = ks0; ks < ks1; ++ks)
    {
        const half2* ap = A2 + (size_t) ks * 8 + q;
        const uint32_t bw = pf[0];
        pf[0] = pf[1];
        pf[1] = load(ks + PF);

        FragB f0, f1;
        if constexpr (Q4NX)
        {
            const uint32_t before = exl3_gemv_hip_ns::shfl32(bw, q4nx_first);
            const uint32_t after = exl3_gemv_hip_ns::shfl32(bw, q4nx_second);
            const uint32_t window = __funnelshift_l(after, q4nx_start < 0 ? 0u : before, q4nx_start & 31);
            dq_q4nx_window<STEP_BITS>(window, f0, f1);
        }
        else if constexpr (HALF)
        {
            const uint32_t a7 = exl3_gemv_hip_ns::shfl32(bw, x_lo7);
            const uint32_t b7 = exl3_gemv_hip_ns::shfl32(bw, x_hi7);
            const uint32_t a3 = exl3_gemv_hip_ns::shfl32(bw, x_lo3);
            const uint32_t b3 = exl3_gemv_hip_ns::shfl32(bw, x_hi3);
            exl3_gemv_hip_ns::dq8_regs_half<bits, cb>(a7, b7, x_s7, a3, b3, x_s3, f0, f1);
        }
        else if constexpr (bits == 4)
        {
            exl3_gemv_hip_ns::dq8_regs_4bits<cb>(exl3_gemv_hip_ns::shfl32(bw, (lane + 31) & 31), bw, f0, f1);
        }
        else if constexpr (bits == 2)
        {
            exl3_gemv_hip_ns::dq8_regs_2bits<cb>(exl3_gemv_hip_ns::shfl32(bw, x_src_a),
                                               exl3_gemv_hip_ns::shfl32(bw, x_src_b), lane << 3, f0, f1);
        }
        else
        {
            exl3_gemv_hip_ns::dq8_regs_3bits<cb>(exl3_gemv_hip_ns::shfl32(bw, x_src_a),
                                               exl3_gemv_hip_ns::shfl32(bw, x_src_b), x_s2, f0, f1);
        }

        #pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            if (MMODE == 1 && r >= rows_out) break;
            const half2 a01 = ap[0];
            const half2 a89 = ap[4];
            ap += a_row_stride;
            acc[r][0] = exl3_gemv_hip_ns::dot2(a01, f0[0], acc[r][0]);
            acc[r][0] = exl3_gemv_hip_ns::dot2(a89, f0[1], acc[r][0]);
            acc[r][1] = exl3_gemv_hip_ns::dot2(a01, f1[0], acc[r][1]);
            acc[r][1] = exl3_gemv_hip_ns::dot2(a89, f1[1], acc[r][1]);
        }
    }

    // Fold the four k splits of a tile: lanes 4c + q are contiguous.
    #pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
        if (MMODE == 1 && r >= rows_out) break;
        acc[r][0] += __shfl_xor_sync(__activemask(), acc[r][0], 1);
        acc[r][0] += __shfl_xor_sync(__activemask(), acc[r][0], 2);
        acc[r][1] += __shfl_xor_sync(__activemask(), acc[r][1], 1);
        acc[r][1] += __shfl_xor_sync(__activemask(), acc[r][1], 2);
    }

    __shared__ float sh_red[WK][ROWS][COLS];
    if (q == 0)
    {
        const int c = tile * 16 + c0;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            if (MMODE == 1 && r >= rows_out) break;
            sh_red[ksplit][r][c] = acc[r][0];
            sh_red[ksplit][r][c + 8] = acc[r][1];
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
    {
        const int r = idx / COLS;
        const int c = idx - r * COLS;
        float sum = sh_red[0][r][c];
        #pragma unroll
        for (int j = 1; j < WK; ++j) sum += sh_red[j][r][c];
        const int col = blockIdx.x * COLS + c;
        if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
        else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
    }
}

// Input Hadamard + suh, one warp per 128 elements. Same arithmetic as the CUDA kernel's
// first stage; a separate launch replaces its grid.sync(). Internal linkage so the fp16 and
// int8 GEMV translation units can each include this header.
static __global__ __launch_bounds__(256)
void exl3_gemv_hip_had_in_kernel
(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ suh,
    const int size_k,
    const int total
)
{
    const int warp = (blockIdx.x * 256 + threadIdx.x) >> 5;
    const int off = warp * 128;
    if (off >= total) return;
    had_hf_r_128_inner<true, false>(input + off, output + off, suh + (off % size_k), 0.088388347648f);
}

// Output Hadamard + svh, in place, one warp per 128 elements.
template <bool c_fp32>
__global__ __launch_bounds__(256)
void exl3_gemv_hip_had_out_kernel
(
    const void* __restrict__ input,
    void* __restrict__ output,
    const half* __restrict__ svh,
    const int size_n,
    const int total,
    const float r_scale
)
{
    const int warp = (blockIdx.x * 256 + threadIdx.x) >> 5;
    const int off = warp * 128;
    if (off >= total) return;
    if constexpr (c_fp32)
        had_ff_r_128_inner<false, true>
        (
            ((const float*) input) + off, ((float*) output) + off, svh + (off % size_n), r_scale
        );
    else
        had_hf_r_128_inner<false, true>
        (
            ((const half*) input) + off, ((half*) output) + off, svh + (off % size_n), r_scale
        );
}

#endif
