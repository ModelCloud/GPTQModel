// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Native EXL3 tensor-core GEMM for RDNA4 (gfx120x) on ROCm/HIP.
//
// The CUDA implementation in `exl3_gemm.cu` leans on NVIDIA `mma.sync` PTX and a
// cooperative split-K kernel table that is empty on HIP, so ROCm used to fall
// back to `reconstruct` + `hgemm`: decode the whole `[K, N]` weight matrix into
// FP16 in global memory and hand it to hipBLAS. That round trip costs
// `4 * K * N` bytes of HBM traffic per call (write plus read) and dominates
// every shape.
//
// This kernel decodes the trellis tile straight into the wave32 WMMA operand
// registers and accumulates in FP32, so the expanded weight never reaches
// memory. The surrounding Hadamard rotations (`suh` before, `svh` after) reuse
// the portable `had_r_128` kernel, which keeps the numerics identical to the
// reconstruct path.
//
// Layout of `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` on gfx12
// (probed on hardware; every fragment is 8 halves/floats per lane):
//
//   A (M16 x K16): lane l, slot i -> (row = (l&7) + 8*((l>>3)&1), k = i + 8*(l>>4))
//   B (K16 x N16): lane l, slot i -> (k = i + 8*(l>>4),              col = l&15)
//   C (M16 x N16): lane l, slot i -> (row = i + 8*(l>>4),            col = l&15)
//
// A is therefore eight consecutive `k` of one row: a 16-byte vector load out of
// the row-major activation. C is eight consecutive rows of one column.
//
// B is the awkward one. The packed trellis is bit-interleaved so that
// `dq_dispatch` hands lane l eight consecutive *tile positions*, and those
// positions are (row, col) = (2q + (j&1) + 8*((j>>1)&1),
// 2*(l>>3) + ((l>>2)&1) + 8*(j>>2)) for q = l&3 and j = 0..7. B wants eight
// consecutive `k` of a single column instead, which is four lanes' worth of
// `half2` (rows 2q, 2q+1). Each wave stages its 16x16 tile through a private
// 512-byte shared-memory buffer written column-major (`s[col][k]`): the decode
// becomes four 32-bit stores and the operand fetch one 128-bit load. The
// transposition stays inside one wave, so `__syncwarp()` is the only barrier.

#include <cuda_fp16.h>

#include "../../torch_stable_compat.h"

#include <cstring>

#include "../frag.cuh"
#include "../hgemm.cuh"
#include "../util.h"
#include "../util.cuh"
#include "exl3_dq.cuh"
#include "exl3_gemm_hip.cuh"
#include "hadamard.cuh"
#include "reconstruct.cuh"

// gfx12 (RDNA4) is the only family with this wave32 16x16x16 FP16 WMMA and this
// operand layout. The macro is per offload target: a fat binary that also
// carries, say, gfx90a compiles the NaN stub there, and the host dispatcher
// refuses to launch on anything that is not gfx120x at runtime.
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12__)
#define EXL3_HIP_WMMA_GFX12 1
#else
#define EXL3_HIP_WMMA_GFX12 0
#endif

// Rows of `m` per block. Each warp decodes one 16-column trellis tile and reuses
// it across `EXL3_HIP_MT` 16-row activation tiles, which is what amortizes the
// decode: without the reuse a warp issues a single WMMA per 256 decoded weights.
#define EXL3_HIP_MT 4
#define EXL3_HIP_MAX_WARPS 8

// Above this many activation rows the `m`-fold re-decode (one per 64-row block)
// costs more than expanding the weight once for hipBLAS, so the caller's
// reconstruct fallback wins. Measured, not guessed: see the ratio table in the
// pull request.
#define EXL3_HIP_MAX_M 512

// Declared outside the target guard: the host pass of the HIP compilation also
// parses the kernel body, and `acc` is shared by the gfx12 body and the stub.
typedef _Float16 exl3_h8 __attribute__((ext_vector_type(8)));
typedef float exl3_f8 __attribute__((ext_vector_type(8)));

namespace {

#if EXL3_HIP_WMMA_GFX12
__device__ __forceinline__ void stage_b_fragment(half2* sw, int lane, FragB& f0, FragB& f1)
{
    // j = 0,1 is a `half2` at (rows 2q, 2q+1) of one column and j = 2,3 is the
    // same column eight rows down. Writing `sw[col][k]` puts the four halves of
    // a column pair next to each other, which is what the B fragment wants.
    const int q = lane & 3;
    const int c = 2 * (lane >> 3) + ((lane >> 2) & 1);
    sw[c * 8 + q] = f0[0];
    sw[c * 8 + q + 4] = f0[1];
    sw[(c + 8) * 8 + q] = f1[0];
    sw[(c + 8) * 8 + q + 4] = f1[1];
}
#endif

template <int bits, int cb, bool half_k, int MT>
__global__ __launch_bounds__(EXL3_HIP_MAX_WARPS * 32)
void exl3_hip_gemm_kernel
(
    const half* __restrict__ A,      // [size_m, size_k], Hadamard-rotated
    const uint16_t* __restrict__ B,  // [size_k/16, size_n/16, 16*bits + 8*half_k]
    float* __restrict__ C,           // [size_m, size_n]
    int size_m, int size_k, int size_n, int mt_tiles
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_tile = blockIdx.x * (blockDim.x >> 5) + warp;
    const int m0 = blockIdx.y * MT * 16;
    const int mt_left = mt_tiles - int(blockIdx.y) * MT;
    const int mt_count = mt_left < MT ? mt_left : MT;

    // C: lane l, slot i -> row i + 8*(l>>4), column l&15.
    const int c_row = m0 + ((lane >> 4) << 3);
    const int c_col = n_tile * 16 + (lane & 15);

    exl3_f8 acc[MT];

#if EXL3_HIP_WMMA_GFX12
    constexpr int words = 16 * bits + (half_k ? 8 : 0);
    const uint16_t* bp = B + (int64_t) n_tile * words;
    const int64_t b_step = (int64_t) (size_n >> 4) * words;
    // A row/slot pair: row = (l&7) + 8*((l>>3)&1), k base = 8*(l>>4).
    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const half* ap = A + (int64_t) (m0 + a_row) * size_k + 8 * (lane >> 4);

    // 16 columns x 16 halves per warp, 16-byte aligned so the operand fetch is a
    // single 128-bit load.
    __shared__ __align__(16) half2 stage[EXL3_HIP_MAX_WARPS][16][8];
    half2* sw = &stage[warp][0][0];

    bool a_valid[MT];
    #pragma unroll
    for (int t = 0; t < MT; ++t) a_valid[t] = (m0 + t * 16 + a_row) < size_m;
    #pragma unroll
    for (int t = 0; t < MT; ++t)
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.0f;
    }

    const int k_tiles = size_k >> 4;
    for (int kt = 0; kt < k_tiles; ++kt)
    {
        FragB f0, f1;
        dq_dispatch<bits, cb, half_k>(reinterpret_cast<const uint32_t*>(bp), lane * 8, f0, f1);
        stage_b_fragment(sw, lane, f0, f1);
        __syncwarp();

        exl3_h8 bf;
        __builtin_memcpy(&bf, &sw[(lane & 15) * 8 + ((lane >> 4) << 2)], sizeof(bf));

        #pragma unroll
        for (int t = 0; t < MT; ++t)
        {
            if (t < mt_count)
            {
                // A row past `size_m` is zero-filled so the partial last tile
                // accumulates a clean zero instead of reading out of bounds.
                exl3_h8 af = {};
                if (a_valid[t]) __builtin_memcpy(&af, ap + (int64_t) t * 16 * size_k, sizeof(af));
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf, acc[t]);
            }
        }
        bp += b_step;
        ap += 16;
    }
#else
    // Unreachable: the host dispatcher only launches on a gfx120x device, whose
    // code object takes the branch above. Poison the output so a mis-dispatch is
    // loud instead of silently returning zeros.
    (void) lane; (void) warp; (void) size_k;
    #pragma unroll
    for (int t = 0; t < MT; ++t)
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = __int_as_float(0x7fc00000);
    }
#endif

    #pragma unroll
    for (int t = 0; t < MT; ++t)
    {
        if (t < mt_count)
        {
            const int row = c_row + t * 16;
            float* cp = C + (int64_t) row * size_n + c_col;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                if (row + i < size_m) cp[(int64_t) i * size_n] = acc[t][i];
            }
        }
    }
}

template <int bits, int cb, bool half_k>
void launch_native
(
    const half* a, const uint16_t* b, float* c,
    int size_m, int size_k, int size_n, int warps, dim3 grid, cudaStream_t stream
)
{
    exl3_hip_gemm_kernel<bits, cb, half_k, EXL3_HIP_MT><<<grid, warps * 32, 0, stream>>>
    (
        a, b, c, size_m, size_k, size_n, (size_m + 15) >> 4
    );
}

template <int cb>
bool dispatch_bits
(
    int bits, const half* a, const uint16_t* b, float* c,
    int size_m, int size_k, int size_n, int warps, dim3 grid, cudaStream_t stream
)
{
    switch (bits)
    {
        case 1: launch_native<1, cb, false>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
        case 2: launch_native<2, cb, false>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
        case 3: launch_native<3, cb, false>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
        case 4: launch_native<4, cb, false>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
        case 5: launch_native<5, cb, false>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
        case 6: launch_native<6, cb, false>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
        case 7: launch_native<7, cb, false>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
        case 8: launch_native<8, cb, false>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
        default: return false;
    }
}

bool dispatch_native
(
    int bits, int cb, bool half_k, const half* a, const uint16_t* b, float* c,
    int size_m, int size_k, int size_n, int warps, dim3 grid, cudaStream_t stream
)
{
    if (half_k)
    {
        // Half-integer rates only exist for the mul1 codebook (1.5/2.5/3.5 bpw).
        if (cb != 2) return false;
        switch (bits)
        {
            case 1: launch_native<1, 2, true>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
            case 2: launch_native<2, 2, true>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
            case 3: launch_native<3, 2, true>(a, b, c, size_m, size_k, size_n, warps, grid, stream); return true;
            default: return false;
        }
    }
    switch (cb)
    {
        case 0: return dispatch_bits<0>(bits, a, b, c, size_m, size_k, size_n, warps, grid, stream);
        case 1: return dispatch_bits<1>(bits, a, b, c, size_m, size_k, size_n, warps, grid, stream);
        case 2: return dispatch_bits<2>(bits, a, b, c, size_m, size_k, size_n, warps, grid, stream);
        default: return false;
    }
}

}  // namespace

bool exl3_gemm_hip_available(int device)
{
    const auto* prop = at::cuda::getDeviceProperties(device);
    // gfx1200 / gfx1201 are RDNA4; a suffix such as `:sramecc-` is ignored.
    return prop && std::strncmp(prop->gcnArchName, "gfx120", 6) == 0;
}

int exl3_gemm_hip
(
    const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
    const at::Tensor& suh, const at::Tensor& A_had, const at::Tensor& svh,
    int size_m, int size_k, int size_n, int K, bool half_k, int cb, bool c_fp32
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = A.get_device();

    TORCH_CHECK(cb != EXL3_Q4NX_CODEBOOK,
                "EXL3 Q4NX inference is not available on ROCm/HIP; use the torch "
                "reconstruct path (EXL3_TORCH) for Q4NX checkpoints");
    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous()
                && A_had.is_contiguous(),
                "EXL3 GEMM requires contiguous buffers");
    TORCH_CHECK(size_k % 128 == 0 && size_n % 128 == 0,
                "EXL3 fused transforms require K and N divisible by 128");

    auto x = at::view(A, {size_m, size_k});
    auto xh = at::view(A_had, {size_m, size_k});
    auto y = at::view(C, {size_m, size_n});

    // Shared by both paths: the rotation of the activation is what turns the
    // trellis-domain weight into a plain FP16 matrix product.
    had_r_128(x, xh, suh, c10::nullopt, 1.0f);

    const bool native = c_fp32 && size_m > 0 && size_m <= EXL3_HIP_MAX_M
                        && exl3_gemm_hip_available(device);
    if (native)
    {
        // Spread the work over the device when a wide `n` would otherwise leave
        // most CUs idle on a small grid: shrink the block until the grid has at
        // least two blocks per CU (or the block is a single wave).
        const int sms = at::cuda::getDeviceProperties(device)->multiProcessorCount;
        const int mt_blocks = (size_m + EXL3_HIP_MT * 16 - 1) / (EXL3_HIP_MT * 16);
        int warps = EXL3_HIP_MAX_WARPS;
        while (warps > 1 && (size_n / (16 * warps)) * mt_blocks < 2 * sms) warps >>= 1;
        dim3 grid(size_n / (16 * warps), mt_blocks);
        if (dispatch_native(K, cb, half_k,
                            reinterpret_cast<const half*>(xh.const_data_ptr()),
                            reinterpret_cast<const uint16_t*>(B.const_data_ptr()),
                            reinterpret_cast<float*>(y.data_ptr()),
                            size_m, size_k, size_n, warps, grid, stream))
        {
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            had_r_128(y, y, c10::nullopt, svh, 1.0f);
            return EXL3_HIP_DISPATCH_NATIVE;
        }
    }

    // Portable fallback: expand the weight once, then let hipBLAS do the GEMM.
    // This is the only path for half-precision accumulators, very wide batches
    // and non-RDNA4 devices.
    auto weight = at::empty({size_k, size_n}, at::options(A));
    if (half_k) reconstruct_frac(weight, B, K);
    else reconstruct(weight, B, K, cb == 1, cb == 2);
    hgemm(xh, weight, y);
    had_r_128(y, y, c10::nullopt, svh, 1.0f);
    return EXL3_HIP_DISPATCH_FALLBACK;
}
