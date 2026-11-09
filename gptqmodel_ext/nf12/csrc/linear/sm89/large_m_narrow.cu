// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 / NF12-AD kernel computation.
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original implementation: https://github.com/dphnAI/nf12
// Copyright (c) 2026 AlpinDale and the nf12 contributors.
// Licensed under the Apache License, Version 2.0.
//
// Vendored into GPT-QModel (https://github.com/ModelCloud/GPTQModel) and
// modified for the NF12-AD (AlpinDale) method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
#include "linear/sm89/large_m_narrow.h"
#include "linear/mma.cuh"

namespace nf12::detail {
namespace {
struct NarrowParams {
  const __nv_bfloat16* x;
  int64_t ldx;
  const uint8_t* weight;
  const uint4* escapes;
  uint32_t base_x4;
  int m, n, kb;
  const __nv_bfloat16* bias;
  __nv_bfloat16* y;
  int64_t ldy;
};

// Each warp owns an n16 x BM tile over a contiguous K interval. Separate
// activation rings allow warp-local waits until the final CTA FP32 reduction.
// No persistent state or workspace.
template <int BM, int SPLITS, WeightSource S, int SLOTS = 2>
__global__ void __launch_bounds__(32 * SPLITS) narrow_kernel(NarrowParams p) {
  using WT = WeightTraits<S>;
  constexpr int XB = BM * 128;
  constexpr int TF = BM / 8;
  extern __shared__ __align__(128) uint8_t ring[];
  auto* partial = reinterpret_cast<float4 (*)[TF][32]>(ring + SPLITS * SLOTS * XB);
  const int lane = threadIdx.x % 32, rank = threadIdx.x / 32;
  const int n0 = blockIdx.x * 16, m0 = blockIdx.y * BM;
  const int begin = p.kb * rank / SPLITS, end = p.kb * (rank + 1) / SPLITS;
  const uint32_t sm = smem_addr(ring + rank * SLOTS * XB);
  auto load = [&](int kb, int slot) {
#pragma unroll
    for (int c = lane; c < BM * 8; c += 32) {
      const int row = c / 8, ch = c % 8;
      const bool in = m0 + row < p.m;
      const auto* src = p.x + static_cast<int64_t>(in ? m0 + row : 0) * p.ldx + kb * 64 + ch * 8;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
        "r"(mma::act_chunk_dst(sm + slot * XB, row, ch)), "l"(src), "r"(in ? 16u : 0u));
    }
    cp_async_commit();
  };
  float acc[TF][4] = {};
  if (begin < end) load(begin, 0);
  for (int kb = begin; kb < end; ++kb) {
    const int slot = (kb - begin) % SLOTS;
    uint4 planes[WT::kLoads];
    const auto* src = p.weight + (static_cast<int64_t>(n0 / 16) * p.kb + kb) * WT::kSuperAtomBytes + lane * 16;
#pragma unroll
    for (int j = 0; j < WT::kLoads; ++j) planes[j] = ldg_stream(src + j * kPlaneBytes);
    cp_async_wait<0>();
    __syncwarp();
    if constexpr (SLOTS > 1) {
      if (kb + 1 < end) load(kb + 1, 1 - slot);
    }
    uint32_t a[4][4];
    WT::fragments(planes, p.base_x4, p.escapes, a);
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t b[TF][2];
      if constexpr (BM == 8) {
        const int row = lane & 7, hi = (lane >> 3) & 1;
        const uint32_t addr = sm + slot * XB + row * 128 + (((2 * kk + hi) ^ row) * 16);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(b[0][0]), "=r"(b[0][1]) : "r"(addr));
      } else {
        const mma::BFragments<mma::Tile<16, BM, 1, 1, 2>> bf(0, lane);
        bf.load(b, sm + slot * XB, kk);
      }
#pragma unroll
      for (int t = 0; t < TF; ++t) mma_bf16_16816(acc[t], a[kk], b[t][0], b[t][1]);
    }
    __syncwarp();
    if constexpr (SLOTS == 1) {
      if (kb + 1 < end) load(kb + 1, 0);
    }
  }
#pragma unroll
  for (int t = 0; t < TF; ++t)
    partial[rank][t][lane] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
  __syncthreads();
  // Different warp ranks own different token fragments of the final tile.
  for (int t = rank; t < TF; t += SPLITS) {
    float4 sum = partial[0][t][lane];
#pragma unroll
    for (int r = 1; r < SPLITS; ++r) {
      const float4 v = partial[r][t][lane];
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    const int g = lane >> 2, col = n0 + g, tok = m0 + t * 8 + 2 * (lane & 3);
    const float blo = p.bias ? __bfloat162float(p.bias[col]) : 0.f;
    const float bhi = p.bias ? __bfloat162float(p.bias[col + 8]) : 0.f;
    if (tok < p.m) {
      p.y[static_cast<int64_t>(tok) * p.ldy + col] = __float2bfloat16_rn(sum.x + blo);
      p.y[static_cast<int64_t>(tok) * p.ldy + col + 8] = __float2bfloat16_rn(sum.z + bhi);
    }
    if (tok + 1 < p.m) {
      p.y[static_cast<int64_t>(tok + 1) * p.ldy + col] = __float2bfloat16_rn(sum.y + blo);
      p.y[static_cast<int64_t>(tok + 1) * p.ldy + col + 8] = __float2bfloat16_rn(sum.w + bhi);
    }
  }
}

template <int BM, int SPLITS, int SLOTS = 2>
cudaError_t launch(const LinearArgs& a, const NarrowParams& p) {
  const dim3 grid(a.n / 16, (a.m + BM - 1) / BM);
  constexpr int bytes = SPLITS * SLOTS * BM * 128 + SPLITS * (BM / 8) * 32 * sizeof(float4);
  if (a.source == WeightSource::kNF12) {
    constexpr auto kernel = narrow_kernel<BM, SPLITS, WeightSource::kNF12, SLOTS>;
    static std::atomic<int> granted[64];
    auto err = mma::ensure_smem(reinterpret_cast<const void*>(kernel), granted, a.dev->device, bytes);
    if (err != cudaSuccess) return err;
    kernel<<<grid, 32 * SPLITS, bytes, a.stream>>>(p);
  } else {
    constexpr auto kernel = narrow_kernel<BM, SPLITS, WeightSource::kBF16T, SLOTS>;
    static std::atomic<int> granted[64];
    auto err = mma::ensure_smem(reinterpret_cast<const void*>(kernel), granted, a.dev->device, bytes);
    if (err != cudaSuccess) return err;
    kernel<<<grid, 32 * SPLITS, bytes, a.stream>>>(p);
  }
  return cudaGetLastError();
}
}  // namespace

bool narrow_sm8x_supported(const DeviceInfo& dev) { return dev.sm_arch == 89; }

int narrow_sm8x_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo&) {
  // Grow token tiles beyond one output wave to amortize weight decoding.
  const int64_t area = m * n;
  if (area <= 8192) return k <= 2048 ? 8 : 9;
  if (area <= 32768) return 6;
  return 10;
}

cudaError_t linear_large_m_sm8x_narrow(const LinearArgs& a, int config) {
  if (!narrow_sm8x_supported(*a.dev)) return cudaErrorNotSupported;
  if (a.m < 1 || a.m > 16384 || a.n < 16 || a.n > 256 || a.n % 16 ||
      a.k < 512 || a.k > 16384 || a.k % 64) return cudaErrorNotSupported;
  NarrowParams p{a.x, a.ldx, a.weight, a.escapes, exp_base_x4(a.exp_base), static_cast<int>(a.m),
                 static_cast<int>(a.n), static_cast<int>(a.k / 64), a.bias, a.y, a.ldy};
  switch (config < 0 ? narrow_sm8x_heuristic(a.m, a.n, a.k, *a.dev) : config) {
    case 6: return launch<16, 16>(a, p);
    case 8: return launch<8, 16>(a, p);
    case 9: return launch<8, 32>(a, p);
    case 10: return launch<32, 16, 1>(a, p);
    default: return cudaErrorInvalidValue;
  }
}
}  // namespace nf12::detail
