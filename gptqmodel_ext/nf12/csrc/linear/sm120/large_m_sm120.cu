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
// Dense SM120 GEMM using linear/sm120/large_m_sm120_kernel.cuh.
// Tiles [0, dp_tiles) are data-parallel. Remaining k-blocks are partitioned
// into contiguous stream-K ranges. A split tile's k = 0 segment is its
// owner's last segment, while other contributors process theirs first and
// publish FP32 partials followed by tags. Tags are zeroed before every launch
// so prior workspace contents cannot satisfy a wait. The owner reduces in
// ascending CTA order, fixed by shape, config and grid.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>

#include "linear/sm120/large_m_sm120.h"
#include "linear/sm120/large_m_sm120_kernel.cuh"
#include "nf12/format.cuh"

namespace nf12::detail {
namespace {

using sm120k::kThreads;
constexpr int kMaxDevices = 64;  // per-device host caches

struct DenseParams {
  const uint4* escapes;
  uint32_t num_escapes;
  uint32_t base_x4;
  const __nv_bfloat16* bias;
  __nv_bfloat16* y;
  int64_t ldy;
  int m, n, kb;  // kb = k / 64
  int n_tiles, m_tiles;
  int group_n;       // raster: n-tiles per group; n varies fastest inside a group
  int dp_tiles;      // tiles [0, dp_tiles) are data-parallel
  // Stream-K splits (tiles - dp_tiles) * kb k-blocks (< 2^31) into sk_groups
  // equal slices across sk_group_ctas CTAs each. sk_q/sk_rem hold quotient
  // and remainder to avoid device-side 64-bit division.
  int sk_units, sk_groups, sk_group_ctas, sk_per_group, sk_q, sk_rem;
  int split;         // kSplit schedule: group g = tile g, reduced by all of its CTAs (see epilogue)
  float* partials;   // [grid][BN * BM] FP32, stream-K contributors
  uint64_t* tags;    // [grid]: publication tags, zeroed by the host before every launch
};

#if defined(NF12_SM120_DEVICE)
// Nonzero tag for (tile, first k-block), published after the partial. Owners
// wait for an exact match. The host zeroes all slots on the launch stream,
// including graph capture, so stale workspace tags cannot satisfy a wait.
__device__ __forceinline__ uint64_t stream_k_tag(int tile, int kb0) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(tile) + 1u) << 32) | static_cast<uint32_t>(kb0);
}
#endif

class DenseProblem {
 public:
  using Params = DenseParams;
  struct Work {
    int tile, kb0, kb1;
    int nt, mt;
  };
  static constexpr int kExtraBytes = 0;
  static constexpr bool kXWarp = false;
  static constexpr int kXArrivals = 1;  // lane 0's expect_tx

#if defined(NF12_SM120_DEVICE)
  __device__ static void init(uint32_t) {}

  __device__ DenseProblem(const Params& p, uint8_t*, uint32_t)
      : p_(p), unit_(static_cast<int>(blockIdx.x)), units_(static_cast<int>(gridDim.x)) {
    dp_ = unit_;
    const int pos = sk_begin(unit_);
    sk_left_ = sk_begin(unit_ + 1) - pos;
    if (sk_left_ > 0) {
      sk_tile_ = pos / p_.kb;
      sk_k0_ = pos - sk_tile_ * p_.kb;
    }
  }

  template <class C>
  __device__ bool next(Work& w) {
    if (dp_ < p_.dp_tiles) {
      w.tile = dp_;
      w.kb0 = 0;
      w.kb1 = p_.kb;
      dp_ += units_;
    } else if (sk_left_ > 0) {
      const int k1 = sk_k0_ + min(p_.kb - sk_k0_, sk_left_);
      w.tile = p_.dp_tiles + sk_tile_;
      w.kb0 = sk_k0_;
      w.kb1 = k1;
      sk_left_ -= k1 - sk_k0_;
      ++sk_tile_;  // later segments start at k = 0 of the next tile
      sk_k0_ = 0;
    } else {
      return false;
    }
    tile_raster(w.tile, p_.n_tiles, p_.m_tiles, p_.group_n, w.nt, w.mt);
    return true;
  }

  template <class C>
  __device__ int w_block(const Work& w) const {
    return w.nt * C::kBlocks;
  }
  // Weights read by a single m-tile are dead after this pass.
  __device__ bool w_streamed_once(const Work&) const { return p_.m_tiles == 1; }
  template <class C>
  __device__ void x_tile(const Work&, int) const {}
  template <class C>
  __device__ void x_load(const CUtensorMap* xmap, uint32_t dst, int kb, const Work& w, uint32_t bar, uint64_t policy,
                         int) const {
    sm120::mbar_arrive_expect_tx(bar, C::kXBytes);  // rows past m are zero-filled and counted
    sm120::tma_load_2d(dst, xmap, kb * 64, w.mt * C::BM, bar, policy);
  }
  __device__ uint32_t base_x4(const Work&) const { return p_.base_x4; }
  __device__ const uint4* escapes(const Work&) const { return p_.escapes; }
  __device__ uint32_t num_escapes(const Work&) const { return p_.num_escapes; }
  __device__ int n() const { return p_.n; }
  __device__ const void* zfill_src() const { return p_.y; }

  // Finish whole tiles or reduce stream-K partials, then add bias in FP32.
  // Transpose each 8x8 fragment to token g, rows 2t/2t+1 for row-major stores.
  template <class C, int RB, int TF>
  __device__ void epilogue(float (&acc)[RB][TF][4], const Work& w, int cw, int lane, int valid_rb) const {
    using namespace sm120;
    constexpr int kPartial = C::BN * C::BM;  // floats per CTA slot
    auto partial_at = [&](int slot, int rb, int tf) {
      return reinterpret_cast<float4*>(p_.partials + static_cast<int64_t>(slot) * kPartial) +
             ((cw * RB + rb) * TF + tf) * 32 + lane;
    };
    if (p_.split) {
      split_epilogue<C, RB, TF>(acc, w, cw, lane);
      return;
    }
    if (w.kb0 != 0) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int tf = 0; tf < TF; ++tf)
          __stcg(partial_at(unit_, rb, tf), make_float4(acc[rb][tf][0], acc[rb][tf][1], acc[rb][tf][2],
                                                        acc[rb][tf][3]));
      named_bar_sync(1, 32 * sm120k::kConsumerWarps);
      if (cw == 0 && lane == 0) {
        __threadfence();
        st_release64(&p_.tags[unit_], stream_k_tag(w.tile, w.kb0));
      }
      return;
    }
    if (w.kb1 != p_.kb) {
      // Contributors are the following CTAs whose range starts inside this tile.
      const int t0 = (w.tile - p_.dp_tiles) * p_.kb;
      for (int u = unit_ + 1; u < units_; ++u) {
        const int a = sk_begin(u);
        if (a >= t0 + p_.kb) break;
        if (a == sk_begin(u + 1)) continue;  // empty range
        if (cw == 0 && lane == 0) {
          const uint64_t tag = stream_k_tag(w.tile, a - t0);
          while (ld_acquire64(&p_.tags[u]) != tag) {
          }
        }
        named_bar_sync(1, 32 * sm120k::kConsumerWarps);
#pragma unroll
        for (int rb = 0; rb < RB; ++rb)
#pragma unroll
          for (int tf = 0; tf < TF; ++tf) {
            const float4 v = __ldcg(partial_at(u, rb, tf));
            acc[rb][tf][0] += v.x;
            acc[rb][tf][1] += v.y;
            acc[rb][tf][2] += v.z;
            acc[rb][tf][3] += v.w;
          }
      }
    }

    const int wn = cw % C::WN, wm = cw / C::WN;
    const int n0 = w.nt * C::BN + wn * (C::BN / C::WN);  // first weight row of the warp
    const int m0 = w.mt * C::BM + wm * (C::BM / C::WM);  // first token of the warp
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      if (rb >= valid_rb) break;
      const int row = n0 + rb * 16;
      float b_lo = 0.f, b_hi = 0.f;
      if (p_.bias != nullptr) {
        b_lo = __bfloat162float(p_.bias[row + g]);
        b_hi = __bfloat162float(p_.bias[row + 8 + g]);
      }
#pragma unroll
      for (int tf = 0; tf < TF; ++tf) {
        const uint32_t lo = movmatrix_trans(bf16x2_rn(acc[rb][tf][0] + b_lo, acc[rb][tf][1] + b_lo));
        const uint32_t hi = movmatrix_trans(bf16x2_rn(acc[rb][tf][2] + b_hi, acc[rb][tf][3] + b_hi));
        const int tok = m0 + tf * 8 + g;
        if (tok < p_.m) {
          uint32_t* dst = reinterpret_cast<uint32_t*>(p_.y + static_cast<int64_t>(tok) * p_.ldy + row + 2 * t);
          dst[0] = lo;
          dst[4] = hi;
        }
      }
    }
  }

 private:
  // kSplit keeps all grid CTAs resident (grid <= SMs). Each tile's CTAs
  // publish partials, wait for the group and reduce equal output slices in
  // ascending CTA order. This avoids a serial owner reduction.
  template <class C, int RB, int TF>
  __device__ void split_epilogue(float (&acc)[RB][TF][4], const Work& w, int cw, int lane) const {
    using namespace sm120;
    constexpr int kPartial = C::BN * C::BM;
    constexpr int kF4 = kPartial / 4;
    constexpr int kThreadsC = 32 * sm120k::kConsumerWarps;
    float4* const mine = reinterpret_cast<float4*>(p_.partials + static_cast<int64_t>(unit_) * kPartial);
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int tf = 0; tf < TF; ++tf)
        __stcg(mine + ((cw * RB + rb) * TF + tf) * 32 + lane,
               make_float4(acc[rb][tf][0], acc[rb][tf][1], acc[rb][tf][2], acc[rb][tf][3]));
    named_bar_sync(1, kThreadsC);
    const int ctas = p_.sk_group_ctas;
    const int u0 = w.tile * ctas, t0 = w.tile * p_.kb;
    if (cw == 0 && lane == 0) {
      __threadfence();
      st_release64(&p_.tags[unit_], stream_k_tag(w.tile, w.kb0));
      for (int u = u0; u < u0 + ctas; ++u) {
        const uint64_t tag = stream_k_tag(w.tile, sk_begin(u) - t0);
        while (ld_acquire64(&p_.tags[u]) != tag) {
        }
      }
    }
    named_bar_sync(1, kThreadsC);
    const int r = unit_ - u0;
    const int i1 = static_cast<int>(static_cast<int64_t>(kF4) * (r + 1) / ctas);
    const int n0 = w.nt * C::BN, m0 = w.mt * C::BM;
    for (int i = static_cast<int>(static_cast<int64_t>(kF4) * r / ctas) + cw * 32 + lane; i < i1; i += kThreadsC) {
      const float4* src = reinterpret_cast<const float4*>(p_.partials + static_cast<int64_t>(u0) * kPartial) + i;
      float4 v = __ldcg(src);
      for (int u = 1; u < ctas; ++u) {
        const float4 q = __ldcg(src + static_cast<int64_t>(u) * kF4);
        v.x += q.x;
        v.y += q.y;
        v.z += q.z;
        v.w += q.w;
      }
      // Index ((cw * RB + rb) * TF + tf) * 32 + lane of the fragment layout.
      const int ln = i % 32, tf = i / 32 % TF, rb = i / (32 * TF) % RB, c = i / (32 * TF * RB);
      const int wn = c % C::WN, wm = c / C::WN;
      const int row = n0 + wn * (C::BN / C::WN) + rb * 16 + ln / 4;
      const int tok = m0 + wm * (C::BM / C::WM) + tf * 8 + 2 * (ln % 4);
      if (row >= p_.n) continue;
      float b_lo = 0.f, b_hi = 0.f;
      if (p_.bias != nullptr) {
        b_lo = __bfloat162float(p_.bias[row]);
        b_hi = __bfloat162float(p_.bias[row + 8]);
      }
      __nv_bfloat16* y = p_.y + static_cast<int64_t>(tok) * p_.ldy + row;
      if (tok < p_.m) {
        y[0] = __float2bfloat16_rn(v.x + b_lo);
        y[8] = __float2bfloat16_rn(v.z + b_hi);
      }
      if (tok + 1 < p_.m) {
        y[p_.ldy] = __float2bfloat16_rn(v.y + b_lo);
        y[p_.ldy + 8] = __float2bfloat16_rn(v.w + b_hi);
      }
    }
  }

  // CTA u's first stream-K k-block. Group g of the CTAs splits the g-th
  // equal slice of the space, all groups at the same offsets: when a slice is
  // one tile, or one m-tile's row of tiles, CTAs of different groups stream
  // the same weight k-blocks at the same time, so each is fetched from DRAM
  // once and shared via L2.
  __device__ int sk_begin(int u) const {
    const int ctas = p_.sk_group_ctas;
    if (u >= p_.sk_groups * ctas) return p_.sk_units;
    const int g = u / ctas, r = u - g * ctas;
    // g * per_group + floor(per_group * r / ctas), without overflow.
    return g * p_.sk_per_group + p_.sk_q * r + p_.sk_rem * r / ctas;
  }

  const Params& p_;
  int unit_, units_, dp_;
  int sk_left_ = 0, sk_tile_ = 0, sk_k0_ = 0;
#endif  // NF12_SM120_DEVICE
};

// Only images with the SM120 body carry static shared memory in the probe.
// Production kernels have none so dynamic shared memory starts 1024-byte aligned.
__global__ void large_m_sm120_probe(int* out) {
#if defined(NF12_SM120_DEVICE)
  __shared__ volatile int marker[32];  // volatile: must survive optimization
  marker[threadIdx.x % 32] = static_cast<int>(threadIdx.x);
  __syncthreads();
  *out = marker[(threadIdx.x + 1) % 32];
#else
  (void)out;
#endif
}


// Forced schedule = config / 16 % 16. kSplit distributes each tile over
// config / 256 CTAs, all resident, to parallelize reduction for narrow weights.
enum class Mode { kHybrid = 0, kDataParallel = 1, kStreamK = 2, kSplit = 3 };
constexpr int kNumModes = 4;

struct Sm120Config {
  int bn, bm, wn, sx;
  const char* name;
  cudaError_t (*launch)(const LinearArgs&, const struct Plan&);
};

struct Plan {
  int config;
  Mode mode;
  int grid;
  int n_tiles, m_tiles, kb;
  int64_t tiles;
  int group_n;
  int dp_tiles;
  int64_t sk_units;
  int sk_groups, sk_group_ctas;
  size_t partial_bytes, tag_bytes;
  size_t workspace() const { return partial_bytes + tag_bytes; }
};

PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) !=
            cudaSuccess ||
        q != cudaDriverEntryPointSuccess)
      return static_cast<PFN_cuTensorMapEncodeTiled_v12000>(nullptr);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }();
  return fn;
}

template <int BN, int BM, int WN, int SX, WeightSource S>
cudaError_t launch(const LinearArgs& a, const Plan& pl) {
  using C = sm120k::Cfg<BN, BM, WN, SX, S, 0>;
  auto kernel = sm120k::large_m_sm120_kernel<BN, BM, WN, SX, S, DenseProblem>;
  if (static_cast<size_t>(C::kSmemBytes) > a.dev->smem_optin_bytes) return cudaErrorNotSupported;
  // Cache successful per-device shared-memory grants, permitting retries on failure.
  static std::atomic<bool> granted[kMaxDevices];
  const int dev = a.dev->device;
  if (dev < 0 || dev >= kMaxDevices || !granted[dev].load(std::memory_order_acquire)) {
    const cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes);
    if (err != cudaSuccess) return err;
    if (dev >= 0 && dev < kMaxDevices) granted[dev].store(true, std::memory_order_release);
  }

  // Tensor-map limits: strides < 2^40 bytes and multiples of 16, 16-byte
  // aligned base addresses.
  constexpr uint64_t kMaxStride = uint64_t(1) << 40;
  if (static_cast<uint64_t>(a.ldx) * 2 >= kMaxStride || static_cast<uint64_t>(pl.kb) * C::SA >= kMaxStride ||
      a.ldx % 8 != 0 || reinterpret_cast<uintptr_t>(a.x) % 16 != 0 || reinterpret_cast<uintptr_t>(a.weight) % 16 != 0)
    return cudaErrorNotSupported;
  auto encode = tensor_map_encoder();
  if (encode == nullptr) return cudaErrorNotSupported;
  CUtensorMap xmap, wmap;
  {
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(a.k), static_cast<cuuint64_t>(a.m)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(a.ldx) * 2};
    const cuuint32_t box[2] = {64, BM};
    const cuuint32_t estr[2] = {1, 1};
    if (encode(&xmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<__nv_bfloat16*>(a.x), dims, strides, box, estr,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }
  {
    // Weights as [16-row block][k super-atom][super-atom bytes as u64]: one
    // TMA per stage fetches the CTA's BN / 16 super-atoms (each block's run is
    // contiguous along K, blocks are strided).
    const cuuint64_t dims[3] = {C::SA / 8, static_cast<cuuint64_t>(pl.kb), static_cast<cuuint64_t>(a.n / 16)};
    const cuuint64_t strides[2] = {C::SA, static_cast<cuuint64_t>(pl.kb) * C::SA};
    const cuuint32_t box[3] = {C::SA / 8, 1, C::kBlocks};
    const cuuint32_t estr[3] = {1, 1, 1};
    if (encode(&wmap, CU_TENSOR_MAP_DATA_TYPE_UINT64, 3, const_cast<uint8_t*>(a.weight), dims, strides, box, estr,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }

  DenseParams p{};
  p.escapes = a.escapes;
  p.num_escapes = a.escapes != nullptr ? a.num_escapes : 0;
  p.base_x4 = exp_base_x4(a.exp_base);
  p.bias = a.bias;
  p.y = a.y;
  p.ldy = a.ldy;
  p.m = static_cast<int>(a.m);
  p.n = static_cast<int>(a.n);
  p.kb = pl.kb;
  p.n_tiles = pl.n_tiles;
  p.m_tiles = pl.m_tiles;
  p.group_n = pl.group_n;
  p.dp_tiles = pl.dp_tiles;
  p.sk_units = static_cast<int>(pl.sk_units);
  p.sk_groups = pl.sk_groups;
  p.sk_group_ctas = pl.sk_group_ctas;
  p.sk_per_group = static_cast<int>(pl.sk_units / pl.sk_groups);
  p.sk_q = p.sk_per_group / pl.sk_group_ctas;
  p.sk_rem = p.sk_per_group % pl.sk_group_ctas;
  p.split = pl.mode == Mode::kSplit && pl.sk_units > 0;
  if (pl.sk_units > 0) {
    p.partials = static_cast<float*>(a.workspace);
    p.tags = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(a.workspace) + pl.partial_bytes);
    const cudaError_t err = cudaMemsetAsync(p.tags, 0, pl.tag_bytes, a.stream);
    if (err != cudaSuccess) return err;
  }

  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(pl.grid);
  lc.blockDim = dim3(kThreads);
  lc.dynamicSmemBytes = C::kSmemBytes;
  lc.stream = a.stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = attr;
  lc.numAttrs = 1;
  return cudaLaunchKernelEx(&lc, kernel, xmap, wmap, p);
}

template <int BN, int BM, int WN, int SX>
cudaError_t launch_src(const LinearArgs& a, const Plan& pl) {
  return a.source == WeightSource::kNF12 ? launch<BN, BM, WN, SX, WeightSource::kNF12>(a, pl)
                                         : launch<BN, BM, WN, SX, WeightSource::kBF16T>(a, pl);
}

#define NF12_SM120_CONFIG(BN, BM, WN, SX) \
  Sm120Config { BN, BM, WN, SX, "w" #BN "_t" #BM "_wn" #WN "_sx" #SX, &launch_src<BN, BM, WN, SX> }

// BN weight rows x BM tokens, WN consumer warps along rows and SX x stages.
constexpr Sm120Config kConfigs[] = {
    NF12_SM120_CONFIG(256, 128, 8, 2),  // 0: warp tiles 32 x 128
    NF12_SM120_CONFIG(128, 128, 8, 3),  // 1: 16 x 128
    NF12_SM120_CONFIG(128, 192, 8, 2),  // 2: 16 x 192
    NF12_SM120_CONFIG(256, 64, 8, 3),   // 3: 32 x 64
    NF12_SM120_CONFIG(128, 64, 8, 4),   // 4: 16 x 64
    NF12_SM120_CONFIG(64, 64, 4, 4),    // 5: 16 x 32
    NF12_SM120_CONFIG(64, 128, 4, 3),   // 6: 16 x 64
    NF12_SM120_CONFIG(128, 256, 8, 2),  // 7: 16 x 256
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

bool valid_config(int config) { return config < 0 || (config % 16 < kNumConfigs && config / 16 % 16 < kNumModes); }

// More tokens per tile amortize decoding, while narrow weights need smaller
// token tiles to retain grid parallelism.
int heuristic_config(int64_t m, int64_t n, int64_t k, const DeviceInfo&) {
  enum { kW256T128 = 0, kW128T128 = 1, kW128T192 = 2, kW128T64 = 4, kW128T256 = 7 };
  if (m <= 192) return n <= 4096 ? kW128T64 : m <= 128 ? kW128T128 : kW128T192;
  if (m <= 256) return kW128T128;
  if (m <= 384) return kW128T192;
  if (m <= 512) return n * k <= 4096 * 4096 ? kW128T128 : kW256T128;
  return kW128T256;
}

struct Choice {
  int config;
  Mode mode;
  int splits = 0;
};

// Prefer hybrid scheduling unless stream-K gives fewer than 24 k-blocks
// per CTA and a data-parallel plan models faster. The model includes stage
// overhead, MMA throughput, SM-dependent bandwidth and partial reductions.
// kSplit parallelizes reduction when too few tiles would overload one owner.
Choice choose(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  const double units = std::max(1, dev.num_sms), kb = static_cast<double>(k / 64);
  const double mem_us = 1.5 * static_cast<double>(n) * static_cast<double>(k) / 1.4e6;
  auto stage_us = [](const Sm120Config& c) { return 0.33 + 4.3e-5 * c.bn * c.bm; };
  auto tiles = [&](const Sm120Config& c) {
    return static_cast<double>((n + c.bn - 1) / c.bn) * static_cast<double>((m + c.bm - 1) / c.bm);
  };
  const Choice hybrid{heuristic_config(m, n, k, dev), Mode::kHybrid};
  const Sm120Config& h = kConfigs[hybrid.config];
  const double per_cta = tiles(h) * kb / units;
  if (per_cta >= 24.0) return hybrid;
  Choice best = hybrid;
  const double contributors = std::floor(units / tiles(h));
  const double fixup_us = std::max(0.0, contributors - 4.0) * h.bn * h.bm * 4.0 / 50e3;
  double best_t = std::max(mem_us, per_cta * stage_us(h)) + 6.0 + fixup_us;
  // More tokens per equal-area tile amortize decode work and break model ties.
  for (int c = 0; c < kNumConfigs; ++c) {
    const Sm120Config& t = kConfigs[c];
    const double waves = std::ceil(tiles(t) / units);
    const double busy = tiles(t) / (waves * units);
    const double us = std::max(mem_us / (0.55 + 0.45 * busy), waves * kb * stage_us(t));
    const bool tie = std::abs(us - best_t) <= 1e-9 * best_t;
    if (us < best_t && !tie) {
      best_t = us;
      best = Choice{c, Mode::kDataParallel};
    } else if (tie && best.mode == Mode::kDataParallel && t.bm > kConfigs[best.config].bm) {
      best = Choice{c, Mode::kDataParallel};
    }
  }
  for (int c = 0; c < kNumConfigs; ++c) {
    const Sm120Config& t = kConfigs[c];
    const int max_s = static_cast<int>(std::min(std::floor(units / tiles(t)), kb));
    for (int s = 2; s <= max_s; ++s) {
      const double us = std::max(mem_us, std::ceil(kb / s) * stage_us(t)) + 8.0 + 0.25 * s;
      if (us < best_t) {
        best_t = us;
        best = Choice{c, Mode::kSplit, s};
      }
    }
  }
  return best;
}

// Fraction of the ideal (perfectly balanced) time that whole data-parallel
// waves plus an aligned split of the remaining tiles over units / remaining
// CTAs each achieve.
double aligned_tail_efficiency(int64_t tiles, int units) {
  const int64_t waves = tiles / units, rest = tiles % units;
  const double tail = rest > 0 ? 1.0 / std::max<int64_t>(1, units / rest) : 0.0;
  return static_cast<double>(tiles) / units / (waves + tail);
}

Plan make_plan(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config) {
  Plan pl{};
  int splits = 0;
  if (config < 0) {
    const Choice ch = choose(m, n, k, dev);
    pl.config = ch.config;
    pl.mode = ch.mode;
    splits = ch.splits;
  } else {
    pl.config = config % 16;
    pl.mode = static_cast<Mode>(config / 16 % 16);
    splits = config / 256;
  }
  const Sm120Config& c = kConfigs[pl.config];
  pl.n_tiles = static_cast<int>((n + c.bn - 1) / c.bn);
  pl.m_tiles = static_cast<int>((m + c.bm - 1) / c.bm);
  pl.tiles = static_cast<int64_t>(pl.n_tiles) * pl.m_tiles;
  pl.kb = static_cast<int>(k / 64);
  const int units = std::max(1, dev.num_sms);
  // Groups fitting one third of L2 retain weight panels while x streams.
  // Otherwise minimize concurrent panel traffic at
  // group_n = sqrt(units * x_panel / w_panel).
  const int64_t w_panel = static_cast<int64_t>(c.bn) * k * 3 / 2;
  const int64_t x_panel = static_cast<int64_t>(c.bm) * k * 2;
  const int64_t budget = static_cast<int64_t>(dev.l2_bytes ? dev.l2_bytes : (32 << 20)) / 3;
  const int64_t balanced = std::llround(std::sqrt(static_cast<double>(units) * x_panel / w_panel));
  pl.group_n = static_cast<int>(std::clamp<int64_t>(std::max(budget / w_panel, balanced), 1, pl.n_tiles));
  Mode mode = pl.mode;
  if (mode == Mode::kHybrid && pl.tiles % units == 0) mode = Mode::kDataParallel;
  pl.grid = static_cast<int>(std::min<int64_t>(pl.tiles, units));
  pl.dp_tiles = static_cast<int>(pl.tiles);
  pl.sk_units = 0;
  pl.sk_groups = 1;
  pl.sk_group_ctas = units;
  // Small grids use stream-K groups aligned to m-tiles when possible, losing
  // at most m_tiles - 1 CTAs. Larger grids keep whole waves data-parallel
  // and split the tail. Long K favors tile-aligned splits to reuse weights
  // in L2, unless the alignment would leave too many CTAs idle.
  if (mode == Mode::kSplit) {
    splits = static_cast<int>(std::min<int64_t>({std::max(splits, 1), units / pl.tiles, pl.kb}));
    if (splits > 1) {
      pl.grid = static_cast<int>(pl.tiles) * splits;
      pl.dp_tiles = 0;
      pl.sk_units = pl.tiles * pl.kb;
      pl.sk_groups = static_cast<int>(pl.tiles);
      pl.sk_group_ctas = splits;
    }
  } else if (mode == Mode::kStreamK || (mode == Mode::kHybrid && (pl.m_tiles <= 8 || pl.tiles < 2 * units))) {
    pl.grid = units;
    pl.dp_tiles = 0;
    pl.sk_units = pl.tiles * pl.kb;
    if (pl.m_tiles <= units / 2) {
      pl.group_n = pl.n_tiles;
      pl.sk_groups = pl.m_tiles;
      pl.sk_group_ctas = units / pl.m_tiles;
    }
  } else if (mode == Mode::kHybrid && pl.kb >= 256 && aligned_tail_efficiency(pl.tiles, units) >= 0.93) {
    pl.grid = units;
    pl.dp_tiles = static_cast<int>(pl.tiles / units * units);
    const int rest = static_cast<int>(pl.tiles - pl.dp_tiles);
    pl.sk_units = static_cast<int64_t>(rest) * pl.kb;
    pl.sk_groups = rest;
    pl.sk_group_ctas = std::min(units / rest, pl.kb);
  } else if (mode == Mode::kHybrid) {
    pl.grid = units;
    pl.dp_tiles = static_cast<int>(std::max<int64_t>(0, pl.tiles / units - 1) * units);
    pl.sk_units = (pl.tiles - pl.dp_tiles) * pl.kb;
  }
  if (pl.sk_units > INT32_MAX) {  // the kernel's stream-K arithmetic is 32-bit
    pl.grid = static_cast<int>(std::min<int64_t>(pl.tiles, units));
    pl.dp_tiles = static_cast<int>(pl.tiles);
    pl.sk_units = 0;
    pl.sk_groups = 1;
    pl.sk_group_ctas = units;
  }
  if (pl.sk_units > 0) {
    pl.partial_bytes = static_cast<size_t>(pl.grid) * c.bn * c.bm * sizeof(float);
    pl.tag_bytes = (static_cast<size_t>(pl.grid) * sizeof(uint64_t) + 255) & ~size_t(255);
  }
  return pl;
}

}  // namespace

bool large_m_sm120_supported(const DeviceInfo& dev) {
  if (dev.sm_arch != 120 && dev.sm_arch != 121) return false;
  // The arch-specific image must be the one loaded for this device: a
  // baseline sm_120 cubin or JIT-compiled generic PTX matches binaryVersion
  // but carries only the trapping body, whose probe has no static shared
  // memory. Cached per device: the probe sits on the launch path.
  static std::atomic<int8_t> cache[kMaxDevices] = {};  // 0 unknown, 1 yes, -1 no
  const bool cacheable = dev.device >= 0 && dev.device < kMaxDevices;
  if (cacheable) {
    const int8_t c = cache[dev.device].load(std::memory_order_relaxed);
    if (c != 0) return c > 0;
  }
  cudaFuncAttributes attr{};
  bool ok = false;
  if (cudaFuncGetAttributes(&attr, large_m_sm120_probe) == cudaSuccess)
    ok = attr.binaryVersion == dev.sm_arch && attr.sharedSizeBytes > 0;
  else
    cudaGetLastError();  // clear the non-sticky lookup error we caused
  if (cacheable) cache[dev.device].store(ok ? 1 : -1, std::memory_order_relaxed);
  return ok;
}

int large_m_sm120_num_configs() { return kNumConfigs; }

int large_m_sm120_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  const Choice c = choose(m, n, k, dev);
  return c.config + 16 * static_cast<int>(c.mode) + 256 * c.splits;
}

const char* large_m_sm120_config_name(int config) {
  return config >= 0 && config < kNumConfigs ? kConfigs[config].name : "invalid";
}

size_t linear_large_m_sm120_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config) {
  if (!valid_config(config)) return 0;
  return make_plan(m, n, k, dev, config).workspace();
}

cudaError_t linear_large_m_sm120(const LinearArgs& a, int config) {
  if (!valid_config(config)) return cudaErrorInvalidValue;
  if (a.dev == nullptr || !large_m_sm120_supported(*a.dev)) return cudaErrorNotSupported;
  if (a.m < 1) return cudaErrorInvalidValue;
  // Kernel row/token/k-block indices and TMA coordinates are int32.
  if (a.m > INT32_MAX || a.n > INT32_MAX || a.k > INT32_MAX) return cudaErrorNotSupported;
  const Plan pl = make_plan(a.m, a.n, a.k, *a.dev, config);
  if (pl.tiles > INT32_MAX / 2) return cudaErrorNotSupported;
  if (a.workspace_bytes < pl.workspace() || (pl.workspace() > 0 && a.workspace == nullptr))
    return cudaErrorInvalidValue;
  return kConfigs[pl.config].launch(a, pl);
}

}  // namespace nf12::detail
