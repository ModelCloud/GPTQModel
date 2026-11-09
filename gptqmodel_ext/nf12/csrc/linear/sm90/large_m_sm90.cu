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
// Dense SM90 GEMM uses register-A or cooperative shared-A decode.
// Scheduling units are CTAs or clusters whose ranks take consecutive m-tiles,
// each with a partial slot. Tiles [0, dp_tiles) are data-parallel. Remaining
// k-blocks form contiguous stream-K ranges. A split tile's k = 0 segment is
// its owner's last segment, while other contributors process theirs first
// and publish FP32 partials followed by tags. Tags are zeroed before every
// launch so stale workspace cannot satisfy a wait. Reduction follows
// ascending unit order, fixed by shape, config and grid.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>

#include "linear/sm90/large_m_sm90.h"
#include "linear/sm90/large_m_sm90_cluster_kernel.cuh"
#include "linear/sm90/large_m_sm90_kernel.cuh"
#include "nf12/format.cuh"

namespace nf12::detail {
namespace {

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
  // equal slices across sk_group_ctas units each. sk_q/sk_rem hold quotient
  // and remainder to avoid device-side 64-bit division.
  int sk_units, sk_groups, sk_group_ctas, sk_per_group, sk_q, sk_rem;
  float* partials;   // [grid][BN * BM] FP32, stream-K contributors
  uint64_t* tags;    // [grid]: publication tags, zeroed by the host before every launch
  // Cluster split-K (> 1): each cluster of ksplit CTAs runs one tile, rank r
  // its r-th equal share of the k-blocks, and the ranks reduce their FP32
  // partials through distributed shared memory in rank order.
  int ksplit;
};

#if defined(NF12_SM90_DEVICE)
// Nonzero tag for (tile, first k-block), published after the partial. Owners
// wait for an exact match. The host zeroes all slots on the launch stream,
// including graph capture, so stale workspace tags cannot satisfy a wait.
__device__ __forceinline__ uint64_t stream_k_tag(int tile, int kb0) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(tile) + 1u) << 32) | static_cast<uint32_t>(kb0);
}
#endif

template <bool kClusterSplit>
class DenseProblem {
 public:
  using Params = DenseParams;
  struct Work {
    int tile, kb0, kb1;
    int nt, mt;
  };
  static constexpr int kExtraBytes = 0;

#if defined(NF12_SM90_DEVICE)
  static constexpr bool kXWarp = false;
  enum : int { kXWarps = 1 };  // only the register-A kernel uses producer warps
  static constexpr bool kXTensorMap = true;
  static constexpr int kXArrivals = 1;  // lane 0's expect_tx

  __device__ static void init(uint32_t) {}

  // Units are clusters (single CTAs without a cluster launch): every CTA of a
  // cluster runs its cluster's work list, on m-tile mt * cluster + rank of
  // the cluster's m-tile mt.
  __device__ DenseProblem(const Params& p, uint8_t*, uint32_t)
      : p_(p),
        unit_(static_cast<int>(sm90::cluster_id_x())),
        units_(static_cast<int>(sm90::nclusters_x())),
        rank_(static_cast<int>(sm90::cluster_ctarank())),
        cl_(static_cast<int>(sm90::cluster_nctarank())) {
    if (kClusterSplit && p_.ksplit > 1) {
      // The cluster's CTAs share one m-tile and split K instead.
      kr_ = rank_;
      rank_ = 0;
      cl_ = 1;
    }
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
    if (kClusterSplit && p_.ksplit > 1) {
      if (dp_ != unit_) return false;  // one tile per cluster
      w.tile = dp_;
      w.kb0 = kr_ * p_.kb / p_.ksplit;
      w.kb1 = (kr_ + 1) * p_.kb / p_.ksplit;
      ++dp_;
    } else if (dp_ < p_.dp_tiles) {
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
    w.mt = w.mt * cl_ + rank_;
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
                         int t) const {
    if (t != 0) return;  // one TMA, from the first caller
    sm90::mbar_arrive_expect_tx(bar, C::kXBytes);  // rows past m are zero-filled and counted
    sm90::tma_load_2d(dst, xmap, kb * 64, w.mt * C::BM, bar, policy);
  }
  __device__ uint32_t base_x4(const Work&) const { return p_.base_x4; }
  __device__ const uint4* escapes(const Work&) const { return p_.escapes; }
  __device__ uint32_t num_escapes(const Work&) const { return p_.num_escapes; }
  __device__ int n() const { return p_.n; }

  // Finish whole tiles or reduce stream-K partials, then add bias in FP32.
  // Transpose each 8x8 fragment to token g, rows 2t/2t+1 for row-major stores.
  template <class C, int RB, int TF>
  __device__ void epilogue(float (&acc)[RB][TF][4], const Work& w, int cw, int lane, int valid_rb,
                           uint8_t* = nullptr) const {
    using namespace sm90;
    constexpr int kPartial = C::BN * C::BM;  // floats per CTA slot
    // Cluster split-K: which of the warp's fragments this rank stores.
    uint32_t owned = ~0u;
    if (kClusterSplit && p_.ksplit > 1) {
      static_assert(RB * TF <= 32, "fragment mask");
      // Fragment f belongs to rank f % ks, which receives every rank's copy
      // at slot (q, f / ks) in its operand stages. All cluster wgmmas and
      // TMAs must finish before those stages become reduction scratch.
      // Remote stores let the ordered sum read local shared memory.
      constexpr int kFrags = sm90k::kConsumerWarps * RB * TF;
      const int ks = p_.ksplit;
      const int per_rank = (kFrags + ks - 1) / ks;
      extern __shared__ __align__(1024) uint8_t dyn_smem[];
      const uint32_t part = smem_u32(dyn_smem);
      static_assert((kFrags + 8) * 512 <= C::kSmemBytes, "every rank's share fits the stages");
      auto slot = [&](int q, int f) { return part + ((q * per_rank + f / ks) * 32 + lane) * 16; };
      cluster_sync();
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int tf = 0; tf < TF; ++tf) {
          const int f = (cw * RB + rb) * TF + tf;
          st_cluster128(mapa(slot(kr_, f), static_cast<uint32_t>(f % ks)), acc[rb][tf]);
        }
      cluster_sync();
      // Sums over ranks 0..ks-1 in order.
      owned = 0;
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int tf = 0; tf < TF; ++tf) {
          const int f = (cw * RB + rb) * TF + tf;
          if (f % ks != kr_) continue;
          owned |= 1u << (rb * TF + tf);
          float s[4] = {0.f, 0.f, 0.f, 0.f};
          for (int q = 0; q < ks; ++q) {
            const uint4 v = lds128(slot(q, f));
            s[0] += __uint_as_float(v.x);
            s[1] += __uint_as_float(v.y);
            s[2] += __uint_as_float(v.z);
            s[3] += __uint_as_float(v.w);
          }
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[rb][tf][c] = s[c];
        }
    }
    // A CTA's slot: its rank in the slots of its unit.
    auto partial_at = [&](int unit, int rb, int tf) {
      return reinterpret_cast<float4*>(p_.partials + static_cast<int64_t>(unit * cl_ + rank_) * kPartial) +
             ((cw * RB + rb) * TF + tf) * 32 + lane;
    };
    if ((!kClusterSplit || p_.ksplit <= 1) && w.kb0 != 0) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int tf = 0; tf < TF; ++tf)
          __stcg(partial_at(unit_, rb, tf), make_float4(acc[rb][tf][0], acc[rb][tf][1], acc[rb][tf][2],
                                                        acc[rb][tf][3]));
      named_bar_sync(1, 32 * sm90k::kConsumerWarps);
      if (cw == 0 && lane == 0) {
        __threadfence();
        st_release64(&p_.tags[unit_ * cl_ + rank_], stream_k_tag(w.tile, w.kb0));
      }
      return;
    }
    if ((!kClusterSplit || p_.ksplit <= 1) && w.kb1 != p_.kb) {
      // Contributors are the following CTAs whose range starts inside this tile.
      const int t0 = (w.tile - p_.dp_tiles) * p_.kb;
      for (int u = unit_ + 1; u < units_; ++u) {
        const int a = sk_begin(u);
        if (a >= t0 + p_.kb) break;
        if (a == sk_begin(u + 1)) continue;  // empty range
        if (cw == 0 && lane == 0) {
          const uint64_t tag = stream_k_tag(w.tile, a - t0);
          while (ld_acquire64(&p_.tags[u * cl_ + rank_]) != tag) {
          }
        }
        named_bar_sync(1, 32 * sm90k::kConsumerWarps);
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
        if (!((owned >> (rb * TF + tf)) & 1u)) continue;
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
  int unit_, units_, rank_, cl_, dp_;
  int kr_ = 0;  // cluster split-K rank
  int sk_left_ = 0, sk_tile_ = 0, sk_k0_ = 0;
#endif  // NF12_SM90_DEVICE
};

// Only images with the SM90 body carry static shared memory in the probe.
// Production kernels have none so dynamic shared memory starts 1024-byte aligned.
__global__ void large_m_sm90_probe(int* out) {
#if defined(NF12_SM90_DEVICE)
  __shared__ volatile int marker[32];  // volatile: must survive optimization
  marker[threadIdx.x % 32] = static_cast<int>(threadIdx.x);
  __syncthreads();
  *out = marker[(threadIdx.x + 1) % 32];
#else
  (void)out;
#endif
}


// Forced schedule = config / 16 % 16. kSplit distributes each tile across
// config / 256 units. kClusterSplit instead uses that many CTAs per cluster
// and reduces through DSMEM without workspace or tag reset.
enum class Mode { kHybrid = 0, kDataParallel = 1, kStreamK = 2, kSplit = 3, kClusterSplit = 4 };
constexpr int kNumModes = 5;

struct Sm90Config {
  int bn, bm, sx;
  int cl;  // CTAs per scheduling unit; shared-A also has single-CTA cooperative configs
  const char* name;
  cudaError_t (*launch)(const LinearArgs&, const struct Plan&);
  // Units (clusters, or CTAs) that run concurrently, 0 if unknown.
  int (*units)(const DeviceInfo&);
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
  int ksplit;  // CTAs per cluster splitting K (kClusterSplit), else 1
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

// The two kernels: K::C its Cfg, K::kCluster CTAs per cluster, K::kernel.
template <int BN, int BM, int SX, WeightSource S>
struct RegisterA {
  static constexpr int kWK = 2;
  using C = sm90k::Cfg<BN, BM, SX, S, 0, kWK>;
  static constexpr int kCluster = 1;
  static constexpr int kSuper = C::SA;
  static constexpr int kBoxBlocks = C::kBlocks;
  static constexpr auto kernel = sm90k::large_m_sm90_kernel<BN, BM, SX, S, DenseProblem<true>, kWK>;
};
template <int BM, int CL, int ST, WeightSource S>
struct SharedA {
  using C = sm90k::ClusterCfg<BM, CL, ST, S, 0>;
  static constexpr int kCluster = CL;
  static constexpr int kSuper = C::kSuper;
  static constexpr int kBoxBlocks = CL == 1 ? C::kBlocks / sm90k::kDecoders : C::kShareBlocks;
  static constexpr int kWK = 1;
  // This family shares decoded weights across token tiles, never across K.
  // Exclude the cluster split-K policy from its register allocation and epilogue.
  static constexpr auto kernel = sm90k::large_m_sm90_cluster_kernel<BM, CL, ST, S, DenseProblem<false>>;
};

// Dynamic shared memory is a per-device function attribute. Cache success
// only, so failed grants can be retried.
template <class K>
cudaError_t grant_smem(const DeviceInfo& d) {
  if (static_cast<size_t>(K::C::kSmemBytes) > d.smem_optin_bytes) return cudaErrorNotSupported;
  static std::atomic<bool> granted[kMaxDevices];
  const int dev = d.device;
  if (dev < 0 || dev >= kMaxDevices || !granted[dev].load(std::memory_order_acquire)) {
    const cudaError_t err =
        cudaFuncSetAttribute(K::kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, K::C::kSmemBytes);
    if (err != cudaSuccess) return err;
    if (dev >= 0 && dev < kMaxDevices) granted[dev].store(true, std::memory_order_release);
  }
  return cudaSuccess;
}

int sm_units(const DeviceInfo& d) { return std::max(1, d.num_sms); }

// Cache co-resident cluster counts per device, accounting for GPC packing.
// Return 0 when the query fails.
template <class K>
int cluster_units(const DeviceInfo& d) {
  static std::atomic<int> cache[kMaxDevices] = {};  // 0 unknown, -1 failed
  const bool cacheable = d.device >= 0 && d.device < kMaxDevices;
  if (cacheable) {
    const int c = cache[d.device].load(std::memory_order_relaxed);
    if (c != 0) return std::max(c, 0);
  }
  int n = 0;
  if (grant_smem<K>(d) == cudaSuccess) {
    cudaLaunchConfig_t lc{};
    lc.gridDim = dim3(K::kCluster * std::max(1, d.num_sms / K::kCluster));
    lc.blockDim = dim3(sm90k::kThreads);
    lc.dynamicSmemBytes = K::C::kSmemBytes;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = K::kCluster;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    lc.attrs = attr;
    lc.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&n, K::kernel, &lc) != cudaSuccess) {
      cudaGetLastError();  // clear the non-sticky error we caused
      n = 0;
    }
  } else {
    cudaGetLastError();
  }
  if (cacheable) cache[d.device].store(n > 0 ? n : -1, std::memory_order_relaxed);
  return n;
}

template <class K>
cudaError_t launch(const LinearArgs& a, const Plan& pl) {
  using C = typename K::C;
  if (const cudaError_t err = grant_smem<K>(*a.dev); err != cudaSuccess) return err;

  // Tensor-map limits: strides < 2^40 bytes and multiples of 16, 16-byte
  // aligned base addresses.
  constexpr uint64_t kMaxStride = uint64_t(1) << 40;
  if (static_cast<uint64_t>(a.ldx) * 2 >= kMaxStride || static_cast<uint64_t>(pl.kb) * K::kSuper >= kMaxStride ||
      a.ldx % 8 != 0 || reinterpret_cast<uintptr_t>(a.x) % 16 != 0 || reinterpret_cast<uintptr_t>(a.weight) % 16 != 0)
    return cudaErrorNotSupported;
  auto encode = tensor_map_encoder();
  if (encode == nullptr) return cudaErrorNotSupported;
  CUtensorMap xmap, wmap;
  {
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(a.k), static_cast<cuuint64_t>(a.m)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(a.ldx) * 2};
    const cuuint32_t box[2] = {64, C::BM};
    const cuuint32_t estr[2] = {1, 1};
    if (encode(&xmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<__nv_bfloat16*>(a.x), dims, strides, box, estr,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }
  {
    // Weights as [16-row block][k super-atom][super-atom bytes as u64]: one
    // TMA per stage fetches kWK consecutive super-atoms of each of the CTA's
    // kBoxBlocks row blocks (each block's run is contiguous along K, blocks
    // are strided).
    const cuuint64_t dims[3] = {K::kSuper / 8, static_cast<cuuint64_t>(pl.kb), static_cast<cuuint64_t>(a.n / 16)};
    const cuuint64_t strides[2] = {K::kSuper, static_cast<cuuint64_t>(pl.kb) * K::kSuper};
    const cuuint32_t box[3] = {K::kSuper / 8, K::kWK, K::kBoxBlocks};
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
  p.ksplit = pl.ksplit;
  if (pl.sk_units > 0) {
    p.partials = static_cast<float*>(a.workspace);
    p.tags = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(a.workspace) + pl.partial_bytes);
    const cudaError_t err = cudaMemsetAsync(p.tags, 0, pl.tag_bytes, a.stream);
    if (err != cudaSuccess) return err;
  }

  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(pl.grid * K::kCluster);
  lc.blockDim = dim3(sm90k::kThreads);
  lc.dynamicSmemBytes = C::kSmemBytes;
  lc.stream = a.stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  const int cluster = K::kCluster * pl.ksplit;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = cluster;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  lc.attrs = attr;
  lc.numAttrs = cluster > 1 ? 2 : 1;
  return cudaLaunchKernelEx(&lc, K::kernel, xmap, wmap, p);
}

template <int BN, int BM, int SX>
cudaError_t launch_reg(const LinearArgs& a, const Plan& pl) {
  return a.source == WeightSource::kNF12 ? launch<RegisterA<BN, BM, SX, WeightSource::kNF12>>(a, pl)
                                         : launch<RegisterA<BN, BM, SX, WeightSource::kBF16T>>(a, pl);
}
template <int BM, int CL, int ST>
cudaError_t launch_shared(const LinearArgs& a, const Plan& pl) {
  return a.source == WeightSource::kNF12 ? launch<SharedA<BM, CL, ST, WeightSource::kNF12>>(a, pl)
                                         : launch<SharedA<BM, CL, ST, WeightSource::kBF16T>>(a, pl);
}

#define NF12_SM90_CONFIG(BN, BM, SX)                                    \
  Sm90Config {                                                          \
    BN, BM, SX, 1, "w" #BN "_t" #BM "_sx" #SX, &launch_reg<BN, BM, SX>, \
        &sm_units                                                       \
  }
// The plan (units, hence the reduction order) is the NF12 kernel's for both
// sources: both have one CTA per SM.
#define NF12_SM90_SHARED_CONFIG(BM, CL, ST)                                         \
  Sm90Config {                                                                      \
    128, BM, ST, CL, "w128_t" #BM "_st" #ST "_c" #CL, &launch_shared<BM, CL, ST>,   \
        &cluster_units<SharedA<BM, CL, ST, WeightSource::kNF12>>                    \
  }

// BN weight rows x BM tokens per CTA (BN / 2 rows per consumer warpgroup, BM
// the wgmma N), SX activation stages; shared-decode configs: ST stages, CL
// CTAs per cluster.
constexpr Sm90Config kConfigs[] = {
    NF12_SM90_CONFIG(256, 128, 4),       // 0: two m64n128 wgmmas per warpgroup and k16 step
    NF12_SM90_CONFIG(128, 256, 4),       // 1: one m64n256
    NF12_SM90_CONFIG(128, 128, 4),       // 2
    NF12_SM90_CONFIG(256, 64, 4),        // 3
    NF12_SM90_CONFIG(128, 64, 6),        // 4
    NF12_SM90_CONFIG(128, 192, 4),       // 5
    NF12_SM90_SHARED_CONFIG(256, 2, 4),  // 6: 128 x 256 per CTA, pairs of CTAs share the decode
    NF12_SM90_SHARED_CONFIG(64, 1, 4),   // 7: all four producer warps decode every stage
    NF12_SM90_SHARED_CONFIG(128, 1, 4),  // 8
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);
constexpr int kSharedConfig = 6;
constexpr int kCooperative64Config = 7;
constexpr int kCooperative128Config = 8;

bool valid_config(int config) {
  if (config < 0) return true;
  if (config % 16 >= kNumConfigs || config / 16 % 16 >= kNumModes) return false;
  // Only the register-A family supports cluster split-K. The shared-A
  // producer lifetime and stage-release protocol do not support that epilogue.
  return static_cast<Mode>(config / 16 % 16) != Mode::kClusterSplit || config % 16 < kSharedConfig;
}

// One m64n256 wgmma per step reads fewer A fragments per FLOP than two
// m64n128 instructions. Smaller M needs shorter token tiles for grid coverage.
int heuristic_config(int64_t m, int64_t n, int64_t, const DeviceInfo&) {
  enum { kW256T128 = 0, kW128T256 = 1, kW128T128 = 2, kW256T64 = 3, kW128T64 = 4 };
  if (m <= 64) return n <= 8192 ? kW128T64 : kW256T64;
  if (m <= 128) return n <= 16384 ? kW128T128 : kW256T128;
  return kW128T256;
}

struct Choice {
  int config;
  Mode mode;
  int splits = 1;
};

// Compare data-parallel and hybrid schedules using stage cost, throughput,
// SM-dependent bandwidth and stream-K publication/reduction overhead.
Choice choose(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  // Shared decode amortizes work across each cluster's 512-token tile.
  // Short K and nearly full data-parallel waves do not justify stream-K fixups.
  if (m > 256 && (k >= 2048 || m > 384)) {
    const Sm90Config& c = kConfigs[kSharedConfig];
    const int known = c.units(dev);
    const double units = known > 0 ? known : std::max(1, dev.num_sms / c.cl);
    const int64_t bm = static_cast<int64_t>(c.bm) * c.cl;  // tokens per cluster
    const double tiles = static_cast<double>((n + c.bn - 1) / c.bn) * static_cast<double>((m + bm - 1) / bm);
    const double busy = tiles / (std::ceil(tiles / units) * units);
    return Choice{kSharedConfig, k < 2048 || busy >= 0.95 ? Mode::kDataParallel : Mode::kHybrid};
  }
  // Cooperative decode separates decoding from consumer issue for moderate
  // token counts and weight footprints.
  if (m >= 64 && m <= 128 && n >= 4096 && n <= 8192 && k >= 4096 && k <= 16384 &&
      (m <= 64 || n * k <= (int64_t{32} << 20)))
    return Choice{m <= 64 ? kCooperative64Config : kCooperative128Config, Mode::kHybrid};
  // Small-K single-token-tile shapes cannot amortize stream-K tags and partials.
  if (m <= 128 && n >= 4096 && n <= 8192 && k >= 512 && k <= 2048)
    return Choice{m <= 64 ? 4 : 2, Mode::kClusterSplit, k <= 1024 ? 3 : 2};
  const double units = std::max(1, dev.num_sms), kb = static_cast<double>(k / 64);
  const double mem_us = 1.5 * static_cast<double>(n) * static_cast<double>(k) / 4.3e6;
  auto stage_us = [](const Sm90Config& c) { return 0.2 + 1.9e-5 * c.bn * c.bm; };
  auto tiles = [&](const Sm90Config& c) {
    return static_cast<double>((n + c.bn - 1) / c.bn) * static_cast<double>((m + c.bm - 1) / c.bm);
  };
  const Choice hybrid{heuristic_config(m, n, k, dev), Mode::kHybrid};
  const Sm90Config& h = kConfigs[hybrid.config];
  const double per_cta = tiles(h) * kb / units;
  if (per_cta >= 24.0) return hybrid;
  Choice best = hybrid;
  double best_t = std::max(mem_us, per_cta * stage_us(h)) + 4.0;
  // More tokens per equal-area tile amortize decode work and break model ties.
  for (int c = 0; c < kNumConfigs; ++c) {
    const Sm90Config& t = kConfigs[c];
    if (c >= kSharedConfig) continue;
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
  const Sm90Config& c = kConfigs[pl.config];
  // Plans count units: CTAs, or clusters of c.cl CTAs on c.cl consecutive
  // m-tiles (one m-tile of the plan).
  const int64_t bm = static_cast<int64_t>(c.bm) * c.cl;
  pl.n_tiles = static_cast<int>((n + c.bn - 1) / c.bn);
  pl.m_tiles = static_cast<int>((m + bm - 1) / bm);
  pl.tiles = static_cast<int64_t>(pl.n_tiles) * pl.m_tiles;
  pl.kb = static_cast<int>(k / 64);
  // Stream-K owners wait for other units' partials, which needs all units
  // co-resident (see below).
  const int known = c.units(dev);
  const bool co_resident = known > 0;
  const int units = co_resident ? known : std::max(1, dev.num_sms / c.cl);
  // Groups fitting one third of L2 retain weight panels while x streams.
  // Otherwise minimize concurrent panel traffic at
  // group_n = sqrt(units * x_panel / w_panel).
  const int64_t w_panel = static_cast<int64_t>(c.bn) * k * 3 / 2;
  const int64_t x_panel = bm * k * 2;
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
  pl.ksplit = 1;
  // Small grids use stream-K groups aligned to m-tiles when possible, losing
  // at most m_tiles - 1 units. Larger grids keep whole waves data-parallel
  // and split the tail. Long K favors tile-aligned splits to reuse weights
  // in L2, unless the alignment would leave too many units idle.
  if (mode == Mode::kClusterSplit) {
    // Clusters of up to 8 CTAs (the portable limit), each with k-blocks.
    pl.ksplit = static_cast<int>(std::clamp<int64_t>(splits, 1, std::min(8, pl.kb)));
    pl.grid = static_cast<int>(std::min<int64_t>(pl.tiles * pl.ksplit, INT32_MAX));
    pl.dp_tiles = static_cast<int>(pl.tiles);
  } else if (mode == Mode::kSplit) {
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
  // Stream-K needs int32 arithmetic and co-resident units so owners cannot
  // block unscheduled contributors. Fall back to data-parallel otherwise.
  if (pl.sk_units > INT32_MAX || (pl.sk_units > 0 && !co_resident)) {
    pl.grid = static_cast<int>(std::min<int64_t>(pl.tiles, units));
    pl.dp_tiles = static_cast<int>(pl.tiles);
    pl.sk_units = 0;
    pl.sk_groups = 1;
    pl.sk_group_ctas = units;
  }
  if (pl.sk_units > 0) {
    const size_t ctas = static_cast<size_t>(pl.grid) * c.cl;  // one partial slot and tag per CTA
    pl.partial_bytes = ctas * c.bn * c.bm * sizeof(float);
    pl.tag_bytes = (ctas * sizeof(uint64_t) + 255) & ~size_t(255);
  }
  return pl;
}

}  // namespace

bool large_m_sm90_supported(const DeviceInfo& dev) {
  if (dev.sm_arch != 90) return false;
  // The arch-specific image must be the one loaded for this device: a
  // baseline sm_90 cubin or JIT-compiled generic PTX matches binaryVersion
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
  if (cudaFuncGetAttributes(&attr, large_m_sm90_probe) == cudaSuccess)
    ok = attr.binaryVersion == dev.sm_arch && attr.sharedSizeBytes > 0;
  else
    cudaGetLastError();  // clear the non-sticky lookup error we caused
  if (cacheable) cache[dev.device].store(ok ? 1 : -1, std::memory_order_relaxed);
  return ok;
}

int large_m_sm90_num_configs() { return kNumConfigs; }

int large_m_sm90_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  const Choice c = choose(m, n, k, dev);
  return c.config + 16 * static_cast<int>(c.mode) + (c.mode == Mode::kClusterSplit ? 256 * c.splits : 0);
}

const char* large_m_sm90_config_name(int config) {
  return config >= 0 && config < kNumConfigs ? kConfigs[config].name : "invalid";
}

size_t linear_large_m_sm90_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config) {
  if (!valid_config(config)) return 0;
  return make_plan(m, n, k, dev, config).workspace();
}

cudaError_t linear_large_m_sm90(const LinearArgs& a, int config) {
  if (!valid_config(config)) return cudaErrorInvalidValue;
  if (a.dev == nullptr || !large_m_sm90_supported(*a.dev)) return cudaErrorNotSupported;
  if (a.m < 1) return cudaErrorInvalidValue;
  // Kernel row/token/k-block indices and TMA coordinates are int32.
  if (a.m > INT32_MAX || a.n > INT32_MAX || a.k > INT32_MAX) return cudaErrorNotSupported;
  const Plan pl = make_plan(a.m, a.n, a.k, *a.dev, config);
  if (pl.tiles > INT32_MAX / (kConfigs[pl.config].cl * pl.ksplit)) return cudaErrorNotSupported;
  if (a.workspace_bytes < pl.workspace() || (pl.workspace() > 0 && a.workspace == nullptr))
    return cudaErrorInvalidValue;
  return kConfigs[pl.config].launch(a, pl);
}

}  // namespace nf12::detail
