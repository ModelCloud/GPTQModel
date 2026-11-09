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
// Expert-major tiles use the dense grouped raster to retain panels in L2.
// Each output has one fixed-order accumulation, shared by NF12 and bf16t.
// TMA gather4 and cp.async share each activation stage's mbarrier.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>

#include "linear/sm120/large_m_sm120_kernel.cuh"
#include "moe/common.cuh"
#include "moe/sm120/large_sm120.h"
#include "moe/schedule.cuh"

namespace nf12::detail {
namespace {

constexpr int kMaxDevices = 64;  // per-device host caches

struct MoeSm120Params {
  const __nv_bfloat16* x;
  int64_t ldx;
  const int32_t* gather;
  int64_t gather_div;
  const int32_t* scatter;
  const float* row_scale;
  const int64_t* expert_offsets;
  int64_t max_assignments;
  int num_experts;
  const uint4* escapes;  // null for bf16t, or when no expert has escapes
  const int64_t* escape_offsets;
  const uint8_t* exp_base;
  __nv_bfloat16* y;
  int64_t ldy;
  int n, kb;  // kb = k / 64
  int n_tiles, group_n;
};

struct RowOut {
  int32_t out;
  float scale;
};

template <int BM>
class MoeProblem {
  static constexpr int kSlots = 2;  // producer fills tile i + 1's map during tile i
  static constexpr int kMapBytes = kSlots * BM * static_cast<int>(sizeof(RowOut));
  static constexpr int kInBytes = BM * 4;  // x rows of the producer's current tile
  static constexpr int kSchedBytes = 4 * moe_schedule_smem_ints(kMoeSm120MaxExperts);
  static constexpr int kBarOffset = kMapBytes + kInBytes + kSchedBytes;  // map_full[kSlots], map_empty[kSlots]

 public:
  using Params = MoeSm120Params;
  struct Work {
    int tile, kb0, kb1;
    int nt, mt;
    int e, mtiles;  // expert, its m-tile count
    int row0, rows;  // sorted positions [row0, row0 + rows)
    int slot;       // epilogue map slot
    uint32_t phase;
    uint32_t base_x4, num_escapes;
    const uint4* escapes;
  };
  static constexpr int kExtraBytes = (kBarOffset + 16 * kSlots + 15) / 16 * 16;
  static constexpr bool kXWarp = true;
  static constexpr int kXArrivals = 33;  // lane 0's expect_tx (gather4 bytes) + every lane's cp.async arrive

#if defined(NF12_SM120_DEVICE)
  __device__ static void init(uint32_t extra) {
    for (int s = 0; s < kSlots; ++s) {
      sm120::mbar_init(extra + kBarOffset + 8 * s, 32);                                  // producer lanes
      sm120::mbar_init(extra + kBarOffset + 8 * (kSlots + s), sm120k::kConsumerWarps);  // consumer warps
    }
  }

  // All CTA threads must participate in the schedule scan's barriers.
  __device__ MoeProblem(const Params& p, uint8_t* extra, uint32_t extra_addr)
      : p_(p), maps_(reinterpret_cast<RowOut*>(extra)), in_(reinterpret_cast<int32_t*>(extra + kMapBytes)),
        sched_(reinterpret_cast<int*>(extra + kMapBytes + kInBytes)), bars_(extra_addr + kBarOffset) {
    tiles_ = moe_mtile_prefix(p.expert_offsets, p.num_experts, BM, p.max_assignments, sched_) * p.n_tiles;
  }

  template <class C>
  __device__ bool next(Work& w) {
    static_assert(C::BM == BM, "problem and tile disagree");
    const int tile = static_cast<int>(blockIdx.x) + iter_ * static_cast<int>(gridDim.x);
    if (tile >= tiles_) return false;  // tiles_ < 2^31 - gridDim.x: bounded by the host
    w.slot = iter_ % kSlots;
    w.phase = static_cast<uint32_t>(iter_ / kSlots) & 1u;
    ++iter_;
    const int E = p_.num_experts;
    w.tile = tile;
    w.e = moe_mtile_expert(sched_, E, tile / p_.n_tiles);
    const int first = sched_[w.e];
    w.mtiles = sched_[w.e + 1] - first;
    tile_raster(tile - first * p_.n_tiles, p_.n_tiles, w.mtiles, p_.group_n, w.nt, w.mt);
    moe_mtile_rows(sched_, E, w.e, w.mt, BM, w.row0, w.rows);
    w.kb0 = 0;
    w.kb1 = p_.kb;
    w.base_x4 = p_.exp_base != nullptr ? exp_base_x4(__ldg(p_.exp_base + w.e)) : 0u;
    w.escapes = nullptr;
    w.num_escapes = 0;
    if (p_.escapes != nullptr) {
      const int64_t e0 = __ldg(p_.escape_offsets + w.e);
      w.escapes = p_.escapes + e0;
      w.num_escapes = static_cast<uint32_t>(__ldg(p_.escape_offsets + w.e + 1) - e0);
    }
    return true;
  }

  template <class C>
  __device__ int w_block(const Work& w) const {
    return w.e * (p_.n / 16) + w.nt * C::kBlocks;
  }
  // An expert's weights read by a single m-tile are dead after this pass.
  __device__ bool w_streamed_once(const Work& w) const { return w.mtiles == 1; }

  // The activation producer publishes row indices and waits before reusing
  // an epilogue-map slot.
  template <class C>
  __device__ void x_tile(const Work& w, int lane) {
    for (int r = lane; r < BM; r += 32)
      in_[r] = static_cast<int32_t>(moe_in_row(p_.gather, p_.gather_div, w.row0 + min(r, w.rows - 1)));
    sm120::mbar_wait(bars_ + 8 * (kSlots + w.slot), w.phase ^ 1);  // consumers done with the slot
    RowOut* map = maps_ + w.slot * BM;
    for (int r = lane; r < w.rows; r += 32) {
      const int32_t out = static_cast<int32_t>(moe_out_row(p_.scatter, w.row0 + r));
      map[r] = RowOut{out, p_.row_scale != nullptr ? __ldg(p_.row_scale + out) : 1.f};
    }
    sm120::mbar_arrive(bars_ + 8 * w.slot);  // releases this lane's map writes
    __syncwarp();
  }
  // Split rows between gather4 and cp.async to use both TMA and load/store
  // units. A partial gather4 group repeats the last live row through in_.
  // Padded accumulator columns are independent of live rows and never stored.
  template <class C>
  __device__ void x_load(const CUtensorMap* xmap, uint32_t dst, int kb, const Work& w, uint32_t bar,
                         uint64_t policy, int lane) const {
    const int groups = min(kTmaRows, w.rows + 3) / 4;
    if (lane == 0) sm120::mbar_arrive_expect_tx(bar, groups * 512);
    if (lane < groups)
      sm120::tma_gather4(dst + lane * 512, xmap, kb * 64, in_[4 * lane], in_[4 * lane + 1], in_[4 * lane + 2],
                         in_[4 * lane + 3], bar, policy);
    const int ch = lane & 7;
    const __nv_bfloat16* src = p_.x + kb * 64 + ch * 8;
#pragma unroll 4
    for (int row = kTmaRows + (lane >> 3); row < w.rows; row += 4)
      sm120::cp_async_16_hint(dst + row * 128 + ((ch ^ (row & 7)) << 4), src + in_[row] * p_.ldx, policy);
    sm120::cp_async_mbar_arrive_noinc(bar);
  }

  __device__ uint32_t base_x4(const Work& w) const { return w.base_x4; }
  __device__ const uint4* escapes(const Work& w) const { return w.escapes; }
  __device__ uint32_t num_escapes(const Work& w) const { return w.num_escapes; }
  __device__ int n() const { return p_.n; }
  __device__ const void* zfill_src() const { return p_.y; }

  // Scale in FP32 before BF16 rounding. Transpose 8x8 fragments so each lane
  // holds one token's output pairs.
  template <class C, int RB, int TF>
  __device__ void epilogue(float (&acc)[RB][TF][4], const Work& w, int cw, int lane, int valid_rb) const {
    using namespace sm120;
    mbar_wait(bars_ + 8 * w.slot, w.phase);
    const RowOut* map = maps_ + w.slot * BM;
    const int wn = cw % C::WN, wm = cw / C::WN;
    const int n0 = w.nt * C::BN + wn * (C::BN / C::WN);  // first weight row of the warp
    const int t0 = wm * (BM / C::WM);                    // first tile row of the warp
    const int g = lane >> 2, t = lane & 3;
    const bool scale = p_.row_scale != nullptr;
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      if (rb >= valid_rb) break;
      const int col = n0 + rb * 16 + 2 * t;
#pragma unroll
      for (int tf = 0; tf < TF; ++tf) {
        float v[4] = {acc[rb][tf][0], acc[rb][tf][1], acc[rb][tf][2], acc[rb][tf][3]};
        if (scale) {
          // Padded rows may read stale scales because they are never stored.
          const float s0 = map[t0 + tf * 8 + 2 * t].scale, s1 = map[t0 + tf * 8 + 2 * t + 1].scale;
          v[0] *= s0;
          v[1] *= s1;
          v[2] *= s0;
          v[3] *= s1;
        }
        const uint32_t lo = movmatrix_trans(bf16x2_rn(v[0], v[1]));
        const uint32_t hi = movmatrix_trans(bf16x2_rn(v[2], v[3]));
        const int r = t0 + tf * 8 + g;
        if (r < w.rows) {
          uint32_t* dst = reinterpret_cast<uint32_t*>(p_.y + static_cast<int64_t>(map[r].out) * p_.ldy + col);
          dst[0] = lo;
          dst[4] = hi;
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(bars_ + 8 * (kSlots + w.slot));
  }

 private:
  static constexpr int kTmaRows = BM / 2 < 128 ? BM / 2 : 128;
  static_assert(kTmaRows % 4 == 0, "gather4 rows: four per lane");
  const Params& p_;
  RowOut* maps_;
  int32_t* in_;
  int* sched_;
  uint32_t bars_;
  int tiles_ = 0, iter_ = 0;
#endif  // NF12_SM120_DEVICE
};

// The probe's static shared memory identifies images carrying the SM120 body.
// Production kernels keep none so dynamic shared memory starts 1024-byte aligned.
__global__ void moe_large_sm120_probe(int* out) {
#if defined(NF12_SM120_DEVICE)
  __shared__ volatile int marker[32];  // volatile: must survive optimization
  marker[threadIdx.x % 32] = static_cast<int>(threadIdx.x);
  __syncthreads();
  *out = marker[(threadIdx.x + 1) % 32];
#else
  (void)out;
#endif
}

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
cudaError_t launch(const MoeLinearArgs& a) {
  using Prob = MoeProblem<BM>;
  using C = sm120k::Cfg<BN, BM, WN, SX, S, Prob::kExtraBytes>;
  auto kernel = sm120k::large_m_sm120_kernel<BN, BM, WN, SX, S, Prob>;
  if (static_cast<size_t>(C::kSmemBytes) > a.dev->smem_optin_bytes) return cudaErrorNotSupported;
  const int64_t n_tiles = (a.n + BN - 1) / BN;
  const int64_t bound = moe_max_mtiles(a.max_assignments, a.num_experts, BM) * n_tiles;
  if (bound == 0) return cudaSuccess;
  const int units = std::max(1, a.dev->num_sms);
  // Tile indices include a final grid stride. Rows, k-blocks and TMA
  // coordinates are int32. Byte offsets are 64-bit.
  if (bound > INT32_MAX - units || a.k > INT32_MAX || a.num_experts * (a.n / 16) > INT32_MAX)
    return cudaErrorNotSupported;
  // Dynamic shared-memory limits are per-device attributes. Cache only success.
  static std::atomic<bool> granted[kMaxDevices];
  const int dev = a.dev->device;
  if (dev < 0 || dev >= kMaxDevices || !granted[dev].load(std::memory_order_acquire)) {
    const cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes);
    if (err != cudaSuccess) return err;
    if (dev >= 0 && dev < kMaxDevices) granted[dev].store(true, std::memory_order_release);
  }

  // Tensor-map strides must be < 2^40 bytes and multiples of 16.
  // Dispatch validates 16-byte base alignment.
  const int64_t kb = a.k / 64;
  constexpr uint64_t kMaxStride = uint64_t(1) << 40;
  if (static_cast<uint64_t>(a.ldx) * 2 >= kMaxStride || static_cast<uint64_t>(kb) * C::SA >= kMaxStride)
    return cudaErrorNotSupported;
  auto encode = tensor_map_encoder();
  if (encode == nullptr) return cudaErrorNotSupported;
  CUtensorMap xmap, wmap;
  {
    // Gathered rows may name any int32 coordinate. The one-row box reads
    // only requested rows, so this bound need not match the allocation size.
    const cuuint64_t rows = a.gather != nullptr ? static_cast<cuuint64_t>(INT32_MAX) : a.max_assignments;
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(a.k), rows};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(a.ldx) * 2};
    const cuuint32_t box[2] = {64, 1};
    const cuuint32_t estr[2] = {1, 1};
    if (encode(&xmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<__nv_bfloat16*>(a.x), dims, strides, box, estr,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }
  {
    // Expert e's weight blocks start at e * n / 16.
    const cuuint64_t dims[3] = {C::SA / 8, static_cast<cuuint64_t>(kb),
                                static_cast<cuuint64_t>(a.num_experts * (a.n / 16))};
    const cuuint64_t strides[2] = {C::SA, static_cast<cuuint64_t>(kb) * C::SA};
    const cuuint32_t box[3] = {C::SA / 8, 1, C::kBlocks};
    const cuuint32_t estr[3] = {1, 1, 1};
    if (encode(&wmap, CU_TENSOR_MAP_DATA_TYPE_UINT64, 3, const_cast<uint8_t*>(a.weight), dims, strides, box, estr,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }

  MoeSm120Params p{};
  const bool nf12 = S == WeightSource::kNF12;
  p.x = a.x;
  p.ldx = a.ldx;
  p.gather = a.gather;
  p.gather_div = a.gather_div;
  p.scatter = a.scatter;
  p.row_scale = a.row_scale;
  p.expert_offsets = a.expert_offsets;
  p.max_assignments = a.max_assignments;
  p.num_experts = static_cast<int>(a.num_experts);
  p.escapes = nf12 ? a.escapes : nullptr;
  p.escape_offsets = nf12 ? a.escape_offsets : nullptr;
  p.exp_base = nf12 ? a.exp_base : nullptr;
  p.y = a.y;
  p.ldy = a.ldy;
  p.n = static_cast<int>(a.n);
  p.kb = static_cast<int>(kb);
  p.n_tiles = static_cast<int>(n_tiles);
  // Traverse n fastest within groups fitting one third of L2, keeping an
  // expert's panels resident across token tiles.
  const int64_t w_panel = static_cast<int64_t>(BN) * a.k * 3 / 2;
  const int64_t budget = static_cast<int64_t>(a.dev->l2_bytes ? a.dev->l2_bytes : (32 << 20)) / 3;
  p.group_n = static_cast<int>(std::clamp<int64_t>(budget / w_panel, 1, n_tiles));

  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(static_cast<unsigned>(std::min<int64_t>(bound, units)));
  lc.blockDim = dim3(sm120k::kThreads);
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
cudaError_t launch_src(const MoeLinearArgs& a) {
  return a.source == WeightSource::kNF12 ? launch<BN, BM, WN, SX, WeightSource::kNF12>(a)
                                         : launch<BN, BM, WN, SX, WeightSource::kBF16T>(a);
}

struct MoeSm120Config {
  int bn, bm;
  // Relative throughput per BN * BM tile area at full occupancy.
  double rate;
  const char* name;
  cudaError_t (*launch)(const MoeLinearArgs&);
};

#define NF12_MOE_SM120_CONFIG(BN, BM, WN, SX, RATE) \
  MoeSm120Config { BN, BM, RATE, "w" #BN "_t" #BM "_wn" #WN "_sx" #SX, &launch_src<BN, BM, WN, SX> }

// A 256x128 tile's bf16t ring cannot fit alongside schedule and epilogue maps.
constexpr MoeSm120Config kConfigs[] = {
    NF12_MOE_SM120_CONFIG(128, 32, 8, 6, 0.66),
    NF12_MOE_SM120_CONFIG(256, 32, 8, 4, 0.67),
    NF12_MOE_SM120_CONFIG(128, 64, 8, 4, 0.40),
    NF12_MOE_SM120_CONFIG(256, 64, 8, 3, 1.00),
    NF12_MOE_SM120_CONFIG(128, 128, 8, 3, 1.03),
    NF12_MOE_SM120_CONFIG(256, 96, 8, 2, 1.07),
    NF12_MOE_SM120_CONFIG(128, 192, 8, 2, 0.47),
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

// Device-only counts are estimated as Poisson(A / E), using a normal tail
// for each expert's expected m-tile count. A partial first wave costs its
// occupancy fraction. Later partial waves cost f^(1/4), reflecting the
// near-full cost of a ragged last wave with long tiles.
int heuristic(int64_t max_assignments, int64_t num_experts, int64_t n, const DeviceInfo& dev) {
  const double mu = static_cast<double>(max_assignments) / static_cast<double>(num_experts);
  const double sd = std::sqrt(mu), units = std::max(1, dev.num_sms);
  int best = 0;
  double best_cost = 0.0;
  for (int c = 0; c < kNumConfigs; ++c) {
    const MoeSm120Config& cf = kConfigs[c];
    double per_expert = 1.0 - std::exp(-mu);  // P(count > 0)
    for (int j = 1; j * cf.bm < mu + 6.0 * sd + cf.bm; ++j)
      per_expert += 0.5 * std::erfc((j * cf.bm + 0.5 - mu) / (sd * std::sqrt(2.0)));  // P(count > j * bm)
    const double tiles = static_cast<double>(num_experts) * per_expert * static_cast<double>((n + cf.bn - 1) / cf.bn);
    const double frac = tiles / units - std::floor(tiles / units);
    const double waves = tiles > units ? std::floor(tiles / units) + std::pow(frac, 0.25) : tiles / units;
    const double cost = waves * cf.bn * cf.bm / cf.rate;
    if (c == 0 || cost < best_cost) {
      best = c;
      best_cost = cost;
    }
  }
  return best;
}

}  // namespace

bool moe_large_sm120_supported(const DeviceInfo& dev, int64_t num_experts) {
  if (dev.sm_arch != 120 && dev.sm_arch != 121) return false;
  if (num_experts < 1 || num_experts > kMoeSm120MaxExperts) return false;
  // Generic cubins and JIT PTX can match binaryVersion while containing only
  // the trapping body. Only the arch-specific probe has static shared memory.
  static std::atomic<int8_t> cache[kMaxDevices] = {};  // 0 unknown, 1 yes, -1 no
  const bool cacheable = dev.device >= 0 && dev.device < kMaxDevices;
  if (cacheable) {
    const int8_t c = cache[dev.device].load(std::memory_order_relaxed);
    if (c != 0) return c > 0;
  }
  cudaFuncAttributes attr{};
  bool ok = false;
  if (cudaFuncGetAttributes(&attr, moe_large_sm120_probe) == cudaSuccess)
    ok = attr.binaryVersion == dev.sm_arch && attr.sharedSizeBytes > 0;
  else
    cudaGetLastError();  // clear the non-sticky lookup error we caused
  if (cacheable) cache[dev.device].store(ok ? 1 : -1, std::memory_order_relaxed);
  return ok;
}

int moe_large_sm120_num_configs() { return kNumConfigs; }

const char* moe_large_sm120_config_name(int config) {
  return config >= 0 && config < kNumConfigs ? kConfigs[config].name : "invalid";
}

int moe_large_sm120_heuristic(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t,
                              const DeviceInfo& dev) {
  return heuristic(max_assignments, num_experts, n, dev);
}

size_t moe_linear_large_sm120_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int) { return 0; }

cudaError_t moe_linear_large_sm120(const MoeLinearArgs& a, int config) {
  if (config >= kNumConfigs) return cudaErrorInvalidValue;
  if (a.dev == nullptr || !moe_large_sm120_supported(*a.dev, a.num_experts)) return cudaErrorNotSupported;
  const int c = config >= 0 ? config : moe_large_sm120_heuristic(a.max_assignments, a.num_experts, a.n, a.k, *a.dev);
  return kConfigs[c].launch(a);
}

}  // namespace nf12::detail
