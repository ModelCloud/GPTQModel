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
// Expert-major tiles use bands of m-tiles with m fastest, retaining weight
// panels and gathered x rows in L2 across n-tiles.
// Gathered rows use cp.async or a sorted workspace copy loaded by TMA.
// Both sources accumulate in a fixed order and apply row_scale in FP32
// before BF16 rounding.
#include <algorithm>
#include <cstdint>

#include "linear/sm100/large_m_tc_kernel.cuh"
#include "moe/common.cuh"
#include "moe/grouped.h"
#include "moe/sm100/large_tc.h"
#include "moe/schedule.cuh"

namespace nf12::detail {
namespace {

using tcg::APath;
using tcg::kBK;
using tcg::kBlocksPerTile;
using tcg::kTileRows;
using tcg::TcTile;
using tcg::Work;

struct MoeTcParams {
  CUtensorMap ymap;  // y [max_assignments][n] for TMA stores (no scatter)
  const __nv_bfloat16* x;  // gathered rows are read directly (else by TMA)
  int64_t ldx;
  const int32_t* gather;  // may be null
  int64_t gather_div;
  const int32_t* scatter;    // may be null
  const float* row_scale;    // may be null
  const int64_t* expert_offsets;
  int64_t max_assignments;
  const void* schedule;  // moe_schedule() list
  const int32_t* row_end;  // moe_schedule(): end of each expert's clamped sorted rows
  const uint4* escapes;  // NF12: may be null
  const int64_t* escape_offsets;
  const uint8_t* exp_base;
  __nv_bfloat16* y;
  int64_t ldy;
  int n, kb;               // kb = k / 64
  int n_tiles;             // per expert
  int blocks_per_expert;   // n / 16
  int band;                // m-tiles per raster band
};

template <int BN, int kCtas_, WeightSource kSrc, bool kLockstep = false>
struct MoeProblem {
  static constexpr int kBN = BN;
  static constexpr int kCtas = kCtas_;
  static constexpr bool kStreamK = false;
  static constexpr bool kClusterSplit = false;
  static constexpr bool kBias = false;
  static constexpr bool kRowMaps = true;
  static constexpr bool kLockstepA = kLockstep;
  using Params = MoeTcParams;

#if defined(NF12_TC_DEVICE)
  class Sched {
   public:
    __device__ Sched(const MoeTcParams& p, int unit, int units)
        : next_(unit), units_(units), kb_(p.kb), total_(p.n_tiles * *moe_schedule_count(p.schedule)) {}
    __device__ bool next(Work& w) {
      if (next_ >= total_) return false;
      w = {next_, 0, kb_};
      next_ += units_;
      return true;
    }

   private:
    int next_, units_, kb_, total_;
  };

  __device__ static TcTile tile(const MoeTcParams& p, const Work& w, int rank) {
    const int j = w.tile / p.n_tiles;
    const int4 m = __ldg(reinterpret_cast<const int4*>(moe_schedule_tiles(p.schedule)) + j);
    const int e = m.x;
    const int start = m.y - m.w * BN;
    const int end = __ldg(p.row_end + e);
    const int m_tiles = moe_ceil_div(end - start, BN);
    int mt, nt;
    tcg::band_coords(w.tile - (j - m.w) * p.n_tiles, p.band, m_tiles, p.n_tiles, mt, nt);
    TcTile t;
    const int block = (nt * kCtas + rank) * kBlocksPerTile;
    t.wblock = e * p.blocks_per_expert + block;
    t.blocks = min(kBlocksPerTile, p.blocks_per_expert - block);
    t.row0 = (nt * kCtas + rank) * kTileRows;
    t.m0 = start + mt * BN;
    t.tokens = min(BN, end - t.m0);
    t.w_once = m_tiles == 1;
    const MoeExpertCodec c = moe_expert_codec<kSrc>(p.exp_base, p.escapes, p.escape_offsets, e);
    t.base_x4 = c.base_x4;
    t.escapes = c.escapes;
    t.num_escapes =
        c.escapes != nullptr ? static_cast<uint32_t>(__ldg(p.escape_offsets + e + 1) - __ldg(p.escape_offsets + e))
                             : 0u;
    return t;
  }
  __device__ static bool x_gathered(const MoeTcParams& p) { return p.gather != nullptr; }
  __device__ static bool y_direct(const MoeTcParams& p) { return p.scatter == nullptr; }
  __device__ static int x_row(const MoeTcParams& p, const TcTile& t, int c) {
    return static_cast<int>(moe_in_row(p.gather, p.gather_div, t.m0 + c));
  }
  __device__ static const __nv_bfloat16* x_ptr(const MoeTcParams& p, int row) {
    return p.x + static_cast<int64_t>(row) * p.ldx;
  }
  __device__ static int out_row(const MoeTcParams& p, const TcTile& t, int c) {
    return static_cast<int>(moe_out_row(p.scatter, t.m0 + c));
  }
  __device__ static float row_scale(const MoeTcParams& p, int out) {
    return p.row_scale != nullptr ? __ldg(p.row_scale + out) : 1.0f;
  }
#endif  // NF12_TC_DEVICE
};

struct TcConfig {
  int bn;      // tokens per tile (UMMA N) = schedule m-tile rows
  int ctas;    // 1, or 2 for a cta_group::2 pair (256 weight rows per tile)
  APath path;  // where decoded A lives, see tcg::TileCfg
  bool pregather;  // gathered x is copied into sorted order first (no effect without a gather)
  const char* name;
};

// The coupled-ring path cannot gather rows because it has no x producer warp.
constexpr TcConfig kConfigs[] = {
    {128, 1, APath::kSmem, false, "tc1_bn128"},    {256, 2, APath::kSmem, false, "tc2_bn256"},
    {256, 1, APath::kTmem, false, "tm1_bn256"},    {256, 2, APath::kTmem, false, "tm2_bn256"},
    {512, 2, APath::kSmem, false, "tc2_bn512"},    {128, 1, APath::kSmem, true, "tc1_bn128_pg"},
    {256, 2, APath::kSmem, true, "tc2_bn256_pg"},  {256, 1, APath::kTmem, true, "tm1_bn256_pg"},
    {256, 2, APath::kTmem, true, "tm2_bn256_pg"},  {512, 2, APath::kSmem, true, "tc2_bn512_pg"},
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);
constexpr int kCfgBn128 = 0, kCfgPairBn256 = 1, kCfgTmBn256 = 2, kCfgTmPairBn256 = 3;
constexpr int kCfgTmBn256Pg = 7, kCfgTmPairBn256Pg = 8, kCfgBn512Pg = 9;

int heuristic_config(int64_t max_assignments, int64_t num_experts, int64_t k, bool gathered, const DeviceInfo& dev) {
  if (dev.sm_arch == 100) {
    if (!gathered) return kCfgTmBn256;
    if (max_assignments < 128 * num_experts) return kCfgTmBn256Pg;
    if (max_assignments >= 1024 * num_experts ||
        (k < 4096 && max_assignments >= 256 * num_experts && max_assignments < 512 * num_experts))
      return kCfgBn512Pg;
    return kCfgTmPairBn256Pg;
  }
  if (max_assignments <= 64 * num_experts) return kCfgBn128;
  return gathered ? kCfgTmPairBn256 : kCfgPairBn256;
}

struct Plan {
  int config;
  bool pregather;  // the call gathers and the config copies x into sorted order
  int grid;        // CTAs
  int n_tiles, kb, band;
  int64_t max_tiles;
  size_t schedule_bytes;
  size_t x_bytes;  // pre-gathered x [A][k] bf16, after the schedule
  size_t workspace() const { return schedule_bytes + x_bytes; }
};

Plan make_plan(int64_t A, int64_t E, int64_t n, int64_t k, const DeviceInfo& dev, int config, bool gathered) {
  Plan pl{};
  pl.config = config < 0 ? heuristic_config(A, E, k, gathered, dev) : config;
  const TcConfig& c = kConfigs[pl.config];
  pl.pregather = gathered && c.pregather;
  const int rows = kTileRows * c.ctas;
  pl.n_tiles = static_cast<int>((n + rows - 1) / rows);
  pl.kb = static_cast<int>(k / kBK);
  pl.max_tiles = pl.n_tiles * moe_max_mtiles(A, E, c.bn);
  const int64_t budget = static_cast<int64_t>(dev.l2_bytes ? dev.l2_bytes : (32 << 20)) / 2;
  pl.band = moe_large_tc_band(budget, A, k, c.bn);
  const int units = std::max(1, dev.num_sms / c.ctas);
  pl.grid = static_cast<int>(std::clamp<int64_t>(pl.max_tiles, 1, units)) * c.ctas;
  pl.schedule_bytes = (moe_schedule_bytes(A, E, c.bn) + 255) & ~size_t(255);
  pl.x_bytes = pl.pregather ? static_cast<size_t>(A) * static_cast<size_t>(k) * sizeof(__nv_bfloat16) : 0;
  return pl;
}

template <int BN, int kCtas, WeightSource kSrc, APath kPath, bool kLockstep = false>
cudaError_t launch(const MoeLinearArgs& a, const Plan& pl) {
  using Prob = MoeProblem<BN, kCtas, kSrc, kLockstep>;
  using Cfg = tcg::ProbCfg<Prob, kSrc, kPath>;
  // The x map is unused for gathered launches.
  CUtensorMap xmap, wmap;
  if (!tcg::encode_x_map(&xmap, a.x, a.k, a.max_assignments, a.ldx, Cfg::kXSubRows) ||
      !tcg::encode_weight_map(&wmap, a.weight, Cfg::kSuperAtom, pl.kb, a.num_experts * (a.n / kSuperAtomRows)))
    return cudaErrorNotSupported;
  cudaError_t err = moe_schedule(a.expert_offsets, a.num_experts, a.max_assignments, BN, a.workspace, a.stream);
  if (err != cudaSuccess) return err;

  MoeTcParams p{};
  if (kCtas == 2 && a.scatter == nullptr && !tcg::encode_y_map(&p.ymap, a.y, a.n, a.max_assignments, a.ldy))
    return cudaErrorNotSupported;
  p.x = a.x;
  p.ldx = a.ldx;
  p.gather = a.gather;
  p.gather_div = a.gather_div;
  p.scatter = a.scatter;
  p.row_scale = a.row_scale;
  p.expert_offsets = a.expert_offsets;
  p.max_assignments = a.max_assignments;
  p.schedule = a.workspace;
  p.row_end = reinterpret_cast<const int32_t*>(static_cast<const uint8_t*>(a.workspace) +
                                              moe_schedule_row_end_offset(a.max_assignments, a.num_experts, BN));
  p.escapes = a.escapes;
  p.escape_offsets = a.escape_offsets;
  p.exp_base = a.exp_base;
  p.y = a.y;
  p.ldy = a.ldy;
  p.n = static_cast<int>(a.n);
  p.kb = pl.kb;
  p.n_tiles = pl.n_tiles;
  p.blocks_per_expert = static_cast<int>(a.n / kSuperAtomRows);
  p.band = pl.band;
  return tcg::launch_kernel<tcg::tc_gemm_kernel<Prob, kSrc, kPath>, Cfg::kSmemBytes>(*a.dev, pl.grid, kCtas,
                                                                                     a.stream, xmap, wmap, p);
}

template <int BN, int kCtas, APath kPath>
cudaError_t launch_src(const MoeLinearArgs& a, const Plan& pl) {
  // SM100 merges x/A frees while Thor retains separate rings.
  if constexpr (kPath == APath::kTmem) {
    if (a.dev->sm_arch == 100)
      return a.source == WeightSource::kNF12 ? launch<BN, kCtas, WeightSource::kNF12, kPath, true>(a, pl)
                                             : launch<BN, kCtas, WeightSource::kBF16T, kPath, true>(a, pl);
  }
  return a.source == WeightSource::kNF12 ? launch<BN, kCtas, WeightSource::kNF12, kPath>(a, pl)
                                         : launch<BN, kCtas, WeightSource::kBF16T, kPath>(a, pl);
}

}  // namespace

bool moe_large_tc_supported(const DeviceInfo& dev) {
  return tcg::kernel_image_supported<
      tcg::tc_gemm_kernel<MoeProblem<128, 1, WeightSource::kNF12>, WeightSource::kNF12, APath::kSmem>>(dev);
}

int moe_large_tc_num_configs() { return kNumConfigs; }

const char* moe_large_tc_config_name(int config) {
  return config >= 0 && config < kNumConfigs ? kConfigs[config].name : "invalid";
}

// Gather presence is unknown when sizing workspace and may change the config.
// Reserve enough for either route.
size_t moe_large_tc_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                    const DeviceInfo& dev, int config) {
  if (config >= kNumConfigs || max_assignments <= 0) return 0;
  return std::max(make_plan(max_assignments, num_experts, n, k, dev, config, false).workspace(),
                  make_plan(max_assignments, num_experts, n, k, dev, config, true).workspace());
}

cudaError_t moe_linear_large_tc(const MoeLinearArgs& a, int config) {
  if (config >= kNumConfigs || a.dev == nullptr || !moe_large_tc_supported(*a.dev)) return cudaErrorNotSupported;
  if (a.max_assignments == 0) return cudaSuccess;
  const Plan pl = make_plan(a.max_assignments, a.num_experts, a.n, a.k, *a.dev, config, a.gather != nullptr);
  // Tile indices (plus one grid stride) and weight-block TMA coordinates are int32.
  if (pl.max_tiles > INT32_MAX / 2 || a.num_experts * (a.n / kSuperAtomRows) > INT32_MAX)
    return cudaErrorNotSupported;
  // The schedule is read as 16-byte vectors.
  if (a.workspace_bytes < pl.workspace() || a.workspace == nullptr ||
      (reinterpret_cast<uintptr_t>(a.workspace) & 15u) != 0)
    return cudaErrorInvalidValue;
  MoeLinearArgs b = a;
  if (pl.pregather) {
    auto* xs = reinterpret_cast<__nv_bfloat16*>(static_cast<uint8_t*>(a.workspace) + pl.schedule_bytes);
    if (cudaError_t err = moe_gather_rows(a, xs); err != cudaSuccess) return err;
    b.x = xs;
    b.ldx = a.k;
    b.gather = nullptr;
    b.gather_div = 1;
  }
  const TcConfig& c = kConfigs[pl.config];
  if (c.bn == 128) return launch_src<128, 1, APath::kSmem>(b, pl);
  if (c.bn == 512) return launch_src<512, 2, APath::kSmem>(b, pl);
  if (c.ctas == 1) return launch_src<256, 1, APath::kTmem>(b, pl);
  return c.path == APath::kSmem ? launch_src<256, 2, APath::kSmem>(b, pl) : launch_src<256, 2, APath::kTmem>(b, pl);
}

}  // namespace nf12::detail
