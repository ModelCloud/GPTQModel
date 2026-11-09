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
// Dense problem for linear/sm100/large_m_tc_kernel.cuh.
// Stream-K scheduling units are CTAs or CTA pairs. Tiles [0, dp_tiles) are
// data-parallel. Remaining k-blocks form contiguous stream-K ranges. A split
// tile's k = 0 segment is its owner's last segment. Other contributors
// process theirs first and publish FP32 partials followed by tags. The owner
// reduces in ascending unit order, fixed by shape, config and grid.
//
// Cluster split-K uses one non-persistent kc-CTA cluster per tile, with rank
// r taking K range r and workspace partials summed in rank order.
// tm1_bn256_cs with kc = 1 uses the persistent data-parallel schedule.
// The narrower cluster-split configs always use the split kernel.
#include <algorithm>
#include <atomic>
#include <cmath>
#if defined(NF12_TC_SWEEP)
#include <cstdlib>
#endif

#include "linear/sm100/large_m_tc.h"
#include "linear/sm100/large_m_tc_kernel.cuh"

namespace nf12::detail {
namespace {

using tcg::APath;
using tcg::kBK;
using tcg::kBlocksPerTile;
using tcg::kMaxDevices;
using tcg::kTileRows;
using tcg::TcTile;
using tcg::Work;

struct TcParams {
  CUtensorMap ymap;  // CTA pairs: y [m][n] for the epilogue's TMA stores
  const uint4* escapes;
  uint32_t num_escapes;
  uint32_t base_x4;
  const __nv_bfloat16* bias;
  __nv_bfloat16* y;
  int64_t ldy;
  int m, n, kb;  // kb = k / 64
  int n_tiles, m_tiles;
  int band;          // raster: tiles per band (see make_plan)
  bool m_fast;       // raster: bands of m-tiles walked m-fastest, else bands of n-tiles walked n-fastest
  int dp_tiles;      // tiles [0, dp_tiles) are data-parallel
  int64_t sk_units;  // (tiles - dp_tiles) * kb k-blocks split over the units
  float* partials;   // [grid CTAs][BN][128] FP32: stream-K contributors, every cluster split-K CTA
  uint64_t* tags;    // [grid CTAs]: publication tags, zeroed by the launcher
  int kc;            // cluster split-K: CTAs per tile
};

template <int BN, int kCtas_, bool kSplit = false>
struct DenseProblem {
  static constexpr int kBN = BN;
  static constexpr int kCtas = kCtas_;
  static constexpr bool kStreamK = !kSplit;
  static constexpr bool kClusterSplit = kSplit;
  static constexpr bool kBias = true;
  static constexpr bool kRowMaps = false;
  static constexpr bool kLockstepA = false;
  // Cluster split-K: deep weight and x rings (TileCfg::kSplitRings).
  static constexpr int kSplitStages = kSplit ? 4 : 0;
  using Params = TcParams;

#if defined(NF12_TC_DEVICE)
  class Sched {
   public:
    __device__ Sched(const TcParams& p, int unit, int units) : p_(p), unit_(unit), units_(units) {
      dp_ = unit_;
      sk_pos_ = sk_begin(unit_);
      sk_end_ = sk_begin(unit_ + 1);
    }
    __device__ int unit() const { return unit_; }
    __device__ int units() const { return units_; }
    __device__ int64_t sk_begin(int64_t u) const {
      return p_.sk_units * u / units_;
    }
    __device__ bool next(Work& w) {
      if constexpr (kSplit) {
        // One item: k-range unit % kc of tile unit / kc.
        if (dp_ < 0) return false;
        const int r = unit_ % p_.kc;
        w = {unit_ / p_.kc, p_.kb * r / p_.kc, p_.kb * (r + 1) / p_.kc};
        dp_ = -1;
        return true;
      }
      if (dp_ < p_.dp_tiles) {
        w = {dp_, 0, p_.kb};
        dp_ += units_;
        return true;
      }
      if (sk_pos_ < sk_end_) {
        const int t = static_cast<int>(sk_pos_ / p_.kb);
        const int k0 = static_cast<int>(sk_pos_ % p_.kb);
        const int k1 = static_cast<int>(k0 + min(static_cast<int64_t>(p_.kb - k0), sk_end_ - sk_pos_));
        w = {p_.dp_tiles + t, k0, k1};
        sk_pos_ += k1 - k0;
        return true;
      }
      return false;
    }
    // Units (other than this one) that published a partial of split tile
    // `tile`: those whose stream-K range starts strictly inside it, ascending.
    __device__ bool contributes(int u, int tile) const {
      const int64_t t0 = static_cast<int64_t>(tile - p_.dp_tiles) * p_.kb;
      const int64_t a = sk_begin(u);
      return a > t0 && a < t0 + p_.kb && a < sk_begin(u + 1);
    }

   private:
    const TcParams& p_;
    int unit_, units_, dp_;
    int64_t sk_pos_, sk_end_;
  };

  __device__ static TcTile tile(const TcParams& p, const Work& w, int rank) {
    int a, c;
    tcg::band_coords(w.tile, p.band, p.m_fast ? p.m_tiles : p.n_tiles, p.m_fast ? p.n_tiles : p.m_tiles, a, c);
    const int mt = p.m_fast ? a : c, nt = p.m_fast ? c : a;
    TcTile t;
    t.wblock = (nt * kCtas + rank) * kBlocksPerTile;
    t.blocks = min(kBlocksPerTile, p.n / kSuperAtomRows - t.wblock);
    t.row0 = (nt * kCtas + rank) * kTileRows;
    t.m0 = mt * BN;
    t.tokens = min(BN, p.m - t.m0);
    // Split-K with one m-tile reads every weight byte once.
    t.w_once = (kSplit || p.sk_units > 0) && p.m_tiles == 1;
    t.base_x4 = p.base_x4;
    t.escapes = p.escapes;
    t.num_escapes = p.num_escapes;
    return t;
  }
  __device__ static __nv_bfloat16* y_row(const TcParams& p, const TcTile& t, int c) {
    return p.y + static_cast<int64_t>(t.m0 + c) * p.ldy;
  }
#endif  // NF12_TC_DEVICE
};

struct TcConfig {
  int bn;              // tokens per tile (UMMA N)
  int ctas;            // 1, or 2 for a cta_group::2 pair (256 weight rows per tile)
  bool cluster_split;  // cluster split-K (kc CTAs per tile), else stream-K hybrid
  APath path;          // where decoded A lives / ring structure, see TileCfg
  const char* name;
  bool data_parallel = false;
};

// tc* stores decoded A in shared memory, tm* in TMEM. tc2_bn512_sk reuses
// one 256-row A tile for two N = 256 MMAs over all 512 TMEM columns,
// halving decode work per FLOP compared with BN = 256.
constexpr TcConfig kConfigs[] = {
    {128, 1, false, APath::kSmem, "tc1_bn128_sk"},
    {256, 2, false, APath::kSmem, "tc2_bn256_sk"},
    {256, 1, false, APath::kTmem, "tm1_bn256_sk"},
    {256, 2, false, APath::kTmem, "tm2_bn256_sk"},
    {256, 2, false, APath::kTmemCoupled, "tw2_bn256_sk"},
    {256, 1, true, APath::kTmem, "tm1_bn256_cs"},
    {512, 2, false, APath::kSmem, "tc2_bn512_sk"},
    {64, 1, true, APath::kTmem, "tm1_bn64_cs"},
    {128, 1, true, APath::kTmem, "tm1_bn128_cs"},
    {512, 2, false, APath::kSmem, "tc2_bn512_dp", true},
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);
constexpr int kCfgBn128 = 0, kCfgTmBn256 = 2, kCfgTmPairBn256 = 3, kCfgCoupledPairBn256 = 4, kCfgClusterSplit = 5,
              kCfgBn512 = 6, kCfgSplitBn64 = 7, kCfgSplitBn128 = 8, kCfgBn512Dp = 9;
constexpr int kMaxClusterSplit = 8;  // portable cluster size

struct Plan {
  int config;
  int grid;  // CTAs
  int kc;    // cluster split-K: CTAs per tile (1: none)
  int n_tiles, m_tiles, kb;
  int64_t tiles;  // the kernel indexes tiles as int: linear_large_m_tc rejects > INT32_MAX / 2
  int band;
  bool m_fast;
  int dp_tiles;
  int64_t sk_units;
  size_t partial_bytes, tag_bytes;
  size_t workspace() const { return partial_bytes + tag_bytes; }
};

// Cache resident CTAs in kc-sized clusters per device, returning 0 if
// unavailable. Use the NF12 kernel for both sources to preserve the plan.
template <int BN>
int split_resident(const DeviceInfo& dev, int kc) {
  using Prob = DenseProblem<BN, 1, true>;
  constexpr auto kKernel = tcg::tc_gemm_kernel<Prob, WeightSource::kNF12, APath::kTmem>;
  static std::atomic<int> cache[kMaxDevices][kMaxClusterSplit + 1];  // 0 unknown, -1 unavailable
  const bool cacheable = dev.device >= 0 && dev.device < kMaxDevices;
  if (cacheable) {
    const int c = cache[dev.device][kc].load(std::memory_order_relaxed);
    if (c != 0) return std::max(c, 0);
  }
  const int r =
      tcg::resident_ctas<kKernel, tcg::ProbCfg<Prob, WeightSource::kNF12, APath::kTmem>::kSmemBytes>(dev, kc);
  if (cacheable) cache[dev.device][kc].store(r > 0 ? r : -1, std::memory_order_relaxed);
  return r;
}
int split_resident(const DeviceInfo& dev, int bn, int kc) {
  return bn == 64 ? split_resident<64>(dev, kc) : bn == 128 ? split_resident<128>(dev, kc) : split_resident<256>(dev, kc);
}

// With multiple m-tiles, stream-K pays only when balancing idle units
// outweighs cold split segments and partial fixups. A single m-tile keeps
// stream-K to sustain weight streaming through the final wave.
bool dp_efficient(int64_t tiles, int64_t m_tiles, int64_t units) {
  const int64_t waves = (tiles + units - 1) / units;
  return m_tiles > 1 && tiles >= units && tiles * 10 >= waves * units * 9;
}

// The stream-K / data-parallel configs by shape, see heuristic_config.
int tiled_config(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (m <= 128) return kCfgBn128;
  const int64_t pair_tiles = (n + 255) / 256 * ((m + 255) / 256);
  if (m > 256 && (pair_tiles < std::max(1, dev.num_sms / 2) || (k >= 8192 && m <= 768))) return kCfgTmBn256;
  if (dev.sm_arch != 100 && k >= 5120 && m * n >= (int64_t(64) << 20)) return kCfgCoupledPairBn256;
  return kCfgTmPairBn256;
}

// Single-CTA model in microseconds. A stage costs
// kStageUs + kStageTokUs * tokens / 256, a launch costs kWaveUs and
// stream-K fixup costs kFixupUs * (1 + tokens / 256).
// tokens = min(M, BN), the tile's live columns.
constexpr double kStageUs = 0.25, kStageTokUs = 0.22, kWaveUs = 9.5, kFixupUs = 13.8;

// CTA-pair model in microseconds. Data-parallel plans run whole waves.
// Hybrid plans add a stream-K remainder at sk_slow times the stage cost,
// fixup_us and launch_us. BN = 512 chooses the lower modelled plan.
struct PairCost {
  int bn;
  double stage_us, sk_slow, fixup_us, launch_us;
};
constexpr PairCost kTmPairCost{256, 0.508, 1.106, 19.3, 1.13};
constexpr PairCost kBn512Cost{512, 0.87, 1.041, 42.1, 0.75};

struct PlanUs {
  double dp, hybrid;  // data-parallel waves; full waves + stream-K remainder
};
PlanUs pair_plan_us(int64_t m, int64_t n, int kb, const PairCost& c, int64_t units) {
  const int64_t tiles = (n + 255) / 256 * ((m + c.bn - 1) / c.bn);
  const double tile_us = kb * c.stage_us;
  const int64_t dp_tiles = std::max<int64_t>(0, tiles / units - 1) * units;
  return {static_cast<double>((tiles + units - 1) / units) * tile_us + c.launch_us,
          static_cast<double>(dp_tiles / units) * tile_us +
              static_cast<double>(tiles - dp_tiles) * tile_us / units * c.sk_slow + c.fixup_us + c.launch_us};
}

// The plan make_plan runs, by the model.
double tm_pair_us(int64_t m, int64_t n, int kb, int64_t units) {
  const PlanUs us = pair_plan_us(m, n, kb, kTmPairCost, units);
  const int64_t m_tiles = (m + 255) / 256;
  return dp_efficient((n + 255) / 256 * m_tiles, m_tiles, units) ? us.dp : us.hybrid;
}

// Larger FP32 partials make BN = 512's stream-K tail expensive, so choose
// data-parallel whenever its model wins, even below 90% wave efficiency.
bool bn512_data_parallel(int64_t m, int64_t n, int kb, int64_t units) {
  const PlanUs us = pair_plan_us(m, n, kb, kBn512Cost, units);
  return m > 512 && (n + 255) / 256 * ((m + 511) / 512) >= units && us.dp <= us.hybrid;
}
double bn512_us(int64_t m, int64_t n, int kb, int64_t units) {
  const PlanUs us = pair_plan_us(m, n, kb, kBn512Cost, units);
  return bn512_data_parallel(m, n, kb, units) ? us.dp : us.hybrid;
}

// Cluster split-K model in microseconds for BN tokens x 128 weight rows.
// Each resident wave costs wave_us plus ceil(kb / kc) stages at
// stage_us + stage_tok_us * tokens / 256. A split tile adds
// reduce_us * tokens / 256 for the L2 partial reduction.
// kc = 1 uses persistent data-parallel only for BN = 256.
struct SplitCost {
  int config;
  double wave_us, stage_us, stage_tok_us, reduce_us;
};
constexpr SplitCost kSplitCosts[] = {
    {kCfgClusterSplit, 5.25, 0.299, 0.110, 2.49},
    {kCfgSplitBn64, 4.97, 0.264, 0.013, 2.78},
    {kCfgSplitBn128, 5.42, 0.246, 0.105, 2.04},
};

struct SplitChoice {
  int config;
  int kc;
  double us;
};

// The kc minimizing the model for one split config.
SplitChoice split_ways(int64_t m, int64_t n, int kb, const DeviceInfo& dev, const SplitCost& c) {
  const int bn = kConfigs[c.config].bn;
  const int64_t tiles = (n + kTileRows - 1) / kTileRows * ((m + bn - 1) / bn);
  const double tok = static_cast<double>(std::min<int64_t>(m, bn)) / 256;
  const double stage = c.stage_us + c.stage_tok_us * tok;
  const int64_t units = std::max(1, dev.num_sms);
  SplitChoice best{c.config, 1, static_cast<double>((tiles + units - 1) / units) * (c.wave_us + kb * stage)};
  for (int kc = 2; kc <= std::min(kMaxClusterSplit, kb) && tiles * kc <= INT32_MAX / 2; ++kc) {
    const int resident = split_resident(dev, bn, kc);
    if (resident < kc) continue;
    const double us = static_cast<double>((tiles * kc + resident - 1) / resident) *
                          (c.wave_us + (kb + kc - 1) / kc * stage) +
                      c.reduce_us * tok;
    if (us < best.us) best = {c.config, kc, us};
  }
  return best;
}

// The fastest split config and kc by the model.
SplitChoice best_split(int64_t m, int64_t n, int kb, const DeviceInfo& dev) {
  SplitChoice best{};
  for (const SplitCost& c : kSplitCosts) {
    const SplitChoice s = split_ways(m, n, kb, dev, c);
    if (best.config == 0 || s.us < best.us) best = s;
  }
  return best;
}

const SplitCost& split_cost(int config) {
  for (const SplitCost& c : kSplitCosts)
    if (c.config == config) return c;
  return kSplitCosts[0];
}

// Thor favors TMEM-A CTA pairs for activation reuse, except when tiling
// leaves too few pairs or long K makes single CTAs cheaper. Coupled rings
// help only its larger problems. B200 uses separate rings and compares
// cluster split-K against stream-K for moderate M. Narrow router weights
// need split-K to expose enough parallel work on either architecture.
// BN = 512 shares one decode across more tokens but can lose to tile
// quantization, so the pair model must predict at least a 2% gain.
int heuristic_config(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  const int kb = static_cast<int>(k / kBK);
  if ((dev.sm_arch == 100 || dev.sm_arch == 110) && n <= 256 && m <= 4096) return best_split(m, n, kb, dev).config;
  const int64_t pair_units = std::max(1, dev.num_sms / 2);
  // A sufficiently filled single wave avoids short-K stream reduction.
  // Count useful outputs, not padded tiles, to reject sparse edge waves.
  if (dev.sm_arch == 100 && m >= 512 && k <= 4096 &&
      (n + 255) / 256 * ((m + 511) / 512) <= pair_units &&
      m * n >= int64_t{512} * (k <= 2048 ? 128 : 192) * pair_units)
    return kCfgBn512Dp;
  if (m >= (dev.sm_arch == 100 ? 512 : 1024) &&
      bn512_us(m, n, kb, pair_units) < 0.98 * tm_pair_us(m, n, kb, pair_units))
    return kCfgBn512;
  if (dev.sm_arch != 100 || m > 1024) return tiled_config(m, n, k, dev);
  const int sk_config = m <= 128 ? kCfgTmBn256 : tiled_config(m, n, k, dev);
  const TcConfig& sk = kConfigs[sk_config];
  const int64_t sk_tiles = (n + kTileRows * sk.ctas - 1) / (kTileRows * sk.ctas) * ((m + sk.bn - 1) / sk.bn);
  const double tok = static_cast<double>(std::min<int64_t>(m, sk.bn)) / 256;
  const double sk_us = static_cast<double>(sk_tiles * kb) / std::max(1, dev.num_sms / sk.ctas) *
                           (kStageUs + kStageTokUs * tok) +
                       kWaveUs + kFixupUs * (1 + tok);
  if (m >= 256 && n >= 25600 && k <= 5120) return sk_config;
  const SplitChoice s = best_split(m, n, kb, dev);
  return s.us < 0.9 * sk_us ? s.config : sk_config;
}

Plan make_plan(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config) {
  Plan pl{};
  pl.config = config < 0 ? heuristic_config(m, n, k, dev) : config;
  const TcConfig& c = kConfigs[pl.config];
  const int rows = kTileRows * c.ctas;
  pl.n_tiles = static_cast<int>((n + rows - 1) / rows);
  pl.m_tiles = static_cast<int>((m + c.bn - 1) / c.bn);
  pl.tiles = static_cast<int64_t>(pl.n_tiles) * pl.m_tiles;
  pl.kb = static_cast<int>(k / kBK);
  const int units = std::max(1, dev.num_sms / c.ctas);
  // Compare m-fast bands sharing weights with n-fast bands sharing x.
  // Keep a band within half of L2, but at least large enough to minimize
  // one wave's traffic at b = sqrt(units * other_panel / own_panel).
  // Choose the order with fewer modelled DRAM bytes.
  const int64_t w_panel = static_cast<int64_t>(rows) * k * 3 / 2;
  const int64_t x_panel = static_cast<int64_t>(c.bn) * k * 2;
  const int64_t budget = static_cast<int64_t>(dev.l2_bytes ? dev.l2_bytes : (32 << 20)) / 2;
  const auto band_of = [&](int64_t own, int64_t other, int tiles) {
    const int64_t balanced = std::llround(std::sqrt(static_cast<double>(units) * other / own));
    return std::clamp<int64_t>(std::max(budget / own, balanced), 1, tiles);
  };
  const int64_t band_m = band_of(x_panel, w_panel, pl.m_tiles);
  const int64_t band_n = band_of(w_panel, x_panel, pl.n_tiles);
  const int64_t w_total = w_panel * pl.n_tiles, x_total = x_panel * pl.m_tiles;
  const int64_t traffic_m = w_total * ((pl.m_tiles + band_m - 1) / band_m) + x_total;
  const int64_t traffic_n = x_total * ((pl.n_tiles + band_n - 1) / band_n) + w_total;
  pl.m_fast = traffic_m <= traffic_n;
  pl.band = static_cast<int>(pl.m_fast ? band_m : band_n);
  int grid_units;
  pl.kc = c.cluster_split ? split_ways(m, n, pl.kb, dev, split_cost(pl.config)).kc : 1;
#if defined(NF12_TC_SWEEP)
  // Sweep builds only (tests/kernels/linear/large_m_tc_split_sweep.py).
  if (const char* e = getenv("NF12_TC_KC"); e != nullptr && c.cluster_split) pl.kc = atoi(e);
#endif
  if (c.cluster_split && (pl.kc > 1 || pl.config != kCfgClusterSplit)) {
    // One non-persistent cluster per tile, with one partial per CTA.
    grid_units = static_cast<int>(pl.tiles * pl.kc);
    pl.dp_tiles = static_cast<int>(pl.tiles);
    pl.sk_units = 0;
    pl.partial_bytes = static_cast<size_t>(grid_units) * c.bn * kTileRows * sizeof(float);
  } else if (c.data_parallel || c.cluster_split ||
             (pl.config == kCfgBn512 ? bn512_data_parallel(m, n, pl.kb, units)
                                    : dp_efficient(pl.tiles, pl.m_tiles, units))) {
    grid_units = static_cast<int>(std::min<int64_t>(pl.tiles, units));
    pl.dp_tiles = static_cast<int>(pl.tiles);
    pl.sk_units = 0;
  } else {
    // Keep full waves except the last data-parallel. Balance the remaining
    // one or two waves across all units by K range.
    grid_units = units;
    pl.dp_tiles = static_cast<int>(std::max<int64_t>(0, pl.tiles / units - 1) * units);
    pl.sk_units = (pl.tiles - pl.dp_tiles) * pl.kb;
  }
  pl.grid = grid_units * c.ctas;
  if (pl.sk_units > 0) {
    // Partials, then one 64-bit tag per CTA (zeroed per launch, see
    // stream_k_tag).
    pl.partial_bytes = static_cast<size_t>(pl.grid) * c.bn * kTileRows * sizeof(float);
    pl.tag_bytes = (static_cast<size_t>(pl.grid) * sizeof(uint64_t) + 255) & ~size_t(255);
  }
  return pl;
}

template <int BN, int kCtas, WeightSource kSrc, APath kPath, bool kSplit>
cudaError_t launch(const LinearArgs& a, const Plan& pl) {
  using Prob = DenseProblem<BN, kCtas, kSplit>;
  using Cfg = tcg::ProbCfg<Prob, kSrc, kPath>;
  // Reject shapes exceeding tensor-map limits.
  CUtensorMap xmap, wmap;
  if (!tcg::encode_x_map(&xmap, a.x, a.k, a.m, a.ldx, Cfg::kXSubRows) ||
      !tcg::encode_weight_map(&wmap, a.weight, Cfg::kSuperAtom, pl.kb, a.n / kSuperAtomRows))
    return cudaErrorNotSupported;

  TcParams p{};
  if (kCtas == 2 && !tcg::encode_y_map(&p.ymap, a.y, a.n, a.m, a.ldy)) return cudaErrorNotSupported;
  p.escapes = a.escapes;
  p.num_escapes = a.num_escapes;
  p.base_x4 = exp_base_x4(a.exp_base);
  p.bias = a.bias;
  p.y = a.y;
  p.ldy = a.ldy;
  p.m = static_cast<int>(a.m);
  p.n = static_cast<int>(a.n);
  p.kb = pl.kb;
  p.n_tiles = pl.n_tiles;
  p.m_tiles = pl.m_tiles;
  p.band = pl.band;
  p.m_fast = pl.m_fast;
  p.dp_tiles = pl.dp_tiles;
  p.sk_units = pl.sk_units;
  if constexpr (kSplit) p.partials = static_cast<float*>(a.workspace);
  if (pl.sk_units > 0) {
    p.partials = static_cast<float*>(a.workspace);
    p.tags = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(a.workspace) + pl.partial_bytes);
    cudaError_t err = cudaMemsetAsync(p.tags, 0, pl.tag_bytes, a.stream);
    if (err != cudaSuccess) return err;
  }
  p.kc = pl.kc;
  return tcg::launch_kernel<tcg::tc_gemm_kernel<Prob, kSrc, kPath>, Cfg::kSmemBytes>(
      *a.dev, pl.grid, kSplit ? pl.kc : kCtas, a.stream, xmap, wmap, p, /*pdl=*/kSplit);
}

template <int BN, int kCtas, APath kPath, bool kSplit = false>
cudaError_t launch_src(const LinearArgs& a, const Plan& pl) {
  return a.source == WeightSource::kNF12 ? launch<BN, kCtas, WeightSource::kNF12, kPath, kSplit>(a, pl)
                                       : launch<BN, kCtas, WeightSource::kBF16T, kPath, kSplit>(a, pl);
}

}  // namespace

bool large_m_tc_supported(const DeviceInfo& dev) {
  return tcg::kernel_image_supported<tcg::tc_gemm_kernel<DenseProblem<128, 1>, WeightSource::kNF12, APath::kSmem>>(
      dev);
}

int large_m_tc_num_configs() { return kNumConfigs; }

const char* large_m_tc_config_name(int config) {
  return config >= 0 && config < kNumConfigs ? kConfigs[config].name : "invalid";
}

size_t linear_large_m_tc_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config) {
  if (config >= kNumConfigs) return 0;
  return make_plan(m, n, k, dev, config).workspace();
}

cudaError_t linear_large_m_tc(const LinearArgs& a, int config) {
  if (config >= kNumConfigs || a.dev == nullptr || !large_m_tc_supported(*a.dev)) return cudaErrorNotSupported;
  if (a.m < 1) return cudaErrorInvalidValue;
  // Kernel row/token/k indices and TMA coordinates are int32.
  if (a.m > INT32_MAX || a.n > INT32_MAX || a.k > INT32_MAX) return cudaErrorNotSupported;
  const Plan pl = make_plan(a.m, a.n, a.k, *a.dev, config);
  // Tile indices (plus one grid stride) must stay in int range.
  if (pl.tiles > INT32_MAX / 2) return cudaErrorNotSupported;
  if (a.workspace_bytes < pl.workspace() || (pl.workspace() > 0 && a.workspace == nullptr))
    return cudaErrorInvalidValue;
  // Instantiations in kConfigs order (bn, ctas, path must match the table).
  switch (pl.config) {
    case 0:
      return launch_src<128, 1, APath::kSmem>(a, pl);
    case 1:
      return launch_src<256, 2, APath::kSmem>(a, pl);
    case 2:
      return launch_src<256, 1, APath::kTmem>(a, pl);
    case 3:
      return launch_src<256, 2, APath::kTmem>(a, pl);
    case 4:
      return launch_src<256, 2, APath::kTmemCoupled>(a, pl);
    case 6:
      return launch_src<512, 2, APath::kSmem>(a, pl);
    case 7:
      return launch_src<64, 1, APath::kTmem, true>(a, pl);
    case 8:
      return launch_src<128, 1, APath::kTmem, true>(a, pl);
    case kCfgBn512Dp:
      return launch_src<512, 2, APath::kSmem>(a, pl);
    default:
      return pl.kc > 1 ? launch_src<256, 1, APath::kTmem, true>(a, pl) : launch_src<256, 1, APath::kTmem>(a, pl);
  }
}

}  // namespace nf12::detail
