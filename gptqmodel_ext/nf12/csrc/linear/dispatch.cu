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
// Programmatic dependent launch applies to small-M configs with mt <= 2 on
// SM90+, the SM120 large-M family and SM100/SM110 cluster split-K tcgen05
// configs. These launches use programmatic stream serialization and call
// griddepcontrol.launch_dependents on entry, then griddepcontrol.wait before
// their first global-memory access.
// - All reads of x, weights, escapes and bias and all writes of y wait for
//   the previous kernel to complete and make its writes visible. Only block
//   scheduling and shared-memory setup overlap that kernel's tail.
// - A following programmatically serialized kernel must execute
//   griddepcontrol.wait (cudaGridDependencySynchronize) before reading y.
//   Normal launches, memcpys and events retain ordinary stream ordering.
//   CUDA graph capture records a programmatic edge.
#include <algorithm>
#include <cstdint>

#include <nf12/format.cuh>

#include "internal.h"
#include "linear/large_m.h"
#include "linear/sm120/large_m_sm120.h"
#include "linear/sm90/large_m_narrow_cluster.h"
#include "linear/sm89/large_m_narrow.h"
#include "linear/sm90/large_m_sm90.h"
#include "linear/sm100/large_m_tc.h"
#include "linear/small_m.h"

namespace nf12::detail {
namespace {

constexpr int64_t kNever = INT64_MAX;

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// Architecture-wide shape bands avoid fitting dispatch to individual near-ties.
// SM103 follows the SM100 policy.
bool use_small_m(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (dev.sm_arch == 100 || dev.sm_arch == 103) {
    if (m <= 4) return true;
    if (m <= 8) return k < 12288 || n > 4096;
    return m <= 16 && (n <= 256 || (k < 8192 && (k < 4096 || n > 16384)));
  }
  if (dev.sm_arch == 80 || dev.sm_arch == 86 || dev.sm_arch == 89) {
    if (m > kSmallMMaxM) return false;
    if (n <= 128) return true;
    if (dev.sm_arch == 80) return m <= 32 || (k >= 16384 && n <= 8192);
    if (dev.sm_arch == 86) {
      // The matching MMA choice beats small-M in this measured long-K sub-band.
      if (m > 32 && k > 24576 && k <= 25600 && n >= 4352 && n <= 4608) return false;
      return m <= 32 || (k >= 4096 && k <= 8192 && n <= 16384) ||
             (k >= 16384 && k <= 25600 && n >= 1024 && n <= 5120);
    }
    if (n >= 65536) return m <= 8;
    return k > 1024 || m <= 32;
  }
  // Thor router gates use the tcgen05 split-K family beyond eight tokens.
  if (dev.sm_arch == 110 && n <= 256) return m <= 8;
  if (large_m_sm120_supported(dev) && k <= 2048) return m < 48;
  return m <= (dev.sm_arch == 110 || large_m_sm90_supported(dev) ? 32 : kSmallMMaxM);
}

// Fused kernels decode each weight tile once per token tile. At large M,
// decoding N-slices once can cost less than repeating the decode.
// `internal_min_m` selects workspace bf16t slices consumed by linear_large_m().
// The bf16t twin consumes the same slices directly and produces the same results.
// `external_min_m` selects row-major slices for callers with cuBLAS.
// `narrow_min_m` selects that external route for widths up to narrow_n(),
// including small-M shapes, when use_narrow() does not apply.
// kNever disables a route. SM103 and SM121 follow SM100 and SM120 policies.
struct SliceRoute {
  int sm_arch;
  int64_t internal_min_m;
  int64_t external_min_m;
  int64_t narrow_min_m;
};

constexpr int64_t kNarrowN = 128;

// SM90 also covers the 256/384-row DeepSeek-V3 and Kimi-K2 router gates.
int64_t narrow_n(const DeviceInfo& dev) { return dev.sm_arch == 90 ? 384 : kNarrowN; }

// SM89 reduces warp-K partials inside one CTA without workspace. SM90+
// uses the cluster-reduced kernel in linear/sm90/large_m_narrow_cluster.cu.
bool use_narrow(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (narrow_sm8x_supported(dev)) {
    if (m < 1) return false;
    if (k == 2048 && (n == 64 || n == 128)) return m <= (n == 64 ? 1024 : 512);
    return k == 4096 && n == 128 && m <= 192;
  }
  if (n > narrow_n(dev) || !narrow_sm120_supported(dev)) return false;
  if (dev.sm_arch == 90) {
    if (k <= 4096) return n <= kNarrowN && m <= 512;
    return k <= 8192 && m <= (n <= kNarrowN ? 256 : 128);
  }
  if (dev.sm_arch == 100 || dev.sm_arch == 103) {
    if (k <= 2048) return m <= (n <= 64 ? 2048 : 1024);
    return k <= 4096 && m <= 256;
  }
  return k <= 4096 && (m <= 128 || (m <= 256 && k <= 2048));
}

constexpr SliceRoute kSliceRoutes[] = {
    {80, kNever, 1024, 65},
    {86, kNever, 2048, 65},
    {89, kNever, kNever, 65},
    {90, 8192, 2048, 33},
    {100, kNever, 8192, 2048},
    {103, kNever, 8192, 2048},
    {110, kNever, kNever, 2048},
    {120, kNever, 16384, 16},
    {121, kNever, 16384, 16},
};

// Maximum decoded slice sizes in BF16 bytes. The internal buffer lives in
// workspace, while the external route uses the caller's allocator.
constexpr int64_t kInternalSliceBytes = int64_t(128) << 20;
constexpr int64_t kExternalSliceBytes = int64_t(256) << 20;
constexpr size_t kAlign = 256;

SliceRoute slice_route(const DeviceInfo& dev) {
  for (const SliceRoute& r : kSliceRoutes) {
    if (r.sm_arch == dev.sm_arch) {
      // The internal route requires the architecture-specific fused family.
      if (dev.sm_arch >= 90 && !large_m_sm90_supported(dev) && !large_m_tc_supported(dev))
        return {r.sm_arch, kNever, r.external_min_m, r.narrow_min_m};
      return r;
    }
  }
  return {dev.sm_arch, kNever, kNever, kNever};
}

// Equal slices of at most `slice_bytes` (BF16; never below 16 rows), so the
// last slice is not a sliver.
int64_t slice_rows(int64_t n, int64_t k, int64_t slice_bytes) {
  const int64_t cap = std::max<int64_t>(16, slice_bytes / (2 * k) / 16 * 16);
  const int64_t slices = (n + cap - 1) / cap;
  return (n / 16 + slices - 1) / slices * 16;
}

// Slice height of the internal route for (m, n, k), 0 for the fused kernels.
int64_t internal_slice_rows(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  return !use_small_m(m, n, k, dev) && m >= slice_route(dev).internal_min_m ? slice_rows(n, k, kInternalSliceBytes)
                                                                            : 0;
}

size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

size_t slice_buffer_bytes(int64_t rows, int64_t k) { return align_up(static_cast<size_t>(rows) * k * 2); }

// Workspace: the slice buffer, then the GEMM's own workspace for the larger
// of the two slice heights (every slice but the last has `rows`).
size_t sliced_workspace_bytes(int64_t m, int64_t n, int64_t k, int64_t rows, const DeviceInfo& dev) {
  const int64_t last = n - (n - 1) / rows * rows;
  return slice_buffer_bytes(rows, k) + std::max(linear_large_m_workspace_bytes(m, rows, k, dev),
                                                linear_large_m_workspace_bytes(m, last, k, dev));
}

cudaError_t linear_sliced(const LinearArgs& a, int64_t rows) {
  const size_t buf_bytes = slice_buffer_bytes(rows, a.k);
  uint8_t* buf = static_cast<uint8_t*>(a.workspace);
  LinearArgs s = a;
  s.source = WeightSource::kBF16T;
  s.escapes = nullptr;
  s.num_escapes = 0;
  s.exp_base = 0;
  s.workspace = buf + buf_bytes;
  s.workspace_bytes = a.workspace_bytes - buf_bytes;
  for (int64_t n0 = 0; n0 < a.n; n0 += rows) {
    s.n = std::min(rows, a.n - n0);
    s.y = a.y + n0;
    s.bias = a.bias != nullptr ? a.bias + n0 : nullptr;
    if (a.source == WeightSource::kNF12) {
      // Escape indices are absolute, so a row slice decodes with the whole table.
      if (cudaError_t err = decode_bf16t(a.weight + n0 * a.k * 3 / 2, a.escapes, a.exp_base, s.n, a.k, buf, a.stream);
          err != cudaSuccess)
        return err;
      s.weight = buf;
    } else {
      s.weight = a.weight + n0 * a.k * 2;
    }
    if (cudaError_t err = linear_large_m(s); err != cudaSuccess) return err;
  }
  return cudaSuccess;
}

}  // namespace

size_t linear_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (use_narrow(m, n, k, dev)) return 0;
  if (use_small_m(m, n, k, dev)) return small_m_workspace_bytes(small_m_heuristic(m, n, k, dev), m, n, dev);
  const int64_t rows = internal_slice_rows(m, n, k, dev);
  return rows > 0 ? sliced_workspace_bytes(m, n, k, rows, dev) : linear_large_m_workspace_bytes(m, n, k, dev);
}

int64_t linear_decode_gemm_rows(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  // SM86 2K gates: cuBLAS 13.1 selection cliffs bound the measured fused-MMA win.
  if (dev.sm_arch == 86 && k == 2048 && (n == 64 || n == 128) && m >= 2932 && m <= 4779) return 0;
  // On SM80, medium-width long-K prefill tiles repeat the NF12 decode enough
  // times that one call-local decode followed by BF16 GEMM is faster. The
  // 384/512 by 5120 band was measured on an SM80 PG506-230 at M=128/256/512.
  if (dev.sm_arch == 80 && m >= 128 && m < 1024 && n >= 384 && n <= 512 && k == 5120)
    return slice_rows(n, k, kExternalSliceBytes);
  const SliceRoute r = slice_route(dev);
  const int64_t narrow_min_m = dev.sm_arch == 86 && k >= 4096 ? 49 : r.narrow_min_m;
  const bool route = n <= narrow_n(dev) ? !use_narrow(m, n, k, dev) &&
                                         m >= std::min(narrow_min_m,
                                                       use_small_m(m, n, k, dev) ? kNever : r.external_min_m)
                                   : !use_small_m(m, n, k, dev) && m >= r.external_min_m;
  return route ? slice_rows(n, k, kExternalSliceBytes) : 0;
}

cudaError_t linear(const LinearArgs& a) {
  if (a.dev == nullptr) return cudaErrorInvalidValue;
  if (a.dev->sm_arch < 80) return cudaErrorNotSupported;
  if (a.m < 1 || a.n <= 0 || a.k <= 0 || a.n % 16 != 0 || a.k % 64 != 0) return cudaErrorInvalidValue;
  if (a.n > INT32_MAX || a.k > INT32_MAX || a.m > INT32_MAX) return cudaErrorInvalidValue;
  if (a.x == nullptr || a.y == nullptr || a.weight == nullptr) return cudaErrorInvalidValue;
  // Rows of x and y are moved in 16-byte vectors.
  if (a.ldx < a.k || a.ldx % 8 != 0 || a.ldy < a.n || a.ldy % 8 != 0) return cudaErrorInvalidValue;
  if (!aligned16(a.x) || !aligned16(a.y) || !aligned16(a.weight)) return cudaErrorInvalidValue;
  if (a.bias != nullptr && !aligned16(a.bias)) return cudaErrorInvalidValue;
  if (a.source == WeightSource::kNF12 && (a.exp_base > static_cast<uint32_t>(kMaxExpBase) ||
                                          (a.escapes != nullptr && !aligned16(a.escapes)) ||
                                          (a.num_escapes > 0 && a.escapes == nullptr)))
    return cudaErrorInvalidValue;
  const DeviceInfo& dev = *a.dev;
  const size_t ws = linear_workspace_bytes(a.m, a.n, a.k, dev);
  if (a.workspace_bytes < ws || (ws > 0 && a.workspace == nullptr)) return cudaErrorInvalidValue;

  if (use_narrow(a.m, a.n, a.k, dev))
    return narrow_sm8x_supported(dev) ? linear_large_m_sm8x_narrow(a) : linear_narrow_sm120(a, -1);
  if (use_small_m(a.m, a.n, a.k, dev)) {
    const SmallMConfig cfg = small_m_heuristic(a.m, a.n, a.k, dev);
    return cfg.mt == 0 ? cudaErrorNotSupported : linear_small_m(a, cfg);
  }
  const int64_t rows = internal_slice_rows(a.m, a.n, a.k, dev);
  return rows > 0 ? linear_sliced(a, rows) : linear_large_m(a);
}

}  // namespace nf12::detail
