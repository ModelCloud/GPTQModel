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
#include "fused_decode.h"

#include <cuda_bf16.h>
#include <cooperative_groups.h>

#include <atomic>
#include <climits>
#include <cmath>
#include <cstddef>
#include <mutex>

#include "linear/small_m_core.cuh"
#include "moe/common.cuh"

namespace nf12::torch_detail {
namespace {
using namespace nf12::detail;

constexpr int kSlice = 8;
constexpr int kXStride = kSlice * 128 + 16;
constexpr int kMaxDevices = 64;

struct GateParams {
  FusedDecodeArgs a;
  int tiles;
  int kb;
  int inter_blocks;
  int64_t expert_bytes;
  bool aligned_x;
};

// One CTA owns one top-k slot and one gate/up column tile. Both weight
// representations share the MMA chains, ordered reduction and BF16 boundaries.
template <WeightSource S, int Parts, int Stages>
__global__ void __launch_bounds__(256) gate_kernel(GateParams p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using T = WeightTraits<S>;
  constexpr int Warps = 8;
  constexpr int Rows = Warps / Parts;
  constexpr int HalfRows = Rows / 2;
  constexpr int Columns = HalfRows * 16;
  constexpr int PartialStride = Rows * 16 + 4;
  constexpr bool Bulk = Stages > 0;
  constexpr int AtomsPerPart = kSlice / Parts;
  constexpr int StepsPerSlice = AtomsPerPart / kSmallMBulkAtoms;
  constexpr int StageBytes = kSmallMBulkAtoms * T::kSuperAtomBytes;
  constexpr int Slots = Bulk ? Stages : 1;
  static_assert(Rows % 2 == 0 && kSlice % (2 * Parts) == 0);
  extern __shared__ __align__(128) uint8_t wring[];
  union Scratch {
    uint8_t x[2 * kXStride];
    float partial[Parts * PartialStride];
  };
  __shared__ __align__(16) Scratch scratch;
  __shared__ __align__(16) __nv_bfloat16 rounded[Rows * 16];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int assignment = blockIdx.x / p.tiles;
  const int tile = blockIdx.x % p.tiles;
  const int expert = __ldg(p.a.ids + assignment);
  const int64_t inter = p.a.w13.n / 2;
  auto* out = static_cast<__nv_bfloat16*>(p.a.activation);
  if (expert < 0 || int64_t{expert} >= p.a.w13.experts) {
    for (int c = threadIdx.x; c < Columns; c += Warps * 32) {
      const int64_t n = int64_t{tile} * Columns + c;
      if (n < inter) out[int64_t{assignment} * p.a.ldactivation + n] = __float2bfloat16_rn(0.0f);
    }
    return;
  }

  const int rw = warp % Rows;
  const int part = warp / Rows;
  const int nb = min(tile * HalfRows + rw % HalfRows, p.inter_blocks - 1) +
                 (rw / HalfRows) * p.inter_blocks;
  const auto codec = moe_expert_codec<S>(p.a.w13.exp_base, static_cast<const uint4*>(p.a.w13.escapes),
                                        p.a.w13.escape_offsets, expert);
  const auto* weights = static_cast<const uint8_t*>(p.a.w13.packed);
  const uint8_t* wrow = weights + int64_t{expert} * p.expert_bytes +
                        int64_t{nb} * p.kb * T::kSuperAtomBytes;
  const uint8_t* wp = wrow + min(part, p.kb - 1) * T::kSuperAtomBytes + lane * 16;
  const int wcnt = Bulk ? (p.kb / kSlice) * AtomsPerPart +
                             min(AtomsPerPart, max(0, p.kb % kSlice - part * AtomsPerPart))
                       : p.kb > part ? (p.kb - part + Parts - 1) / Parts : 0;
  const int nstages = (wcnt + kSmallMBulkAtoms - 1) / kSmallMBulkAtoms;
  const uint32_t wslots = smem_addr(wring) + warp * Slots * StageBytes;
  const uint32_t wbars = smem_addr(wring) + Warps * Slots * StageBytes + warp * Slots * 8;
  auto fill_weights = [&](int j) {
    const int u = (j / StepsPerSlice) * kSlice + part * AtomsPerPart +
                  (j % StepsPerSlice) * kSmallMBulkAtoms;
    const int atoms = min(kSmallMBulkAtoms, p.kb - u);
    bulk_load_stream(wslots + (j % Slots) * StageBytes,
                     wrow + int64_t{u} * T::kSuperAtomBytes,
                     atoms * T::kSuperAtomBytes, wbars + (j % Slots) * 8);
  };
  uint4 ring[2][T::kLoads];
  if constexpr (Bulk) {
    if (lane == 0) {
      for (int d = 0; d < Slots; ++d) ring_bar_init(wbars + d * 8, 1);
      ring_bar_fence_init();
      for (int j = 0; j < Slots && j < nstages; ++j) fill_weights(j);
    }
    __syncwarp();
  } else {
#pragma unroll
    for (int d = 0; d < 2; ++d) {
      if (d < wcnt) {
#pragma unroll
        for (int l = 0; l < T::kLoads; ++l)
          ring[d][l] = ldg_stream(wp + d * (Parts * T::kSuperAtomBytes) + l * nf12::kPlaneBytes);
      }
    }
  }

  const uint32_t sbase = smem_addr(scratch.x);
  const int nslices = (p.kb + kSlice - 1) / kSlice;
  auto stage = [&](int s) {
    const int u0 = s * kSlice;
    const int chunks = min(kSlice, p.kb - u0) * 8;
    const int slot = (s & 1) * kXStride;
    const auto* x = static_cast<const __nv_bfloat16*>(p.a.x);
    if (p.aligned_x) {
      for (int c = threadIdx.x; c < chunks; c += Warps * 32)
        cp_async_16(sbase + slot + c * 16, x + int64_t{u0} * 64 + c * 8);
    } else {
      for (int c = threadIdx.x; c < chunks * 8; c += Warps * 32)
        *reinterpret_cast<__nv_bfloat16*>(scratch.x + slot + c * 2) = x[int64_t{u0} * 64 + c];
    }
    cp_async_commit();
  };
  stage(0);
  if (nslices > 1) stage(1);
  const uint32_t xaddr = sbase + (lane >> 3) * 16;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < nslices; ++s) {
    if (s + 1 < nslices) cp_async_wait<1>();
    else cp_async_wait<0>();
    __syncthreads();
    const int u0 = s * kSlice;
    const int u1 = min(p.kb, u0 + kSlice);
    if constexpr (Bulk) {
      const int end = min(u1, u0 + (part + 1) * AtomsPerPart);
      for (int u = u0 + part * AtomsPerPart; u < end; u += kSmallMBulkAtoms) {
        const int j = s * StepsPerSlice + (u - u0 - part * AtomsPerPart) / kSmallMBulkAtoms;
        ring_bar_wait(wbars + (j % Slots) * 8, (j / Slots) & 1);
        const uint32_t src = wslots + (j % Slots) * StageBytes + lane * 16;
#pragma unroll
        for (int d = 0; d < kSmallMBulkAtoms; ++d) {
          if (u + d < end) {
            uint4 planes[T::kLoads];
#pragma unroll
            for (int l = 0; l < T::kLoads; ++l)
              planes[l] = ld_shared_v4(src + d * T::kSuperAtomBytes + l * nf12::kPlaneBytes);
            uint32_t a[4][4];
            T::fragments(planes, codec.base_x4, codec.escapes, a);
            const uint32_t xoff = (s & 1) * kXStride + (u + d - u0) * 128;
            uint32_t b[8];
            ldmatrix_x4(*reinterpret_cast<uint32_t (*)[4]>(&b[0]), xaddr + xoff);
            ldmatrix_x4(*reinterpret_cast<uint32_t (*)[4]>(&b[4]), xaddr + xoff + 64);
#pragma unroll
            for (int q = 0; q < 4; ++q) mma_bf16_16816(acc, a[q], b[2 * q], b[2 * q + 1]);
          }
        }
        // Retire every reader before refilling this asynchronous stage.
        __syncwarp();
        if (lane == 0 && j + Slots < nstages) fill_weights(j + Slots);
      }
    } else {
      const int j0 = u0 / Parts;
      const int j1 = (u1 - part + Parts - 1) / Parts;
      for (int j = j0; j < j1; j += 2) {
#pragma unroll
        for (int d = 0; d < 2; ++d) {
          if (j + d < j1) {
            uint32_t a[4][4];
            T::fragments(ring[d], codec.base_x4, codec.escapes, a);
            if (j + d + 2 < wcnt) {
              const uint8_t* src = wp + int64_t{j + d + 2} * (Parts * T::kSuperAtomBytes);
#pragma unroll
              for (int l = 0; l < T::kLoads; ++l) ring[d][l] = ldg_stream(src + l * nf12::kPlaneBytes);
            }
            const uint32_t xoff = (s & 1) * kXStride + (part + (j + d) * Parts - u0) * 128;
            uint32_t b[8];
            ldmatrix_x4(*reinterpret_cast<uint32_t (*)[4]>(&b[0]), xaddr + xoff);
            ldmatrix_x4(*reinterpret_cast<uint32_t (*)[4]>(&b[4]), xaddr + xoff + 64);
#pragma unroll
            for (int q = 0; q < 4; ++q) mma_bf16_16816(acc, a[q], b[2 * q], b[2 * q + 1]);
          }
        }
      }
    }
    if (s + 2 < nslices) {
      __syncthreads();
      stage(s + 2);
    }
  }
  if constexpr (Bulk) {
    // All fills have completed; invalidate even initialized but unused slots.
    __syncwarp();
    if (lane == 0) {
      for (int d = 0; d < Slots; ++d)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(wbars + d * 8) : "memory");
    }
  }

  // Every warp retires x before reusing the union for ordered K partials.
  __syncthreads();
  const int g = lane >> 2;
  if ((lane & 3) == 0) {
    float* mine = scratch.partial + part * PartialStride;
    mine[rw * 16 + g] = acc[0];
    mine[rw * 16 + g + 8] = acc[2];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < Rows * 16; i += Warps * 32) {
    float sum = scratch.partial[i];
#pragma unroll
    for (int q = 1; q < Parts; ++q) sum = __fadd_rn(sum, scratch.partial[q * PartialStride + i]);
    rounded[i] = __float2bfloat16_rn(sum);
  }
  __syncthreads();
  for (int c = threadIdx.x; c < Columns; c += Warps * 32) {
    const int64_t n = int64_t{tile} * Columns + c;
    if (n < inter) {
      const float gate = __bfloat162float(rounded[c]);
      const float up = __bfloat162float(rounded[Columns + c]);
      const float silu = __fdiv_rn(gate, __fadd_rn(1.0f, expf(-gate)));
      out[int64_t{assignment} * p.a.ldactivation + n] = __float2bfloat16_rn(__fmul_rn(silu, up));
    }
  }
#else
  __trap();
#endif
}

struct DownParams {
  FusedDecodeArgs a;
  int64_t expert_bytes;
};

constexpr int down_core_bytes(int parts) {
  const int partials = parts * small_m_partial_stride(4 / parts) * sizeof(float);
  return 2 * kXStride > partials ? 2 * kXStride : partials;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ float round_assignment(float sum, float scale) {
  return __bfloat162float(__float2bfloat16_rn(__fmul_rn(sum, scale)));
}
#endif

// Each cluster rank owns a top-k slot, including repeated or dropped IDs.
template <WeightSource S, int Stages, int Parts>
__global__ void __launch_bounds__(128) down_kernel(DownParams p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using T = WeightTraits<S>;
  constexpr int Warps = 4;
  constexpr int Rows = Warps / Parts;
  constexpr int Pairs = Rows * 8;
  static_assert(Parts == 1 || Stages == 0);
  static_assert(kSlice % (Parts * small_m_step(Stages)) == 0);
  extern __shared__ __align__(128) uint8_t shared[];
  uint8_t* region = shared + small_m_ring_bytes(Stages, Warps, T::kSuperAtomBytes);
  float* partial = reinterpret_cast<float*>(region);
  const auto& a = p.a;
  const int topk = static_cast<int>(a.topk);
  const int rank = blockIdx.x % topk;
  const int group = blockIdx.x / topk;
  const int n0 = group * Rows * 16;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int kpart = warp / Rows;
  const int kb = static_cast<int>(a.w2.k / 64);
  const int nb = min(group * Rows + warp % Rows, static_cast<int>(a.w2.n / 16) - 1);
  const int expert = __ldg(a.ids + rank);
  if (expert >= 0 && int64_t{expert} < a.w2.experts) {
    const auto codec = moe_expert_codec<S>(a.w2.exp_base, static_cast<const uint4*>(a.w2.escapes),
                                          a.w2.escape_offsets, expert);
    const uint8_t* wp = static_cast<const uint8_t*>(a.w2.packed) + int64_t{expert} * p.expert_bytes +
                        (int64_t{nb} * kb + kpart) * T::kSuperAtomBytes + lane * 16;
    const auto* xrow = static_cast<const __nv_bfloat16*>(a.activation) + int64_t{rank} * a.ldactivation;
    const float scale = __ldg(a.scales + rank);
    // kc=1: never reduce different experts as K partials. The callback writes
    // each pair over its own retired core partials, with no cross-pair alias.
    small_m_core<S, 1, Warps, Stages, Parts>(
        region, shared, wp, kb, codec.base_x4, codec.escapes, 1, kSlice, kXStride, 1, 0,
        [&](int) { return xrow; }, [] {},
        [&](int, int rp, float2 sum) {
          partial[2 * rp] = round_assignment(sum.x, scale);
          partial[2 * rp + 1] = round_assignment(sum.y, scale);
        });
    if constexpr (Stages > 0) {
      if (lane == 0) {
        const uint32_t bars = smem_addr(shared) +
                              Warps * Stages * 2 * T::kSuperAtomBytes + warp * Stages * 8;
        for (int s = 0; s < Stages; ++s)
          asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bars + s * 8) : "memory");
      }
    }
  } else if (tid < Pairs) {
    // Do not read activation, weights or scales for a dropped assignment.
    partial[2 * tid] = 0.0f;
    partial[2 * tid + 1] = 0.0f;
  }
  const auto cluster = cooperative_groups::this_cluster();
  cluster.sync();
  const int begin = rank * Pairs / topk;
  const int end = (rank + 1) * Pairs / topk;
  for (int rp = begin + tid; rp < end; rp += Warps * 32) {
    float2 values[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < topk) {
        const float* remote = cluster.map_shared_rank(partial, j);
        values[j] = *reinterpret_cast<const float2*>(remote + 2 * rp);
      }
    }
    float2 combined = make_float2(-0.0f, -0.0f);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < topk) {
        combined.x = __fadd_rn(combined.x, values[j].x);
        combined.y = __fadd_rn(combined.y, values[j].y);
      }
    }
    const int n = n0 + 2 * rp;
    if (n < a.w2.n) {
      auto* dst = static_cast<__nv_bfloat16*>(a.out) + n;
      dst[0] = __float2bfloat16_rn(combined.x);
      dst[1] = __float2bfloat16_rn(combined.y);
    }
  }
  // All DSM readers must retire before any rank releases shared storage.
  cluster.sync();
#else
  __trap();
#endif
}

using GateKernel = void (*)(GateParams);
using DownKernel = void (*)(DownParams);
enum class GateVariant { Register4, Bulk2, Bulk4 };
enum class DownVariant { Register1, Bulk1, Register4 };

template <WeightSource S>
GateKernel gate_for(GateVariant v) {
  switch (v) {
    case GateVariant::Register4: return gate_kernel<S, 4, 0>;
    case GateVariant::Bulk2: return gate_kernel<S, 2, 2>;
    case GateVariant::Bulk4: return gate_kernel<S, 4, 2>;
  }
  return nullptr;
}

template <WeightSource S>
DownKernel down_for(DownVariant v) {
  switch (v) {
    case DownVariant::Register1: return down_kernel<S, 0, 1>;
    case DownVariant::Bulk1: return down_kernel<S, 2, 1>;
    case DownVariant::Register4: return down_kernel<S, 0, 4>;
  }
  return nullptr;
}

bool aligned(const void* p, uintptr_t alignment) {
  return (reinterpret_cast<uintptr_t>(p) & (alignment - 1)) == 0;
}

bool fits_span(const void* p, int64_t rows, int64_t stride, int64_t width, int bytes) {
  const int64_t limit = INT64_MAX / bytes;
  if (rows < 1 || width < 1 || width > limit || stride < width ||
      rows - 1 > (limit - width) / stride)
    return false;
  const uint64_t span = static_cast<uint64_t>((rows - 1) * stride + width) * bytes;
  return span <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(p);
}

int atom_bytes(const FusedDecodeWeight& w) {
  return w.bf16t ? nf12::kSuperAtomBytesBf16t : nf12::kSuperAtomBytes;
}

int64_t expert_bytes(const FusedDecodeWeight& w) {
  return (w.n / 16) * (w.k / 64) * atom_bytes(w);
}

bool valid_weight(const FusedDecodeWeight& w) {
  if (!w.packed || !aligned(w.packed, 16) || w.experts < 1 || w.experts > INT_MAX ||
      w.n < 16 || w.n > 4096 || w.n % 16 || w.k < 64 || w.k > 4096 || w.k % 64)
    return false;
  if (!w.bf16t && (!w.exp_base || !aligned(w.escapes, 16) || !aligned(w.escape_offsets, 8) ||
                   (w.escapes && !w.escape_offsets)))
    return false;
  const int64_t bytes = expert_bytes(w);
  return fits_span(w.packed, w.experts, bytes, bytes, 1);
}

bool valid_args(const FusedDecodeArgs& a) {
  if (a.topk < 1 || a.topk > 8 || a.w13.k < 1024 || a.w13.k > 4096 ||
      a.w2.n < 1024 || a.w2.n > 4096 || a.w2.k < 512 || a.w2.k > 2048 ||
      a.w13.n != 2 * a.w2.k || a.w13.experts != a.w2.experts ||
      !valid_weight(a.w13) || !valid_weight(a.w2))
    return false;
  if (!a.x || !a.ids || !a.scales || !aligned(a.x, 2) || !aligned(a.ids, 4) ||
      !aligned(a.scales, 4) || !aligned(a.activation, 16) || !aligned(a.out, 2) ||
      a.ldactivation % 8)
    return false;
  return fits_span(a.x, 1, a.ldx, a.w13.k, 2) &&
         fits_span(a.ids, 1, a.ldids, a.topk, 4) &&
         fits_span(a.scales, 1, a.ldscales, a.topk, 4) &&
         fits_span(a.activation, a.topk, a.ldactivation, a.w2.k, 2) &&
         fits_span(a.out, 1, a.ldout, a.w2.n, 2);
}

// Fixed host storage: no allocations, device synchronization or hidden calls
// into libnf12's device cache. Cold-path locks only protect cache publication.
struct DeviceCache {
  std::mutex mutex;
  std::atomic<bool> initialized{false};
  cudaDeviceProp properties{};
  bool supported = false;
  std::atomic<int8_t> gate[2][3]{};
  std::atomic<int8_t> down[2][3]{};
  std::atomic<int8_t> cluster[2][3][8]{};
};
DeviceCache caches[kMaxDevices];

cudaError_t device_ready(int device, DeviceCache& cache) {
  if (cache.initialized.load(std::memory_order_acquire))
    return cache.supported ? cudaSuccess : cudaErrorNotSupported;
  std::lock_guard<std::mutex> lock(cache.mutex);
  if (!cache.initialized.load(std::memory_order_relaxed)) {
    cudaDeviceProp prop{};
    cudaError_t status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) return status;
    const int arch = prop.major * 10 + prop.minor;
    int cluster = 0;
    if (arch == 90 || arch == 100 || arch == 120) {
      status = cudaDeviceGetAttribute(&cluster, cudaDevAttrClusterLaunch, device);
      if (status != cudaSuccess) return status;
    }
    cache.properties = prop;
    cache.supported = (arch == 90 || arch == 100 || arch == 120) &&
                      prop.warpSize == 32 && prop.maxThreadsPerBlock >= 256 && cluster;
    cache.initialized.store(true, std::memory_order_release);
  }
  return cache.supported ? cudaSuccess : cudaErrorNotSupported;
}

cudaError_t kernel_ready(DeviceCache& cache, std::atomic<int8_t>& state,
                         const void* kernel, int threads, size_t dynamic_bytes) {
  int8_t ready = state.load(std::memory_order_acquire);
  if (ready) return ready > 0 ? cudaSuccess : cudaErrorNotSupported;
  std::lock_guard<std::mutex> lock(cache.mutex);
  ready = state.load(std::memory_order_relaxed);
  if (ready) return ready > 0 ? cudaSuccess : cudaErrorNotSupported;
  cudaFuncAttributes attr{};
  cudaError_t status = cudaFuncGetAttributes(&attr, kernel);
  if (status == cudaErrorNoKernelImageForDevice || status == cudaErrorInvalidDeviceFunction) {
    cudaGetLastError();
    state.store(-1, std::memory_order_release);
    return cudaErrorNotSupported;
  }
  if (status != cudaSuccess) return status;
  // Reject a loaded trap-only image, including JIT of pre-SM90 PTX.
  const auto& prop = cache.properties;
  const size_t shared_limit = dynamic_bytes ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock;
  if (attr.binaryVersion < 90 || attr.ptxVersion < 90 || attr.maxThreadsPerBlock < threads ||
      attr.sharedSizeBytes + dynamic_bytes > shared_limit) {
    state.store(-1, std::memory_order_release);
    return cudaErrorNotSupported;
  }
  if (dynamic_bytes > static_cast<size_t>(attr.maxDynamicSharedSizeBytes)) {
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_bytes));
    if (status != cudaSuccess) return status;
  }
  state.store(1, std::memory_order_release);
  return cudaSuccess;
}

void cluster_launch(cudaLaunchConfig_t& launch, cudaLaunchAttribute& attr,
                    int blocks, int topk, size_t smem, cudaStream_t stream) {
  launch = {};
  launch.gridDim = dim3(blocks);
  launch.blockDim = dim3(128);
  launch.dynamicSmemBytes = smem;
  launch.stream = stream;
  attr = {};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = topk;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  launch.attrs = &attr;
  launch.numAttrs = 1;
}

cudaError_t cluster_ready(DeviceCache& cache, std::atomic<int8_t>& state,
                          DownKernel kernel, int topk, size_t smem) {
  int8_t ready = state.load(std::memory_order_acquire);
  if (ready) return ready > 0 ? cudaSuccess : cudaErrorNotSupported;
  std::lock_guard<std::mutex> lock(cache.mutex);
  ready = state.load(std::memory_order_relaxed);
  if (ready) return ready > 0 ? cudaSuccess : cudaErrorNotSupported;
  cudaLaunchConfig_t launch{};
  cudaLaunchAttribute attr{};
  cluster_launch(launch, attr, topk, topk, smem, nullptr);
  int active = 0;
  const cudaError_t status = cudaOccupancyMaxActiveClusters(&active, kernel, &launch);
  if (status == cudaErrorNotSupported || status == cudaErrorInvalidConfiguration || status == cudaErrorInvalidValue) {
    cudaGetLastError();
    state.store(-1, std::memory_order_release);
    return cudaErrorNotSupported;
  }
  if (status != cudaSuccess) return status;
  state.store(active > 0 ? 1 : -1, std::memory_order_release);
  return active > 0 ? cudaSuccess : cudaErrorNotSupported;
}

struct LaunchPlan {
  GateKernel gate;
  DownKernel down;
  size_t gate_smem, down_smem;
  int gate_tiles, down_groups;
};

cudaError_t prepare(int device, const FusedDecodeArgs& a, LaunchPlan& plan) {
  if (device < 0 || device >= kMaxDevices || !valid_args(a)) return cudaErrorNotSupported;
  DeviceCache& cache = caches[device];
  cudaError_t status = device_ready(device, cache);
  if (status != cudaSuccess) return status;
  const int arch = cache.properties.major * 10 + cache.properties.minor;
  if (a.w13.k > 4096 || a.w2.k > 2048) return cudaErrorNotSupported;
  if (arch == 90 && a.topk == 1 && a.w2.k > 1024) return cudaErrorNotSupported;
  if (arch == 120 && (a.topk < 4 || a.w13.k > 2048 || a.w2.k > 1536))
    return cudaErrorNotSupported;
  const GateVariant gate = arch == 120 ? GateVariant::Register4 :
                           a.w2.k > 1024 ? GateVariant::Bulk2 :
                           arch == 100 ? GateVariant::Bulk4 : GateVariant::Register4;
  const DownVariant down = arch == 120 ? DownVariant::Register1 :
                           arch == 100 && a.w2.k > 1024 ? DownVariant::Register4 : DownVariant::Bulk1;
  const int gate_parts = gate == GateVariant::Bulk2 ? 2 : 4;
  const int gate_stages = gate == GateVariant::Register4 ? 0 : 2;
  const int down_parts = down == DownVariant::Register4 ? 4 : 1;
  const int down_stages = down == DownVariant::Bulk1 ? 2 : 0;
  plan.gate = a.w13.bf16t ? gate_for<WeightSource::kBF16T>(gate) : gate_for<WeightSource::kNF12>(gate);
  plan.down = a.w2.bf16t ? down_for<WeightSource::kBF16T>(down) : down_for<WeightSource::kNF12>(down);
  plan.gate_smem = small_m_ring_bytes(gate_stages, 8, atom_bytes(a.w13));
  plan.down_smem = small_m_ring_bytes(down_stages, 4, atom_bytes(a.w2)) + down_core_bytes(down_parts);
  const int columns = 8 / gate_parts / 2 * 16;
  plan.gate_tiles = static_cast<int>((a.w2.k + columns - 1) / columns);
  const int rows = 4 / down_parts;
  plan.down_groups = static_cast<int>((a.w2.n / 16 + rows - 1) / rows);
  if (plan.gate_tiles * a.topk > cache.properties.maxGridSize[0] ||
      plan.down_groups * a.topk > cache.properties.maxGridSize[0])
    return cudaErrorNotSupported;
  status = kernel_ready(cache, cache.gate[a.w13.bf16t][static_cast<int>(gate)],
                        reinterpret_cast<const void*>(plan.gate), 256, plan.gate_smem);
  if (status != cudaSuccess) return status;
  status = kernel_ready(cache, cache.down[a.w2.bf16t][static_cast<int>(down)],
                        reinterpret_cast<const void*>(plan.down), 128, plan.down_smem);
  if (status != cudaSuccess) return status;
  return cluster_ready(cache, cache.cluster[a.w2.bf16t][static_cast<int>(down)][a.topk - 1],
                       plan.down, static_cast<int>(a.topk), plan.down_smem);
}
}  // namespace

bool fused_decode_supported(int device, const FusedDecodeArgs& args) {
  LaunchPlan plan{};
  const cudaError_t status = prepare(device, args, plan);
  if (status != cudaSuccess && status != cudaErrorNotSupported) cudaGetLastError();
  return status == cudaSuccess;
}

cudaError_t fused_decode(const FusedDecodeArgs& args, cudaStream_t stream) {
  if (!args.activation || !args.out) return cudaErrorInvalidValue;
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  LaunchPlan plan{};
  status = prepare(device, args, plan);
  if (status != cudaSuccess) return status;

  GateParams gate{args, plan.gate_tiles, static_cast<int>(args.w13.k / 64),
                  static_cast<int>(args.w13.n / 32), expert_bytes(args.w13), aligned(args.x, 16)};
  void* gate_params[] = {&gate};
  status = cudaLaunchKernel(reinterpret_cast<const void*>(plan.gate),
                            dim3(static_cast<unsigned>(plan.gate_tiles * args.topk)), dim3(256),
                            gate_params, plan.gate_smem, stream);
  if (status != cudaSuccess) return status;

  DownParams down{args, expert_bytes(args.w2)};
  cudaLaunchConfig_t launch{};
  cudaLaunchAttribute attr{};
  cluster_launch(launch, attr, static_cast<int>(plan.down_groups * args.topk),
                  static_cast<int>(args.topk), plan.down_smem, stream);
  return cudaLaunchKernelEx(&launch, plan.down, down);
}

}  // namespace nf12::torch_detail
