// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0

/******************************************************************************
 * Adapted from https://github.com/z-lab/paroquant
 ******************************************************************************/

#include "rotation.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>

template <typename scalar_t, int CTA_M, int GROUP_SIZE, int KROT, bool USE_SCALE, int ROW_PAD>
__global__ void rotate_kernel(const scalar_t *__restrict__ x, scalar_t *__restrict__ out,
                              const int16_t *__restrict__ idx_ij, const scalar_t *__restrict__ theta,
                              const scalar_t *__restrict__ scales, int s, int h) {
  constexpr int ROW_STRIDE = CTA_M + ROW_PAD;
  __shared__ scalar_t x_grp[ROW_STRIDE * GROUP_SIZE];

  int j = blockIdx.x;
  int g = blockIdx.y;
  int t = threadIdx.x;

  RotateAccess<scalar_t>::template load_group<CTA_M, ROW_STRIDE, GROUP_SIZE, USE_SCALE>(
      x_grp, x, scales, s, h, j, g, t);

  float reg_theta[KROT];
  int reg_idx[KROT];
  RotateAccess<scalar_t>::template load_coeffs<KROT, GROUP_SIZE>(reg_theta, reg_idx, idx_ij, theta,
                                                                 h, g, t);
  __syncthreads();

#pragma unroll
  for (int r = 0; r < KROT; r++) {
    RotateAccess<scalar_t>::template apply_one<CTA_M, ROW_STRIDE>(x_grp, reg_idx[r], reg_theta[r]);
    __syncthreads();
  }

  RotateAccess<scalar_t>::template store_group<CTA_M, ROW_STRIDE, GROUP_SIZE>(out, x_grp, s, h, j,
                                                                               g, t);
}

template <int CTA_M, int GROUP_SIZE, int KROT, bool USE_SCALE, int ROW_PAD>
__global__ void rotate_kernel_bf16_half_workspace(const __nv_bfloat16 *__restrict__ x,
                                                  __nv_bfloat16 *__restrict__ out,
                                                  const int16_t *__restrict__ idx_ij,
                                                  const __nv_bfloat16 *__restrict__ theta,
                                                  const __nv_bfloat16 *__restrict__ scales, int s,
                                                  int h) {
  constexpr int ROW_STRIDE = CTA_M + ROW_PAD;
  __shared__ __half x_grp[ROW_STRIDE * GROUP_SIZE];

  int j = blockIdx.x;
  int g = blockIdx.y;
  int t = threadIdx.x;

  RotateAccessBFloat16HalfWorkspace::template load_group<CTA_M, ROW_STRIDE, GROUP_SIZE, USE_SCALE>(
      x_grp, x, scales, s, h, j, g, t);

  float reg_theta[KROT];
  int reg_idx[KROT];
  RotateAccessBFloat16HalfWorkspace::template load_coeffs<KROT, GROUP_SIZE>(reg_theta, reg_idx,
                                                                            idx_ij, theta, h, g,
                                                                            t);
  __syncthreads();

#pragma unroll
  for (int r = 0; r < KROT; r++) {
    RotateAccess<__half>::template apply_one<CTA_M, ROW_STRIDE>(x_grp, reg_idx[r], reg_theta[r]);
    __syncthreads();
  }

  RotateAccessBFloat16HalfWorkspace::template store_group<CTA_M, ROW_STRIDE, GROUP_SIZE>(
      out, x_grp, s, h, j, g, t);
}

#define LAUNCH_ROTATE(CUDA_T, TORCH_T)                                                               \
  {                                                                                                  \
    auto *x_p = reinterpret_cast<CUDA_T *>(x.data_ptr<TORCH_T>());                                   \
    auto *o_p = reinterpret_cast<CUDA_T *>(out.data_ptr<TORCH_T>());                                 \
    auto *t_p = reinterpret_cast<CUDA_T *>(theta_cast.data_ptr<TORCH_T>());                          \
    if (has_scale) {                                                                                 \
      auto *s_p = reinterpret_cast<CUDA_T *>(scales_cast.data_ptr<TORCH_T>());                       \
      rotate_kernel<CUDA_T, CTA_M, GROUP_SIZE, KROT, true, ROW_PAD><<<grid, block, 0, stream>>>(    \
          x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, s_p, seq_len, h);                               \
    } else {                                                                                         \
      rotate_kernel<CUDA_T, CTA_M, GROUP_SIZE, KROT, false, ROW_PAD><<<grid, block, 0, stream>>>(   \
          x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, nullptr, seq_len, h);                           \
    }                                                                                                \
    break;                                                                                           \
  }

template <int KROT, int CTA_M, int GROUP_SIZE, int ROW_PAD>
torch::Tensor rotate_launcher_bf16_half_workspace(at::Tensor x, at::Tensor idx_ij,
                                                  at::Tensor theta, at::Tensor scales) {
  int h = x.size(-1);
  TORCH_CHECK(h % GROUP_SIZE == 0, "h must be divisible by GROUP_SIZE");
  int groups_per_row = h / GROUP_SIZE;
  constexpr int pn = GROUP_SIZE / 2;
  int seq_len = x.numel() / x.size(-1);
  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  at::Tensor out = torch::empty(x.sizes(), options);
  bool has_scale = scales.defined() && scales.numel() > 0;

  auto theta_cast = theta.scalar_type() == at::kBFloat16 ? theta : theta.to(at::kBFloat16);
  auto scales_cast = !has_scale                       ? at::Tensor()
                     : scales.scalar_type() == at::kBFloat16 ? scales
                                                             : scales.to(at::kBFloat16);

  dim3 grid((seq_len + CTA_M - 1) / CTA_M, groups_per_row);
  dim3 block(pn);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto *x_p = reinterpret_cast<__nv_bfloat16 *>(x.data_ptr<c10::BFloat16>());
  auto *o_p = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<c10::BFloat16>());
  auto *t_p = reinterpret_cast<__nv_bfloat16 *>(theta_cast.data_ptr<c10::BFloat16>());
  if (has_scale) {
    auto *s_p = reinterpret_cast<__nv_bfloat16 *>(scales_cast.data_ptr<c10::BFloat16>());
    rotate_kernel_bf16_half_workspace<CTA_M, GROUP_SIZE, KROT, true, ROW_PAD><<<grid, block, 0, stream>>>(
        x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, s_p, seq_len, h);
  } else {
    rotate_kernel_bf16_half_workspace<CTA_M, GROUP_SIZE, KROT, false, ROW_PAD><<<grid, block, 0, stream>>>(
        x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, nullptr, seq_len, h);
  }
  return out;
}

template <int KROT, int CTA_M, int GROUP_SIZE, int ROW_PAD>
torch::Tensor rotate_launcher(at::Tensor x, at::Tensor idx_ij, at::Tensor theta,
                              at::Tensor scales) {
  int h = x.size(-1);
  TORCH_CHECK(h % GROUP_SIZE == 0, "h must be divisible by GROUP_SIZE");
  int groups_per_row = h / GROUP_SIZE;
  constexpr int pn = GROUP_SIZE / 2;
  int seq_len = x.numel() / x.size(-1);
  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  at::Tensor out = torch::empty(x.sizes(), options);
  bool has_scale = scales.defined() && scales.numel() > 0;

  auto dtype = x.scalar_type();
  auto theta_cast = theta.scalar_type() == dtype ? theta : theta.to(x.dtype());
  auto scales_cast = !has_scale                      ? at::Tensor()
                     : scales.scalar_type() == dtype ? scales
                                                     : scales.to(x.dtype());

  dim3 grid((seq_len + CTA_M - 1) / CTA_M, groups_per_row);
  dim3 block(pn);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (dtype) {
  case at::kFloat:
    LAUNCH_ROTATE(float, float)
  case at::kHalf:
    LAUNCH_ROTATE(__half, c10::Half)
  case at::kBFloat16:
    return rotate_launcher_bf16_half_workspace<KROT, CTA_M, GROUP_SIZE, ROW_PAD>(x, idx_ij, theta, scales);
  default:
    TORCH_CHECK(false, "rotate supports Float, Half, and BFloat16, got ", x.scalar_type());
  }
  return out;
}

#undef LAUNCH_ROTATE

#define DISPATCH_ROW_PAD(KROT, CTA, GS)                                                              \
  switch (row_pad) {                                                                                 \
  case 2:                                                                                            \
    return rotate_launcher<KROT, CTA, GS, 2>(x, idx, theta, scales);                                \
  case 0:                                                                                            \
    return rotate_launcher<KROT, CTA, GS, 0>(x, idx, theta, scales);                                \
  default:                                                                                           \
    TORCH_CHECK(false, "Unsupported ROW_PAD = ", row_pad, "; compiled variants: 0 and 2");          \
  }

#define DISPATCH_CTA_M(KROT, GS)                                                                     \
  switch (cta_m) {                                                                                   \
  case 16:                                                                                           \
    DISPATCH_ROW_PAD(KROT, 16, GS)                                                                   \
  case 8:                                                                                            \
    DISPATCH_ROW_PAD(KROT, 8, GS)                                                                    \
  case 4:                                                                                            \
    DISPATCH_ROW_PAD(KROT, 4, GS)                                                                    \
  default:                                                                                           \
    TORCH_CHECK(false, "Unsupported CTA_M = ", cta_m, "; compiled variants: 4, 8, and 16");        \
  }

#define DISPATCH_KROT(GS)                                                                            \
  switch (krot) {                                                                                    \
  case 1:                                                                                            \
    DISPATCH_CTA_M(1, GS)                                                                            \
  case 8:                                                                                            \
    DISPATCH_CTA_M(8, GS)                                                                            \
  default:                                                                                           \
    TORCH_CHECK(false, "Unsupported KROT = ", krot, "; compiled variants: 1 and 8");               \
  }

torch::Tensor dispatch_rotate_variant(at::Tensor x, at::Tensor idx, at::Tensor theta, at::Tensor scales,
                                      int64_t group_size, int64_t krot, int cta_m, int row_pad) {
  if (group_size == 128) {
    DISPATCH_KROT(128)
  }
  TORCH_CHECK(false, "Unsupported group_size: ", group_size, "; expected 128");
}

namespace {

// Resolve launch dimensions in one place so the runtime path can use either a
// fixed explicit launch shape, the legacy default, or a cached autotuned plan.
struct LaunchConfig {
  int cta_m;
  int row_pad;
};

constexpr int kLegacyLaunchSentinel = -1;
constexpr int kAutotuneLaunchSentinel = -2;
constexpr float kAutotuneMinRelativeSpeedup = 0.15f;
// Tiny decode kernels can show large relative swings from timer noise. Require
// a minimum absolute win before overriding the architecture baseline.
constexpr float kAutotuneMinAbsoluteSpeedupMs = 0.01f;
constexpr std::array<LaunchConfig, 6> kAutotuneCandidates = {{
    {4, 0},
    {4, 2},
    {8, 0},
    {8, 2},
    {16, 0},
    {16, 2},
}};

struct AutotuneKey {
  int device_index;
  int scalar_type;
  int seq_len;
  int hidden;
  int group_size;
  int krot;
  bool has_scale;

  bool operator==(const AutotuneKey &other) const {
    return device_index == other.device_index && scalar_type == other.scalar_type &&
           seq_len == other.seq_len && hidden == other.hidden &&
           group_size == other.group_size && krot == other.krot && has_scale == other.has_scale;
  }
};

struct AutotuneKeyHash {
  size_t operator()(const AutotuneKey &key) const noexcept {
    size_t hash = std::hash<int>{}(key.device_index);
    hash ^= std::hash<int>{}(key.scalar_type) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(key.seq_len) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(key.hidden) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(key.group_size) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(key.krot) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<bool>{}(key.has_scale) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

// Process-local cache for the winning launch plan per runtime shape. This is
// intentionally native so the steady-state fused path stays inside one op call.
std::unordered_map<AutotuneKey, LaunchConfig, AutotuneKeyHash> &autotune_cache() {
  static std::unordered_map<AutotuneKey, LaunchConfig, AutotuneKeyHash> cache;
  return cache;
}

std::mutex &autotune_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::mutex &autotune_measurement_mutex() {
  // Serialize cold-shape autotune so free-threaded callers do not benchmark
  // the same or competing launch plans concurrently on cache misses.
  static std::mutex mutex;
  return mutex;
}

int resolve_autotune_count(const char *name, int default_value) {
  if (const char *raw = std::getenv(name)) {
    int parsed = std::atoi(raw);
    if (parsed > 0) {
      return parsed;
    }
  }
  return default_value;
}

float resolve_autotune_target_ms() {
  if (const char *raw = std::getenv("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE_TARGET_MS")) {
    const float parsed = std::atof(raw);
    if (parsed > 0.0f) {
      return parsed;
    }
  }
  return 5.0f;
}

float resolve_autotune_min_relative_speedup() {
  if (const char *raw = std::getenv("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE_MIN_SPEEDUP_PCT")) {
    const float parsed = std::atof(raw);
    if (parsed > 0.0f) {
      return parsed / 100.0f;
    }
  }
  return kAutotuneMinRelativeSpeedup;
}

float resolve_autotune_min_absolute_speedup_seconds() {
  if (const char *raw = std::getenv("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE_MIN_SPEEDUP_US")) {
    const float parsed = std::atof(raw);
    if (parsed > 0.0f) {
      return parsed / 1e6f;
    }
  }
  return kAutotuneMinAbsoluteSpeedupMs / 1e3f;
}

bool requests_autotune(int requested_cta_m, int requested_row_pad) {
  return requested_cta_m == kAutotuneLaunchSentinel || requested_row_pad == kAutotuneLaunchSentinel;
}

AutotuneKey make_autotune_key(const at::Tensor &x, int64_t group_size, int64_t krot, bool has_scale) {
  return {
      x.get_device(),
      static_cast<int>(x.scalar_type()),
      static_cast<int>(x.numel() / x.size(-1)),
      static_cast<int>(x.size(-1)),
      static_cast<int>(group_size),
      static_cast<int>(krot),
      has_scale,
  };
}

std::optional<LaunchConfig> lookup_autotune_cache(const AutotuneKey &key) {
  std::lock_guard<std::mutex> guard(autotune_cache_mutex());
  auto &cache = autotune_cache();
  auto it = cache.find(key);
  if (it == cache.end()) {
    return std::nullopt;
  }
  return it->second;
}

LaunchConfig store_autotune_cache(const AutotuneKey &key, LaunchConfig config) {
  std::lock_guard<std::mutex> guard(autotune_cache_mutex());
  auto &cache = autotune_cache();
  auto [it, inserted] = cache.emplace(key, config);
  if (!inserted) {
    return it->second;
  }
  return config;
}

int current_sm_version() {
  static thread_local c10::DeviceIndex cached_device = -1;
  static thread_local int cached_sm = -1;
  const c10::DeviceIndex current_device = c10::cuda::current_device();
  if (current_device != cached_device) {
    cached_device = current_device;
    const cudaDeviceProp *props = at::cuda::getDeviceProperties(current_device);
    cached_sm = props == nullptr ? -1 : (props->major * 10 + props->minor);
  }
  return cached_sm;
}

int resolve_cta_m(at::ScalarType dtype, int explicit_value) {
  if (explicit_value == 4 || explicit_value == 8 || explicit_value == 16) {
    return explicit_value;
  }

  // These defaults are pinned to manual full-sweep measurements on the A100
  // (sm80) and RTX 4090 (sm89) available on this host. Other architectures may
  // not benefit from these launch shapes and can regress in performance, so
  // they stay on the legacy default until benchmarked explicitly.
  const int sm_version = current_sm_version();
  if (dtype == at::kHalf) {
    switch (sm_version) {
    case 80:
    case 89:
      return 8;
    default:
      return 4;
    }
  }
  if (dtype == at::kBFloat16) {
    switch (sm_version) {
    case 80:
      return 4;
    case 89:
      return 16;
    default:
      return 4;
    }
  }
  return 4;
}

int resolve_row_pad(at::ScalarType dtype, int explicit_value) {
  if (explicit_value == 0 || explicit_value == 2) {
    return explicit_value;
  }
  return dtype == at::kFloat ? 0 : 2;
}

LaunchConfig resolve_legacy_launch_config(at::ScalarType dtype, int explicit_cta_m, int explicit_row_pad) {
  return {
      resolve_cta_m(dtype, explicit_cta_m),
      resolve_row_pad(dtype, explicit_row_pad),
  };
}

float benchmark_launch_config(const at::Tensor &x, const at::Tensor &idx, const at::Tensor &theta,
                              const at::Tensor &scales, int64_t group_size, int64_t krot,
                              LaunchConfig config) {
  const int warmup = resolve_autotune_count("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE_WARMUP", 3);
  const int base_iters = resolve_autotune_count("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE_ITERS", 25);
  const int repeats = resolve_autotune_count("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE_REPEATS", 5);
  const float target_ms = resolve_autotune_target_ms();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto synchronize_stream = [stream]() {
    const cudaError_t status = cudaStreamSynchronize(stream);
    TORCH_CHECK(status == cudaSuccess, "ParoQuant rotation autotune failed to synchronize CUDA stream: ",
                cudaGetErrorString(status));
  };
  auto create_event = []() {
    cudaEvent_t event = nullptr;
    const cudaError_t status = cudaEventCreate(&event);
    TORCH_CHECK(status == cudaSuccess, "ParoQuant rotation autotune failed to create CUDA event: ",
                cudaGetErrorString(status));
    return event;
  };
  auto destroy_event = [](cudaEvent_t event) {
    if (event == nullptr) {
      return;
    }
    const cudaError_t status = cudaEventDestroy(event);
    TORCH_CHECK(status == cudaSuccess, "ParoQuant rotation autotune failed to destroy CUDA event: ",
                cudaGetErrorString(status));
  };
  auto elapsed_ms_for_iterations = [&](int iters) {
    cudaEvent_t start = create_event();
    cudaEvent_t end = create_event();
    const auto cleanup = [&]() {
      destroy_event(end);
      destroy_event(start);
    };

    const cudaError_t start_status = cudaEventRecord(start, stream);
    TORCH_CHECK(start_status == cudaSuccess,
                "ParoQuant rotation autotune failed to record CUDA start event: ",
                cudaGetErrorString(start_status));
    for (int i = 0; i < iters; ++i) {
      auto out = dispatch_rotate_variant(x, idx, theta, scales, group_size, krot, config.cta_m,
                                         config.row_pad);
      (void)out;
    }
    const cudaError_t end_status = cudaEventRecord(end, stream);
    TORCH_CHECK(end_status == cudaSuccess, "ParoQuant rotation autotune failed to record CUDA end event: ",
                cudaGetErrorString(end_status));
    const cudaError_t sync_status = cudaEventSynchronize(end);
    TORCH_CHECK(sync_status == cudaSuccess,
                "ParoQuant rotation autotune failed to synchronize CUDA end event: ",
                cudaGetErrorString(sync_status));
    float elapsed_ms = 0.0f;
    const cudaError_t elapsed_status = cudaEventElapsedTime(&elapsed_ms, start, end);
    TORCH_CHECK(elapsed_status == cudaSuccess,
                "ParoQuant rotation autotune failed to read CUDA elapsed time: ",
                cudaGetErrorString(elapsed_status));
    cleanup();
    return elapsed_ms;
  };

  for (int i = 0; i < warmup; ++i) {
    auto out = dispatch_rotate_variant(x, idx, theta, scales, group_size, krot, config.cta_m,
                                       config.row_pad);
    (void)out;
  }
  synchronize_stream();

  const float pilot_ms = elapsed_ms_for_iterations(base_iters);
  const float min_total_ms = std::max(target_ms, 1.0f);
  int iters = base_iters;
  if (pilot_ms > 0.0f && pilot_ms < min_total_ms) {
    const float scale = min_total_ms / pilot_ms;
    iters = std::max(base_iters, static_cast<int>(std::ceil(static_cast<float>(base_iters) * scale)));
    iters = std::min(iters, 5000);
  }

  std::vector<float> measurements;
  measurements.reserve(repeats);
  if (iters == base_iters) {
    measurements.push_back(pilot_ms / static_cast<float>(base_iters));
  }
  while (static_cast<int>(measurements.size()) < repeats) {
    measurements.push_back(elapsed_ms_for_iterations(iters) / static_cast<float>(iters));
  }
  std::nth_element(measurements.begin(), measurements.begin() + (measurements.size() / 2),
                   measurements.end());
  return measurements[measurements.size() / 2] / 1e3f;
}

LaunchConfig autotune_launch_config(const at::Tensor &x, const at::Tensor &idx, const at::Tensor &theta,
                                    const at::Tensor &scales, int64_t group_size, int64_t krot) {
  const LaunchConfig fallback =
      resolve_legacy_launch_config(x.scalar_type(), kLegacyLaunchSentinel, kLegacyLaunchSentinel);
  const float fallback_seconds =
      benchmark_launch_config(x, idx, theta, scales, group_size, krot, fallback);
  const float min_relative_speedup = resolve_autotune_min_relative_speedup();
  const float min_absolute_speedup_seconds = resolve_autotune_min_absolute_speedup_seconds();
  auto beats_fallback = [&](float candidate_seconds, float baseline_seconds) {
    const float absolute_improvement = baseline_seconds - candidate_seconds;
    return absolute_improvement >= min_absolute_speedup_seconds &&
           candidate_seconds <= baseline_seconds * (1.0f - min_relative_speedup);
  };

  LaunchConfig best = fallback;
  float best_seconds = fallback_seconds;
  for (const LaunchConfig candidate : kAutotuneCandidates) {
    if (candidate.cta_m == fallback.cta_m && candidate.row_pad == fallback.row_pad) {
      continue;
    }
    const float seconds = benchmark_launch_config(x, idx, theta, scales, group_size, krot, candidate);
    if (seconds < best_seconds) {
      best_seconds = seconds;
      best = candidate;
    }
  }
  if (best.cta_m == fallback.cta_m && best.row_pad == fallback.row_pad) {
    return fallback;
  }
  if (!beats_fallback(best_seconds, fallback_seconds)) {
    return fallback;
  }
  const float confirm_fallback_seconds =
      benchmark_launch_config(x, idx, theta, scales, group_size, krot, fallback);
  const float confirm_best_seconds =
      benchmark_launch_config(x, idx, theta, scales, group_size, krot, best);
  if (!beats_fallback(confirm_best_seconds, confirm_fallback_seconds)) {
    return fallback;
  }
  return best;
}

LaunchConfig resolve_cached_autotune_launch_config(const AutotuneKey &key, const at::Tensor &x,
                                                   const at::Tensor &idx, const at::Tensor &theta,
                                                   const at::Tensor &scales, int64_t group_size,
                                                   int64_t krot) {
  if (auto cached = lookup_autotune_cache(key)) {
    return *cached;
  }
  std::lock_guard<std::mutex> guard(autotune_measurement_mutex());
  if (auto cached = lookup_autotune_cache(key)) {
    return *cached;
  }
  LaunchConfig measured = autotune_launch_config(x, idx, theta, scales, group_size, krot);
  return store_autotune_cache(key, measured);
}

at::Tensor build_dummy_pairs(const at::Tensor &x, int64_t group_size, int64_t krot) {
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(x.size(-1) % group_size == 0, "hidden size must be divisible by group_size");
  auto options = torch::TensorOptions().dtype(at::kShort).device(x.device());
  const int64_t groups = x.size(-1) / group_size;
  const auto local_pairs = at::arange(group_size, options);
  return local_pairs.repeat({groups}).unsqueeze(0).repeat({krot, 1}).contiguous();
}

LaunchConfig resolve_runtime_launch_config(const at::Tensor &x, const at::Tensor &idx,
                                           const at::Tensor &theta, const at::Tensor &scales,
                                           int64_t group_size, int64_t krot, int requested_cta_m,
                                           int requested_row_pad, bool has_scale) {
  if (!requests_autotune(requested_cta_m, requested_row_pad)) {
    return resolve_legacy_launch_config(x.scalar_type(), requested_cta_m, requested_row_pad);
  }
  const AutotuneKey key = make_autotune_key(x, group_size, krot, has_scale);
  return resolve_cached_autotune_launch_config(key, x, idx, theta, scales, group_size, krot);
}

LaunchConfig resolve_query_launch_config(const at::Tensor &x, int64_t krot, bool has_scale,
                                         int64_t group_size, int requested_cta_m,
                                         int requested_row_pad) {
  if (!requests_autotune(requested_cta_m, requested_row_pad)) {
    return resolve_legacy_launch_config(x.scalar_type(), requested_cta_m, requested_row_pad);
  }
  const AutotuneKey key = make_autotune_key(x, group_size, krot, has_scale);
  if (auto cached = lookup_autotune_cache(key)) {
    return *cached;
  }
  std::lock_guard<std::mutex> guard(autotune_measurement_mutex());
  if (auto cached = lookup_autotune_cache(key)) {
    return *cached;
  }
  at::Tensor dummy_idx = build_dummy_pairs(x, group_size, krot);
  at::Tensor dummy_theta = at::zeros({krot, x.size(-1) / 2}, x.options());
  at::Tensor dummy_scales = has_scale ? at::ones({1, x.size(-1)}, x.options()) : at::Tensor();
  LaunchConfig measured = autotune_launch_config(x, dummy_idx, dummy_theta, dummy_scales, group_size, krot);
  return store_autotune_cache(key, measured);
}

void clear_rotation_autotune_cache() {
  std::lock_guard<std::mutex> guard(autotune_cache_mutex());
  autotune_cache().clear();
}

int64_t rotation_autotune_cache_size() {
  std::lock_guard<std::mutex> guard(autotune_cache_mutex());
  return static_cast<int64_t>(autotune_cache().size());
}

} // namespace

torch::Tensor rotate_dynamic(at::Tensor x, at::Tensor idx, at::Tensor theta,
                             c10::optional<at::Tensor> scales_opt, int64_t group_size = 128,
                             int64_t requested_cta_m = -1, int64_t requested_row_pad = -1) {
  int64_t krot = theta.size(0);
  TORCH_CHECK(krot == idx.size(0), "theta.size(0) must equal idx_ij.size(0)");
  at::Tensor scales = scales_opt.value_or(at::Tensor());
  const bool has_scale = scales.defined() && scales.numel() > 0;
  const LaunchConfig config = resolve_runtime_launch_config(
      x, idx, theta, scales, group_size, krot, static_cast<int>(requested_cta_m),
      static_cast<int>(requested_row_pad), has_scale);
  return dispatch_rotate_variant(x, idx, theta, scales, group_size, krot, config.cta_m,
                                 config.row_pad);
}

std::vector<int64_t> rotate_launch_config(at::Tensor x, int64_t krot = 8, bool has_scale = true,
                                          int64_t group_size = 128, int64_t cta_m = -1,
                                          int64_t row_pad = -1) {
  const LaunchConfig config = resolve_query_launch_config(
      x, krot, has_scale, group_size, static_cast<int>(cta_m), static_cast<int>(row_pad));
  return {
      static_cast<int64_t>(config.cta_m),
      static_cast<int64_t>(config.row_pad),
  };
}

#undef DISPATCH_ROW_PAD
#undef DISPATCH_CTA_M
#undef DISPATCH_KROT

TORCH_LIBRARY(gptqmodel_paroquant, m) {
  m.def("rotate(Tensor x, Tensor idx_ij, Tensor theta, Tensor? scales=None, int group_size=128, int cta_m=-1, int row_pad=-1) -> Tensor");
  m.def("launch_config(Tensor x, int krot=8, bool has_scale=True, int group_size=128, int cta_m=-1, int row_pad=-1) -> int[]");
  m.def("clear_autotune_cache() -> ()");
  m.def("autotune_cache_size() -> int");
}

TORCH_LIBRARY_IMPL(gptqmodel_paroquant, CUDA, m) {
  m.impl("rotate", &rotate_dynamic);
  m.impl("launch_config", &rotate_launch_config);
}

TORCH_LIBRARY_IMPL(gptqmodel_paroquant, CatchAll, m) {
  m.impl("clear_autotune_cache", &clear_rotation_autotune_cache);
  m.impl("autotune_cache_size", &rotation_autotune_cache_size);
}
