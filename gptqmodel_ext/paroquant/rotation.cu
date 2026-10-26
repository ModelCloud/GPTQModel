// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0

/******************************************************************************
 * Adapted from https://github.com/z-lab/paroquant
 ******************************************************************************/

#include "rotation.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
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

// Sweep the rotations forward (default) or backward.
//
// The ParoQuant autograd backward only needs the inverse transform of a single
// operand whenever the rotation parameters are frozen. Walking the rotations in
// reverse order with negated thetas is the same sweep the forward kernel already
// performs, so `reverse` folds that inverse into one launch instead of one
// launch per rotation. The flag is uniform across the block, so the branch is
// free and no extra kernel instantiation is needed.
template <typename scalar_t, int CTA_M, int GROUP_SIZE, int KROT, bool USE_SCALE, int ROW_PAD>
__global__ void rotate_kernel(const scalar_t *__restrict__ x, scalar_t *__restrict__ out,
                              const int16_t *__restrict__ idx_ij, const scalar_t *__restrict__ theta,
                              const scalar_t *__restrict__ scales, int s, int h, int reverse) {
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
    const int rr = reverse ? (KROT - 1 - r) : r;
    RotateAccess<scalar_t>::template apply_one<CTA_M, ROW_STRIDE>(
        x_grp, reg_idx[rr], reverse ? -reg_theta[rr] : reg_theta[rr]);
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
                                                  int h, int reverse) {
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
    const int rr = reverse ? (KROT - 1 - r) : r;
    RotateAccess<__half>::template apply_one<CTA_M, ROW_STRIDE>(
        x_grp, reg_idx[rr], reverse ? -reg_theta[rr] : reg_theta[rr]);
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
          x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, s_p, seq_len, h, reverse);                      \
    } else {                                                                                         \
      rotate_kernel<CUDA_T, CTA_M, GROUP_SIZE, KROT, false, ROW_PAD><<<grid, block, 0, stream>>>(   \
          x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, nullptr, seq_len, h, reverse);                  \
    }                                                                                                \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                   \
    break;                                                                                           \
  }

template <int KROT, int CTA_M, int GROUP_SIZE, int ROW_PAD>
torch::Tensor rotate_launcher_bf16_half_workspace(at::Tensor x, at::Tensor idx_ij,
                                                  at::Tensor theta, at::Tensor scales,
                                                  int reverse) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
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
        x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, s_p, seq_len, h, reverse);
  } else {
    rotate_kernel_bf16_half_workspace<CTA_M, GROUP_SIZE, KROT, false, ROW_PAD><<<grid, block, 0, stream>>>(
        x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, nullptr, seq_len, h, reverse);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int KROT, int CTA_M, int GROUP_SIZE, int ROW_PAD>
torch::Tensor rotate_launcher(at::Tensor x, at::Tensor idx_ij, at::Tensor theta,
                              at::Tensor scales, int reverse) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
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
    return rotate_launcher_bf16_half_workspace<KROT, CTA_M, GROUP_SIZE, ROW_PAD>(x, idx_ij, theta,
                                                                                 scales, reverse);
  default:
    TORCH_CHECK(false, "rotate supports Float, Half, and BFloat16, got ", x.scalar_type());
  }
  return out;
}

#undef LAUNCH_ROTATE

#define DISPATCH_ROW_PAD(KROT, CTA, GS)                                                              \
  switch (row_pad) {                                                                                 \
  case 2:                                                                                            \
    return rotate_launcher<KROT, CTA, GS, 2>(x, idx, theta, scales, reverse);                       \
  case 0:                                                                                            \
    return rotate_launcher<KROT, CTA, GS, 0>(x, idx, theta, scales, reverse);                       \
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
                                      int64_t group_size, int64_t krot, int cta_m, int row_pad,
                                      int reverse) {
  if (group_size == 128) {
    DISPATCH_KROT(128)
  }
  TORCH_CHECK(false, "Unsupported group_size: ", group_size, "; expected 128");
}

// Fused two-tensor rotation.
//
// The ParoQuant autograd backward rotates both the saved activation and the
// incoming gradient with the same pair/theta set, so the steady-state loop runs
// two `rotate` launches per rotation per layer. Both operands share a shape and
// only differ in their storage, so the work can be folded into one launch by
// doubling the group grid axis and selecting the operand per block. This halves
// the backward rotation launches without changing the per-CTA math.
template <typename scalar_t, int CTA_M, int GROUP_SIZE, int KROT, int ROW_PAD>
__global__ void rotate_pair_kernel(const scalar_t *__restrict__ x, const scalar_t *__restrict__ x2,
                                   scalar_t *__restrict__ out, scalar_t *__restrict__ out2,
                                   const int16_t *__restrict__ idx_ij,
                                   const scalar_t *__restrict__ theta, int s, int h) {
  constexpr int ROW_STRIDE = CTA_M + ROW_PAD;
  __shared__ scalar_t x_grp[ROW_STRIDE * GROUP_SIZE];

  const int j = blockIdx.x;
  const int groups_per_row = h / GROUP_SIZE;
  const int g_full = blockIdx.y;
  // blockIdx.y is uniform across the block, so this branch never diverges.
  const bool second = g_full >= groups_per_row;
  const int g = second ? g_full - groups_per_row : g_full;
  const scalar_t *__restrict__ src = second ? x2 : x;
  scalar_t *__restrict__ dst = second ? out2 : out;
  const int t = threadIdx.x;

  RotateAccess<scalar_t>::template load_group<CTA_M, ROW_STRIDE, GROUP_SIZE, false>(
      x_grp, src, nullptr, s, h, j, g, t);

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

  RotateAccess<scalar_t>::template store_group<CTA_M, ROW_STRIDE, GROUP_SIZE>(dst, x_grp, s, h, j,
                                                                              g, t);
}

template <int KROT, int CTA_M, int GROUP_SIZE, int ROW_PAD>
std::tuple<at::Tensor, at::Tensor> rotate_pair_launcher(at::Tensor x, at::Tensor x2,
                                                        at::Tensor idx_ij, at::Tensor theta) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const int h = static_cast<int>(x.size(-1));
  TORCH_CHECK(h % GROUP_SIZE == 0, "h must be divisible by GROUP_SIZE");
  const int groups_per_row = h / GROUP_SIZE;
  constexpr int pn = GROUP_SIZE / 2;
  const int seq_len = static_cast<int>(x.numel() / x.size(-1));
  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  at::Tensor out = torch::empty(x.sizes(), options);
  at::Tensor out2 = torch::empty(x.sizes(), options);

  const at::ScalarType dtype = x.scalar_type();
  auto theta_cast = theta.scalar_type() == dtype ? theta : theta.to(dtype);

  dim3 grid((seq_len + CTA_M - 1) / CTA_M, groups_per_row * 2);
  dim3 block(pn);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (dtype) {
  case at::kFloat:
    rotate_pair_kernel<float, CTA_M, GROUP_SIZE, KROT, ROW_PAD><<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), x2.data_ptr<float>(), out.data_ptr<float>(), out2.data_ptr<float>(),
        idx_ij.data_ptr<int16_t>(), theta_cast.data_ptr<float>(), seq_len, h);
    break;
  case at::kHalf:
    rotate_pair_kernel<__half, CTA_M, GROUP_SIZE, KROT, ROW_PAD><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<c10::Half>()),
        reinterpret_cast<const __half *>(x2.data_ptr<c10::Half>()),
        reinterpret_cast<__half *>(out.data_ptr<c10::Half>()),
        reinterpret_cast<__half *>(out2.data_ptr<c10::Half>()), idx_ij.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(theta_cast.data_ptr<c10::Half>()), seq_len, h);
    break;
  default:
    TORCH_CHECK(false, "rotate_pair supports Float and Half, got ", dtype);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, out2};
}

#define DISPATCH_PAIR_ROW_PAD(KROT, CTA)                                                             \
  switch (row_pad) {                                                                                 \
  case 2:                                                                                            \
    return rotate_pair_launcher<KROT, CTA, 128, 2>(x, x2, idx, theta);                              \
  case 0:                                                                                            \
    return rotate_pair_launcher<KROT, CTA, 128, 0>(x, x2, idx, theta);                              \
  default:                                                                                           \
    TORCH_CHECK(false, "Unsupported ROW_PAD = ", row_pad, "; compiled variants: 0 and 2");          \
  }

#define DISPATCH_PAIR_CTA_M(KROT)                                                                    \
  switch (cta_m) {                                                                                   \
  case 16:                                                                                           \
    DISPATCH_PAIR_ROW_PAD(KROT, 16)                                                                  \
  case 8:                                                                                            \
    DISPATCH_PAIR_ROW_PAD(KROT, 8)                                                                   \
  case 4:                                                                                            \
    DISPATCH_PAIR_ROW_PAD(KROT, 4)                                                                   \
  default:                                                                                           \
    TORCH_CHECK(false, "Unsupported CTA_M = ", cta_m, "; compiled variants: 4, 8, and 16");        \
  }

std::tuple<at::Tensor, at::Tensor> dispatch_rotate_pair_variant(at::Tensor x, at::Tensor x2,
                                                                at::Tensor idx, at::Tensor theta,
                                                                int64_t group_size, int64_t krot,
                                                                int cta_m, int row_pad) {
  if (group_size == 128) {
    switch (krot) {
    case 1:
      DISPATCH_PAIR_CTA_M(1)
    case 8:
      DISPATCH_PAIR_CTA_M(8)
    default:
      TORCH_CHECK(false, "Unsupported KROT = ", krot, "; compiled variants: 1 and 8");
    }
  }
  TORCH_CHECK(false, "Unsupported group_size: ", group_size, "; expected 128");
}

#undef DISPATCH_PAIR_ROW_PAD
#undef DISPATCH_PAIR_CTA_M

// Fused two-tensor rotation sweep with the rotation-gradient accumulation.
//
// The ParoQuant autograd backward sweeps the rotations in reverse: at every
// step it rotates the saved activation and the incoming gradient by the negated
// theta and then measures the rotation gradient from the freshly rotated pair.
// That is one `rotate_pair` plus one `grad_theta` launch per rotation, plus the
// host-side staging of the sin/cos tables and the gathered index pairs.
//
// This kernel folds the whole sweep into a single launch. The running rotated
// state is kept in shared memory, every rotation's gradient contribution is
// accumulated in registers, and only the cross-tile reduction is left over.
//
// Each thread owns exactly one (i, j) column pair per rotation, so the readback
// after `apply_one` only touches columns the calling thread just wrote; a single
// barrier per rotation covers the shared-memory staging.
template <typename scalar_t, int CTA_M, int GROUP_SIZE, int KROT, int ROW_PAD>
__global__ void rotate_pair_grad_kernel(const scalar_t *__restrict__ x,
                                        const scalar_t *__restrict__ g,
                                        scalar_t *__restrict__ out_x,
                                        scalar_t *__restrict__ out_g,
                                        const int16_t *__restrict__ idx_ij,
                                        const scalar_t *__restrict__ theta,
                                        float *__restrict__ partials, int s, int h,
                                        int row_tiles) {
  constexpr int ROW_STRIDE = CTA_M + ROW_PAD;
  constexpr int HALF_GROUP = GROUP_SIZE / 2;
  __shared__ scalar_t x_grp[ROW_STRIDE * GROUP_SIZE];
  __shared__ scalar_t g_grp[ROW_STRIDE * GROUP_SIZE];

  const int tile = blockIdx.x;
  const int grp = blockIdx.y;
  const int t = threadIdx.x;
  const int hh = h / 2;
  const int base_row = tile * CTA_M;

  RotateAccess<scalar_t>::template load_group<CTA_M, ROW_STRIDE, GROUP_SIZE, false>(
      x_grp, x, nullptr, s, h, tile, grp, t);
  RotateAccess<scalar_t>::template load_group<CTA_M, ROW_STRIDE, GROUP_SIZE, false>(
      g_grp, g, nullptr, s, h, tile, grp, t);

  float reg_theta[KROT];
  int reg_idx[KROT];
  RotateAccess<scalar_t>::template load_coeffs<KROT, GROUP_SIZE>(reg_theta, reg_idx, idx_ij, theta,
                                                                 h, grp, t);
  __syncthreads();

#pragma unroll
  for (int r = KROT - 1; r >= 0; r--) {
    const float tr = reg_theta[r];
    RotateAccess<scalar_t>::template apply_one<CTA_M, ROW_STRIDE>(x_grp, reg_idx[r], -tr);
    RotateAccess<scalar_t>::template apply_one<CTA_M, ROW_STRIDE>(g_grp, reg_idx[r], -tr);
    __syncthreads();

    const int i = reg_idx[r] & 0xFFFF;
    const int j = reg_idx[r] >> 16;
    float sum_a = 0.0f;
    float sum_b = 0.0f;
#pragma unroll
    for (int m = 0; m < CTA_M; m++) {
      if (base_row + m < s) {
        const float xi = RotateAccess<scalar_t>::to_float(x_grp[i * ROW_STRIDE + m]);
        const float xj = RotateAccess<scalar_t>::to_float(x_grp[j * ROW_STRIDE + m]);
        const float gi = RotateAccess<scalar_t>::to_float(g_grp[i * ROW_STRIDE + m]);
        const float gj = RotateAccess<scalar_t>::to_float(g_grp[j * ROW_STRIDE + m]);
        sum_a += gi * xj - gj * xi;
        sum_b += gi * xi + gj * xj;
      }
    }
    float sn, cs;
    __sincosf(tr, &sn, &cs);
    partials[r * row_tiles * hh + tile * hh + grp * HALF_GROUP + t] = sum_a * cs - sum_b * sn;
  }

  RotateAccess<scalar_t>::template store_group<CTA_M, ROW_STRIDE, GROUP_SIZE>(out_x, x_grp, s, h,
                                                                              tile, grp, t);
  RotateAccess<scalar_t>::template store_group<CTA_M, ROW_STRIDE, GROUP_SIZE>(out_g, g_grp, s, h,
                                                                              tile, grp, t);
}

// Collapse the per-row-tile partials into one rotation-gradient tensor. The
// reduction stays deterministic: a fixed tile order per output element, with no
// atomics, so repeated runs produce identical results.
//
// Each output element walks the tile axis with a stride of `hh` floats, so the
// loop trip count is a runtime value and the compiler cannot unroll it. That
// leaves one strided load in flight per thread, which is pure memory latency:
// measured at 334 us for the 1216-tile weight shapes, ~95x off the bandwidth of
// the same byte volume. Walking the axis in fixed-size batches issues the loads
// together while the adds still run in ascending tile order, so the sum is
// bit-identical to the plain sequential loop and only the latency hiding
// changes.
//
// Two batch widths: the wide one buys the most overlap on the long tile axes
// (weight matrices run 224-1216 tiles), while the narrow one keeps the short
// activation axes (16-32 tiles) inside a single batch instead of falling
// through to the scalar tail.
constexpr int kRotateGradReduceBatchWide = 32;
constexpr int kRotateGradReduceBatch = 16;

// Stage one batch of strided tile loads, then fold them into `sum` in ascending
// tile order. The loads run in their own loop so they are all in flight before
// the first add consumes one.
template <int BATCH>
__device__ __forceinline__ float rotate_grad_reduce_batch(const float *__restrict__ src, int tile,
                                                          int hh, float sum) {
  float staged[BATCH];
#pragma unroll
  for (int k = 0; k < BATCH; k++) {
    staged[k] = src[static_cast<int64_t>(tile + k) * hh];
  }
#pragma unroll
  for (int k = 0; k < BATCH; k++) {
    sum += staged[k];
  }
  return sum;
}

__global__ void rotate_grad_reduce_kernel(const float *__restrict__ partials,
                                          float *__restrict__ out, int row_tiles, int hh,
                                          int64_t total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int r = static_cast<int>(idx / hh);
  const int e = static_cast<int>(idx % hh);
  const float *__restrict__ src = partials + static_cast<int64_t>(r) * row_tiles * hh + e;
  float sum = 0.0f;
  int tile = 0;
  for (; tile + kRotateGradReduceBatchWide <= row_tiles; tile += kRotateGradReduceBatchWide) {
    sum = rotate_grad_reduce_batch<kRotateGradReduceBatchWide>(src, tile, hh, sum);
  }
  for (; tile + kRotateGradReduceBatch <= row_tiles; tile += kRotateGradReduceBatch) {
    sum = rotate_grad_reduce_batch<kRotateGradReduceBatch>(src, tile, hh, sum);
  }
  for (; tile < row_tiles; tile++) {
    sum += src[static_cast<int64_t>(tile) * hh];
  }
  out[idx] = sum;
}

template <int KROT, int CTA_M, int GROUP_SIZE, int ROW_PAD>
std::tuple<at::Tensor, at::Tensor, at::Tensor>
rotate_pair_grad_launcher(at::Tensor x, at::Tensor g, at::Tensor idx_ij, at::Tensor theta) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const int h = static_cast<int>(x.size(-1));
  TORCH_CHECK(h % GROUP_SIZE == 0, "h must be divisible by GROUP_SIZE");
  const int groups_per_row = h / GROUP_SIZE;
  constexpr int pn = GROUP_SIZE / 2;
  const int seq_len = static_cast<int>(x.numel() / x.size(-1));
  const int row_tiles = std::max(1, (seq_len + CTA_M - 1) / CTA_M);
  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  at::Tensor out_x = torch::empty(x.sizes(), options);
  at::Tensor out_g = torch::empty(x.sizes(), options);
  at::Tensor partials = at::empty({KROT, row_tiles, h / 2}, x.options().dtype(at::kFloat));
  at::Tensor grad = at::empty({KROT, h / 2}, x.options().dtype(at::kFloat));

  const at::ScalarType dtype = x.scalar_type();
  auto theta_cast = theta.scalar_type() == dtype ? theta : theta.to(dtype);

  dim3 grid(row_tiles, groups_per_row);
  dim3 block(pn);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (dtype) {
  case at::kFloat:
    rotate_pair_grad_kernel<float, CTA_M, GROUP_SIZE, KROT, ROW_PAD><<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), g.data_ptr<float>(), out_x.data_ptr<float>(), out_g.data_ptr<float>(),
        idx_ij.data_ptr<int16_t>(), theta_cast.data_ptr<float>(), partials.data_ptr<float>(),
        seq_len, h, row_tiles);
    break;
  case at::kHalf:
    rotate_pair_grad_kernel<__half, CTA_M, GROUP_SIZE, KROT, ROW_PAD><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<c10::Half>()),
        reinterpret_cast<const __half *>(g.data_ptr<c10::Half>()),
        reinterpret_cast<__half *>(out_x.data_ptr<c10::Half>()),
        reinterpret_cast<__half *>(out_g.data_ptr<c10::Half>()), idx_ij.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(theta_cast.data_ptr<c10::Half>()),
        partials.data_ptr<float>(), seq_len, h, row_tiles);
    break;
  default:
    TORCH_CHECK(false, "rotate_pair_grad supports Float and Half, got ", dtype);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t total = static_cast<int64_t>(KROT) * (h / 2);
  const int reduce_block = 256;
  const int reduce_grid = static_cast<int>((total + reduce_block - 1) / reduce_block);
  rotate_grad_reduce_kernel<<<reduce_grid, reduce_block, 0, stream>>>(
      partials.data_ptr<float>(), grad.data_ptr<float>(), row_tiles, h / 2, total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out_x, out_g, grad};
}

#define DISPATCH_GRADPAIR_ROW_PAD(KROT, CTA)                                                         \
  switch (row_pad) {                                                                                 \
  case 2:                                                                                            \
    return rotate_pair_grad_launcher<KROT, CTA, 128, 2>(x, g, idx, theta);                          \
  case 0:                                                                                            \
    return rotate_pair_grad_launcher<KROT, CTA, 128, 0>(x, g, idx, theta);                          \
  default:                                                                                           \
    TORCH_CHECK(false, "Unsupported ROW_PAD = ", row_pad, "; compiled variants: 0 and 2");          \
  }

#define DISPATCH_GRADPAIR_CTA_M(KROT)                                                                \
  switch (cta_m) {                                                                                   \
  case 16:                                                                                           \
    DISPATCH_GRADPAIR_ROW_PAD(KROT, 16)                                                              \
  case 8:                                                                                            \
    DISPATCH_GRADPAIR_ROW_PAD(KROT, 8)                                                               \
  case 4:                                                                                            \
    DISPATCH_GRADPAIR_ROW_PAD(KROT, 4)                                                               \
  default:                                                                                           \
    TORCH_CHECK(false, "Unsupported CTA_M = ", cta_m, "; compiled variants: 4, 8, and 16");        \
  }

std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_rotate_pair_grad_variant(
    at::Tensor x, at::Tensor g, at::Tensor idx, at::Tensor theta, int64_t group_size, int64_t krot,
    int cta_m, int row_pad) {
  if (group_size == 128) {
    switch (krot) {
    case 1:
      DISPATCH_GRADPAIR_CTA_M(1)
    case 8:
      DISPATCH_GRADPAIR_CTA_M(8)
    default:
      TORCH_CHECK(false, "Unsupported KROT = ", krot, "; compiled variants: 1 and 8");
    }
  }
  TORCH_CHECK(false, "Unsupported group_size: ", group_size, "; expected 128");
}

#undef DISPATCH_GRADPAIR_ROW_PAD
#undef DISPATCH_GRADPAIR_CTA_M

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
                                         config.row_pad, /*reverse=*/0);
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
                                       config.row_pad, /*reverse=*/0);
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

void validate_rotation_inputs(const at::Tensor &x, const at::Tensor &idx,
                              const at::Tensor &theta, const at::Tensor &scales) {
  // Rotation kernels use raw pointers and vectorized loads, so reject invalid
  // devices/layouts before dispatching a specialized launch variant.
  const bool has_scale = scales.defined() && scales.numel() > 0;
  TORCH_CHECK(x.is_cuda() && idx.is_cuda() && theta.is_cuda(),
              "ParoQuant rotation expects CUDA tensors.");
  TORCH_CHECK(!has_scale || scales.is_cuda(),
              "ParoQuant rotation scales must be a CUDA tensor.");
  TORCH_CHECK(x.device() == idx.device() && x.device() == theta.device() &&
                  (!has_scale || x.device() == scales.device()),
              "ParoQuant rotation tensors must be on the same CUDA device.");
  TORCH_CHECK(x.is_contiguous() && idx.is_contiguous() && theta.is_contiguous() &&
                  (!has_scale || scales.is_contiguous()),
              "ParoQuant rotation tensors must be contiguous.");
  TORCH_CHECK(x.is_floating_point() && x.dim() >= 1,
              "ParoQuant rotation input must be a floating-point tensor with at least one dimension.");
  TORCH_CHECK(idx.scalar_type() == at::kShort && idx.dim() == 2,
              "ParoQuant rotation indices must be contiguous int16 pairs.");
  TORCH_CHECK(theta.is_floating_point() && theta.dim() == 2,
              "ParoQuant rotation theta must be a 2D floating-point tensor.");
  TORCH_CHECK(!has_scale || (scales.is_floating_point() && scales.dim() >= 1),
              "ParoQuant rotation scales must be floating-point.");
}

} // namespace

torch::Tensor rotate_dynamic(at::Tensor x, at::Tensor idx, at::Tensor theta,
                             c10::optional<at::Tensor> scales_opt, int64_t group_size = 128,
                             int64_t requested_cta_m = -1, int64_t requested_row_pad = -1,
                             int64_t reverse = 0) {
  TORCH_CHECK(x.is_cuda(), "ParoQuant rotation requires a CUDA input tensor.");
  // Autotune and the final launch must run on x's device, not the thread's
  // previously selected CUDA device.
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  TORCH_CHECK(theta.dim() >= 1 && idx.dim() >= 1,
              "ParoQuant rotation theta and indices must have at least one dimension.");
  int64_t krot = theta.size(0);
  TORCH_CHECK(krot == idx.size(0), "theta.size(0) must equal idx_ij.size(0)");
  at::Tensor scales = scales_opt.value_or(at::Tensor());
  validate_rotation_inputs(x, idx, theta, scales);
  const bool has_scale = scales.defined() && scales.numel() > 0;
  const LaunchConfig config = resolve_runtime_launch_config(
      x, idx, theta, scales, group_size, krot, static_cast<int>(requested_cta_m),
      static_cast<int>(requested_row_pad), has_scale);
  return dispatch_rotate_variant(x, idx, theta, scales, group_size, krot, config.cta_m,
                                 config.row_pad, reverse != 0 ? 1 : 0);
}

std::vector<int64_t> rotate_launch_config(at::Tensor x, int64_t krot = 8, bool has_scale = true,
                                          int64_t group_size = 128, int64_t cta_m = -1,
                                          int64_t row_pad = -1) {
  TORCH_CHECK(x.is_cuda(), "ParoQuant rotation launch config requires a CUDA input tensor.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  TORCH_CHECK(x.is_contiguous() && x.is_floating_point() && x.dim() >= 1,
              "ParoQuant rotation launch config expects contiguous floating-point input.");
  const LaunchConfig config = resolve_query_launch_config(
      x, krot, has_scale, group_size, static_cast<int>(cta_m), static_cast<int>(row_pad));
  return {
      static_cast<int64_t>(config.cta_m),
      static_cast<int64_t>(config.row_pad),
  };
}

std::tuple<at::Tensor, at::Tensor> rotate_pair(at::Tensor x, at::Tensor x2, at::Tensor idx,
                                               at::Tensor theta, int64_t group_size = 128,
                                               int64_t requested_cta_m = -1,
                                               int64_t requested_row_pad = -1) {
  TORCH_CHECK(x.is_cuda() && x2.is_cuda(), "ParoQuant rotate_pair requires CUDA tensors.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  TORCH_CHECK(x.sizes() == x2.sizes(), "ParoQuant rotate_pair expects operands with equal shapes.");
  TORCH_CHECK(x.scalar_type() == x2.scalar_type(),
              "ParoQuant rotate_pair expects operands with equal dtypes.");
  const at::Tensor no_scales;
  validate_rotation_inputs(x, idx, theta, no_scales);
  validate_rotation_inputs(x2, idx, theta, no_scales);
  const int64_t krot = theta.size(0);
  const LaunchConfig config = resolve_runtime_launch_config(
      x, idx, theta, no_scales, group_size, krot, static_cast<int>(requested_cta_m),
      static_cast<int>(requested_row_pad), false);
  return dispatch_rotate_pair_variant(x, x2, idx, theta, group_size, krot, config.cta_m,
                                      config.row_pad);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
rotate_pair_grad(at::Tensor x, at::Tensor g, at::Tensor idx, at::Tensor theta,
                 int64_t group_size = 128, int64_t requested_cta_m = -1,
                 int64_t requested_row_pad = -1) {
  TORCH_CHECK(x.is_cuda() && g.is_cuda(), "ParoQuant rotate_pair_grad requires CUDA tensors.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  TORCH_CHECK(x.sizes() == g.sizes(),
              "ParoQuant rotate_pair_grad expects operands with equal shapes.");
  TORCH_CHECK(x.scalar_type() == g.scalar_type(),
              "ParoQuant rotate_pair_grad expects operands with equal dtypes.");
  const at::Tensor no_scales;
  validate_rotation_inputs(x, idx, theta, no_scales);
  validate_rotation_inputs(g, idx, theta, no_scales);
  const int64_t krot = theta.size(0);
  const LaunchConfig config = resolve_runtime_launch_config(
      x, idx, theta, no_scales, group_size, krot, static_cast<int>(requested_cta_m),
      static_cast<int>(requested_row_pad), false);
  return dispatch_rotate_pair_grad_variant(x, g, idx, theta, group_size, krot, config.cta_m,
                                           config.row_pad);
}

#undef DISPATCH_ROW_PAD
#undef DISPATCH_CTA_M
#undef DISPATCH_KROT

// Fused ParoQuant rotation-gradient accumulation.
//
// The autograd backward of the fused rotation used to build, for every one of
// the `krot` rotations, four advanced-index gathers of the (batch, hidden)
// activation and gradient plus ~11 elementwise/reduce ops on the gathered
// (batch, groups, half_group) views. On ROCm that is ~21 tiny kernel launches
// per rotation and 8 rotations per layer, so the backward is launch-bound:
// profiling the PARO lifecycle showed ~647k kernel launches with ~89% of host
// time inside kernel dispatch.
//
// This op collapses one rotation's whole contribution into a single launch:
//
//   out[e] = ( sum_b grad[b, i] * rot[b, j] - grad[b, j] * rot[b, i] ) * cos[e]
//          - ( sum_b grad[b, i] * rot[b, i] + grad[b, j] * rot[b, j] ) * sin[e]
//
// with i = idx_i[e], j = idx_j[e]. The two partial sums are accumulated
// separately in fp32 and combined after the batch loop, which matches the
// `(A).sum(0) * cos - (B).sum(0) * sin` form of the torch path.
constexpr int GRAD_THETA_E_TILE = 128;
constexpr int GRAD_THETA_B_CHUNK = 32;

__global__ void grad_theta_kernel(const float *__restrict__ rot, const float *__restrict__ grad,
                                  const int64_t *__restrict__ idx_i,
                                  const int64_t *__restrict__ idx_j,
                                  const float *__restrict__ sin_t,
                                  const float *__restrict__ cos_t, float *__restrict__ out,
                                  int batch, int hidden, int h) {
  const int e = blockIdx.x * GRAD_THETA_E_TILE + threadIdx.x;
  if (e >= h) {
    return;
  }

  const int64_t i = idx_i[e];
  const int64_t j = idx_j[e];

  const int b_begin = blockIdx.y * GRAD_THETA_B_CHUNK;
  const int b_end = min(batch, b_begin + GRAD_THETA_B_CHUNK);

  float sum_a = 0.0f;
  float sum_b = 0.0f;
  for (int b = b_begin; b < b_end; ++b) {
    const int64_t base = static_cast<int64_t>(b) * hidden;
    const float gi = grad[base + i];
    const float gj = grad[base + j];
    const float xi = rot[base + i];
    const float xj = rot[base + j];
    sum_a += gi * xj - gj * xi;
    sum_b += gi * xi + gj * xj;
  }

  atomicAdd(out + e, sum_a * cos_t[e] - sum_b * sin_t[e]);
}

at::Tensor grad_theta(at::Tensor rot, at::Tensor grad, at::Tensor idx_i, at::Tensor idx_j,
                      at::Tensor sin_t, at::Tensor cos_t) {
  TORCH_CHECK(rot.is_cuda() && grad.is_cuda(), "ParoQuant grad_theta expects CUDA tensors.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(rot));
  TORCH_CHECK(rot.dim() == 2 && grad.dim() == 2 && grad.size(0) == rot.size(0) &&
                  grad.size(1) == rot.size(1),
              "ParoQuant grad_theta expects rot and grad to share a (batch, hidden) shape.");
  TORCH_CHECK(rot.is_contiguous() && grad.is_contiguous() && idx_i.is_contiguous() &&
                  idx_j.is_contiguous() && sin_t.is_contiguous() && cos_t.is_contiguous(),
              "ParoQuant grad_theta expects contiguous tensors.");
  TORCH_CHECK(rot.scalar_type() == at::kFloat && grad.scalar_type() == at::kFloat,
              "ParoQuant grad_theta supports float32 activations.");
  TORCH_CHECK(idx_i.scalar_type() == at::kLong && idx_j.scalar_type() == at::kLong &&
                  idx_i.dim() == 1 && idx_j.dim() == 1 && idx_i.size(0) == idx_j.size(0),
              "ParoQuant grad_theta expects matching 1D int64 index tensors.");
  TORCH_CHECK(sin_t.scalar_type() == at::kFloat && cos_t.scalar_type() == at::kFloat,
              "ParoQuant grad_theta expects float32 sin/cos tables.");

  const int64_t h = idx_i.size(0);
  const int64_t batch = rot.size(0);
  const int64_t hidden = rot.size(1);
  TORCH_CHECK(sin_t.numel() == h && cos_t.numel() == h,
              "ParoQuant grad_theta sin/cos tables must match the index length.");

  at::Tensor out = at::zeros({h}, rot.options());
  if (h == 0 || batch == 0) {
    return out;
  }

  const dim3 grid(static_cast<unsigned>((h + GRAD_THETA_E_TILE - 1) / GRAD_THETA_E_TILE),
                  static_cast<unsigned>((batch + GRAD_THETA_B_CHUNK - 1) / GRAD_THETA_B_CHUNK));
  const dim3 block(GRAD_THETA_E_TILE);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  grad_theta_kernel<<<grid, block, 0, stream>>>(rot.data_ptr<float>(), grad.data_ptr<float>(),
                                                idx_i.data_ptr<int64_t>(), idx_j.data_ptr<int64_t>(),
                                                sin_t.data_ptr<float>(), cos_t.data_ptr<float>(),
                                                out.data_ptr<float>(),
                                                static_cast<int>(batch), static_cast<int>(hidden),
                                                static_cast<int>(h));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY(gptqmodel_paroquant, m) {
  m.def("rotate(Tensor x, Tensor idx_ij, Tensor theta, Tensor? scales=None, int group_size=128, int cta_m=-1, int row_pad=-1, int reverse=0) -> Tensor");
  m.def("rotate_pair(Tensor x, Tensor x2, Tensor idx_ij, Tensor theta, int group_size=128, int cta_m=-1, int row_pad=-1) -> (Tensor, Tensor)");
  m.def("rotate_pair_grad(Tensor x, Tensor x2, Tensor idx_ij, Tensor theta, int group_size=128, int cta_m=-1, int row_pad=-1) -> (Tensor, Tensor, Tensor)");
  m.def("launch_config(Tensor x, int krot=8, bool has_scale=True, int group_size=128, int cta_m=-1, int row_pad=-1) -> int[]");
  m.def("clear_autotune_cache() -> ()");
  m.def("autotune_cache_size() -> int");
  m.def("grad_theta(Tensor rot, Tensor grad, Tensor idx_i, Tensor idx_j, Tensor sin_t, Tensor cos_t) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_paroquant, CUDA, m) {
  m.impl("rotate", &rotate_dynamic);
  m.impl("rotate_pair", &rotate_pair);
  m.impl("rotate_pair_grad", &rotate_pair_grad);
  m.impl("launch_config", &rotate_launch_config);
  m.impl("grad_theta", &grad_theta);
}

TORCH_LIBRARY_IMPL(gptqmodel_paroquant, CatchAll, m) {
  m.impl("clear_autotune_cache", &clear_rotation_autotune_cache);
  m.impl("autotune_cache_size", &rotation_autotune_cache_size);
}
