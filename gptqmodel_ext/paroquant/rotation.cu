/******************************************************************************
 * Adapted from https://github.com/z-lab/paroquant
 ******************************************************************************/

#include "rotation.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <torch/extension.h>

template <typename scalar_t, int CTA_M, int GROUP_SIZE, int KROT, bool USE_SCALE>
__global__ void rotate_kernel(const scalar_t *__restrict__ x, scalar_t *__restrict__ out,
                              const int16_t *__restrict__ idx_ij, const scalar_t *__restrict__ theta,
                              const scalar_t *__restrict__ scales, int s, int h) {
  __shared__ scalar_t x_grp[CTA_M * GROUP_SIZE];

  int j = blockIdx.x;
  int g = blockIdx.y;
  int t = threadIdx.x;

  RotateAccess<scalar_t>::template load_group<CTA_M, GROUP_SIZE, USE_SCALE>(x_grp, x, scales, s, h,
                                                                            j, g, t);

  float reg_theta[KROT];
  int reg_idx[KROT];
  RotateAccess<scalar_t>::template load_coeffs<KROT, GROUP_SIZE>(reg_theta, reg_idx, idx_ij, theta,
                                                                 h, g, t);
  __syncthreads();

#pragma unroll
  for (int r = 0; r < KROT; r++) {
    RotateAccess<scalar_t>::template apply_one<CTA_M>(x_grp, reg_idx[r], reg_theta[r]);
    __syncthreads();
  }

  RotateAccess<scalar_t>::template store_group<CTA_M, GROUP_SIZE>(out, x_grp, s, h, j, g, t);
}

#define LAUNCH_ROTATE(CUDA_T, TORCH_T)                                                               \
  {                                                                                                  \
    auto *x_p = reinterpret_cast<CUDA_T *>(x.data_ptr<TORCH_T>());                                   \
    auto *o_p = reinterpret_cast<CUDA_T *>(out.data_ptr<TORCH_T>());                                 \
    auto *t_p = reinterpret_cast<CUDA_T *>(theta_cast.data_ptr<TORCH_T>());                          \
    if (has_scale) {                                                                                 \
      auto *s_p = reinterpret_cast<CUDA_T *>(scales_cast.data_ptr<TORCH_T>());                       \
      rotate_kernel<CUDA_T, CTA_M, GROUP_SIZE, KROT, true><<<grid, block, 0, stream>>>(             \
          x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, s_p, seq_len, h);                               \
    } else {                                                                                         \
      rotate_kernel<CUDA_T, CTA_M, GROUP_SIZE, KROT, false><<<grid, block, 0, stream>>>(            \
          x_p, o_p, idx_ij.data_ptr<int16_t>(), t_p, nullptr, seq_len, h);                           \
    }                                                                                                \
    break;                                                                                           \
  }

template <int KROT, int CTA_M, int GROUP_SIZE>
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
    LAUNCH_ROTATE(__nv_bfloat16, c10::BFloat16)
  default:
    TORCH_CHECK(false, "rotate supports Float, Half, and BFloat16, got ", x.scalar_type());
  }
  return out;
}

#undef LAUNCH_ROTATE

#define DISPATCH_CTA_M(KROT, GS)                                                                     \
  switch (cta_m) {                                                                                   \
  case 16:                                                                                           \
    return rotate_launcher<KROT, 16, GS>(x, idx, theta, scales);                                    \
  case 8:                                                                                            \
    return rotate_launcher<KROT, 8, GS>(x, idx, theta, scales);                                     \
  case 4:                                                                                            \
    return rotate_launcher<KROT, 4, GS>(x, idx, theta, scales);                                     \
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

namespace {

int resolve_cta_m(int seq_len) {
  if (const char *override_value = std::getenv("GPTQMODEL_PAROQUANT_ROTATE_CTA_M")) {
    int parsed = std::atoi(override_value);
    if (parsed == 4 || parsed == 8 || parsed == 16) {
      return parsed;
    }
  }

  if (seq_len >= 1024) {
    return 16;
  }
  if (seq_len >= 128) {
    return 8;
  }
  return 4;
}

} // namespace

torch::Tensor rotate_dynamic(at::Tensor x, at::Tensor idx, at::Tensor theta,
                             c10::optional<at::Tensor> scales_opt, int64_t group_size = 128) {
  int64_t krot = theta.size(0);
  TORCH_CHECK(krot == idx.size(0), "theta.size(0) must equal idx_ij.size(0)");
  at::Tensor scales = scales_opt.value_or(at::Tensor());
  int seq_len = x.numel() / x.size(-1);
  int cta_m = resolve_cta_m(seq_len);

  if (group_size == 128) {
    DISPATCH_KROT(128)
  }
  TORCH_CHECK(false, "Unsupported group_size: ", group_size, "; expected 128");
}

#undef DISPATCH_CTA_M
#undef DISPATCH_KROT

TORCH_LIBRARY(gptqmodel_paroquant, m) {
  m.def("rotate(Tensor x, Tensor idx_ij, Tensor theta, Tensor? scales=None, int group_size=128) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_paroquant, CUDA, m) {
  m.impl("rotate", &rotate_dynamic);
}
