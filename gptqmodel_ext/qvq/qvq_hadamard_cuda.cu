// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

// Fused scale-first fast Walsh-Hadamard transform for QVQ inference.
//
// Replaces the Python butterfly (matmul_hadU_stable) with one kernel launch.
// Bitwise-parity contract (validated): the divisor is fp16(sqrtf(n)) and every
// butterfly add/sub is computed in float32 then rounded to the output dtype,
// exactly matching torch's elementwise half/bfloat16 semantics. Optional
// per-column pre-scale (SU), post-scale (SV), and bias are fused into the same
// launch with the same rounding sequence as the Python forward.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {

template <typename Scalar>
struct QvqHadamardTraits;

template <>
struct QvqHadamardTraits<half> {
  static __device__ __forceinline__ float to_float(half v) { return __half2float(v); }
  static __device__ __forceinline__ half from_float(float v) { return __float2half_rn(v); }
};

template <>
struct QvqHadamardTraits<nv_bfloat16> {
  static __device__ __forceinline__ float to_float(nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }
};

template <>
struct QvqHadamardTraits<float> {
  static __device__ __forceinline__ float to_float(float v) { return v; }
  static __device__ __forceinline__ float from_float(float v) { return v; }
};

constexpr int kHadamardThreads = 1024;

__device__ __forceinline__ float round_fp16_unless_overflow(float value) {
  const float narrowed = __half2float(__float2half_rn(value));
  return isfinite(narrowed) ? narrowed : value;
}

// One block per row; the row lives in dynamic shared memory. log2n must be >= 1
// (n >= 2) and n * sizeof(Scalar) must fit the device's dynamic shared limit.
template <typename Scalar>
__global__ void __launch_bounds__(kHadamardThreads) qvq_hadamard_kernel(
    const Scalar* __restrict__ input,
    Scalar* __restrict__ output,
    const Scalar* __restrict__ pre_scale,   // optional, [n]
    const Scalar* __restrict__ post_scale,  // optional, [n]
    const Scalar* __restrict__ bias,        // optional, [n]
    int n,
    int scale_mode) {
  // scale_mode 0 mirrors matmul_hadU_stable (normalize FIRST, fp16(sqrtf(n))
  // divisor); scale_mode 1 mirrors matmul_hadU (normalize LAST, float sqrtf(n)
  // divisor). Mode 2 is the range-safe QVQ input epilogue: fuse x*SU/sqrt(n)
  // in FP32 before the first narrow store so an algebraically finite transform
  // cannot overflow solely because x*SU was rounded before normalization.
  // Modes 3/4 consume an FP32 inner result while reproducing mode 0/1's FP16
  // rounding whenever that rounding is finite. Only an overflowing narrowing
  // stays FP32, so ordinary model outputs remain bitwise unchanged.
  extern __shared__ char smem_raw[];
  // Padded layout: element i lives at p(i) = i + (i >> 5) so the butterfly's
  // (i, i^bit) pair reads land in distinct shared banks (2-way conflicts were
  // ~half the kernel's latency). Values are unchanged - purely physical.
  auto p = [](int i) { return i + (i >> 5); };
  Scalar* buf = reinterpret_cast<Scalar*>(smem_raw);
  const int row = static_cast<int>(blockIdx.x);
  const Scalar* in_row = input + static_cast<int64_t>(row) * n;
  Scalar* out_row = output + static_cast<int64_t>(row) * n;

  // Normalization divisor semantics mirror the two Python references bitwise:
  //   mode 0 (stable): divide by fp16(sqrtf(n)) with IEEE fp32 division (the
  //     divisor comes from an on-device X.new_tensor(...) -> GPU tensor path);
  //   mode 1 (end-scale): multiply by the fp32-rounded reciprocal 1/sqrtf(n)
  //     (the divisor comes from torch.tensor(...) on CPU -> scalar path, which
  //     torch evaluates as x * fp32(1/d)).
  const bool emulate_fp16 = scale_mode == 3 || scale_mode == 4;
  const bool normalize_first = scale_mode == 0 || scale_mode == 2 || scale_mode == 3;
  const float divisor = normalize_first
      ? (emulate_fp16
          ? __half2float(__float2half_rn(sqrtf(static_cast<float>(n))))
          : QvqHadamardTraits<Scalar>::to_float(
              QvqHadamardTraits<Scalar>::from_float(sqrtf(static_cast<float>(n)))))
      : sqrtf(static_cast<float>(n));
  const float reciprocal = 1.0f / sqrtf(static_cast<float>(n));

  // Modes 3/4 execute the entire butterfly in FP32 storage but round every
  // historical FP16 operation back to FP16 whenever it remains finite. This
  // is stronger than inspecting only the GEMV output: overflow can first
  // occur in a later butterfly, SV multiply, or bias add. Keeping the value in
  // FP32 only at that exact stage preserves all ordinary FP16 results while
  // covering every intermediate range edge.
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float value = QvqHadamardTraits<Scalar>::to_float(in_row[i]);
    if (emulate_fp16) {
      value = round_fp16_unless_overflow(value);
    }
    if (pre_scale != nullptr) {
      const float su = QvqHadamardTraits<Scalar>::to_float(pre_scale[i]);
      value = QvqHadamardTraits<Scalar>::to_float(in_row[i]) * su;
      if (scale_mode == 2) {
        value = round_fp16_unless_overflow(value);
      } else {
        value = QvqHadamardTraits<Scalar>::to_float(QvqHadamardTraits<Scalar>::from_float(value));
      }
    }
    if (normalize_first) {
      value /= divisor;
      if (emulate_fp16) {
        value = round_fp16_unless_overflow(value);
      }
    }
    buf[p(i)] = QvqHadamardTraits<Scalar>::from_float(value);
  }
  __syncthreads();

  // Butterfly: 1, 2, ..., n/2 (ascending), mirroring matmul_hadU's view recursion;
  // each pair (i, i^bit) once, float add/sub + round.
  for (int bit = 1; bit < n; bit <<= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int j = i ^ bit;
      if (i < j) {
        const float a = QvqHadamardTraits<Scalar>::to_float(buf[p(i)]);
        const float b = QvqHadamardTraits<Scalar>::to_float(buf[p(j)]);
        float sum = a + b;
        float difference = a - b;
        if (emulate_fp16) {
          sum = round_fp16_unless_overflow(sum);
          difference = round_fp16_unless_overflow(difference);
        }
        buf[p(i)] = QvqHadamardTraits<Scalar>::from_float(sum);
        buf[p(j)] = QvqHadamardTraits<Scalar>::from_float(difference);
      }
    }
    __syncthreads();
  }

  // Epilogue: late normalization in scale_mode 1, then optional SV post-scale
  // and bias add, each matching Python rounding.
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float value = QvqHadamardTraits<Scalar>::to_float(buf[p(i)]);
    if (scale_mode == 1 || scale_mode == 4) {
      // matmul_hadU rounds the division result back to fp16 before the caller
      // applies SV/bias, so round here before the epilogue scales.
      value *= reciprocal;
      value = emulate_fp16
          ? round_fp16_unless_overflow(value)
          : QvqHadamardTraits<Scalar>::to_float(QvqHadamardTraits<Scalar>::from_float(value));
    }
    if (post_scale != nullptr) {
      value *= QvqHadamardTraits<Scalar>::to_float(post_scale[i]);
      value = emulate_fp16
          ? round_fp16_unless_overflow(value)
          : QvqHadamardTraits<Scalar>::to_float(QvqHadamardTraits<Scalar>::from_float(value));
    }
    if (bias != nullptr) {
      value += QvqHadamardTraits<Scalar>::to_float(bias[i]);
      value = emulate_fp16
          ? round_fp16_unless_overflow(value)
          : QvqHadamardTraits<Scalar>::to_float(QvqHadamardTraits<Scalar>::from_float(value));
    }
    out_row[i] = QvqHadamardTraits<Scalar>::from_float(value);
  }
}

at::Tensor qvq_hadamard_cuda(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_scale,
    const c10::optional<at::Tensor>& bias,
    int64_t scale_mode) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() >= 1, "input must be at least rank one");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "hadamard requires float16, bfloat16, or float32 input");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(scale_mode >= 0 && scale_mode <= 4,
              "scale_mode must be 0/1 (native), 2 (range-safe pre-scale), or 3/4 (FP16 emulation)");
  TORCH_CHECK(scale_mode != 2 || input.scalar_type() == at::kHalf,
              "range-safe pre-scale mode 2 requires float16 input");
  TORCH_CHECK(scale_mode < 3 || input.scalar_type() == at::kFloat,
              "FP16-emulation scale modes 3/4 require float32 input");
  const int64_t n64 = input.size(-1);
  TORCH_CHECK(n64 >= 2 && (n64 & (n64 - 1)) == 0, "hadamard requires a power-of-two last dim");
  TORCH_CHECK(n64 <= 16384, "hadamard fused kernel supports last dim up to 16384");
  const int n = static_cast<int>(n64);
  const int64_t rows = input.numel() / n;
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "row count exceeds int32 kernel limit");

  const auto check_optional = [&](const c10::optional<at::Tensor>& t, const char* name) {
    if (t.has_value()) {
      TORCH_CHECK(t->is_cuda() && t->device() == input.device(), name, " must be on the input device");
      TORCH_CHECK(t->scalar_type() == input.scalar_type(), name, " must match input dtype");
      TORCH_CHECK(t->is_contiguous(), name, " must be contiguous");
      TORCH_CHECK(t->numel() == n, name, " must have ", n, " elements");
    }
  };
  check_optional(pre_scale, "pre_scale");
  check_optional(post_scale, "post_scale");
  check_optional(bias, "bias");

  if (rows == 0) {
    return at::empty_like(input);
  }

  const c10::cuda::CUDAGuard device_guard(input.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  // Padded layout (i + i/32) needs n/32 extra elements.
  const size_t smem_bytes = static_cast<size_t>(n + n / 32) * input.element_size();
  TORCH_CHECK(smem_bytes <= static_cast<size_t>(properties.sharedMemPerBlockOptin),
              "n too large for the device dynamic shared memory limit");

  at::Tensor output = at::empty_like(input);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid(static_cast<unsigned int>(rows));
#define QVQ_HADAMARD_LAUNCH(SCALAR)                                                                                \
  {                                                                                                               \
    const SCALAR* in_ptr = reinterpret_cast<const SCALAR*>(input.const_data_ptr());                                \
    SCALAR* out_ptr = reinterpret_cast<SCALAR*>(output.mutable_data_ptr());                                        \
    const SCALAR* pre = pre_scale.has_value()                                                                      \
        ? reinterpret_cast<const SCALAR*>(pre_scale->const_data_ptr())                                             \
        : nullptr;                                                                                                 \
    const SCALAR* post = post_scale.has_value()                                                                    \
        ? reinterpret_cast<const SCALAR*>(post_scale->const_data_ptr())                                            \
        : nullptr;                                                                                                 \
    const SCALAR* bia = bias.has_value()                                                                           \
        ? reinterpret_cast<const SCALAR*>(bias->const_data_ptr())                                                  \
        : nullptr;                                                                                                 \
    qvq_hadamard_kernel<SCALAR><<<grid, kHadamardThreads, smem_bytes, stream>>>(                                   \
        in_ptr, out_ptr, pre, post, bia, n, static_cast<int>(scale_mode));                                          \
  }
  if (input.scalar_type() == at::kHalf) {
    QVQ_HADAMARD_LAUNCH(half)
  } else if (input.scalar_type() == at::kBFloat16) {
    QVQ_HADAMARD_LAUNCH(nv_bfloat16)
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        qvq_hadamard_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    QVQ_HADAMARD_LAUNCH(float)
  }
#undef QVQ_HADAMARD_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("hadamard(Tensor input, Tensor? pre_scale, Tensor? post_scale, Tensor? bias, int scale_mode) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("hadamard", &qvq_hadamard_cuda);
}
