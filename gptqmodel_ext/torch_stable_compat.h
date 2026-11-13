// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Compatibility layer for JIT kernels built against the LibTorch stable ABI
// ("AnyTorch"). Include this header instead of <torch/extension.h> /
// <torch/library.h>; the build must define TORCH_TARGET_VERSION so that only
// the stable torch::stable / torch::headeronly surfaces are used.
//
// The aliases below intentionally mirror the small subset of the unstable
// libtorch API that the GPT-QModel kernels use. Keeping the mapping in one
// place makes it obvious which symbols are ABI-stable and lets the kernels
// share one binary across torch releases.
#pragma once

#ifndef TORCH_TARGET_VERSION
#error "torch_stable_compat.h requires TORCH_TARGET_VERSION (stable ABI build)"
#endif

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device_struct.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float4_e2m1fn_x2.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <torch/headeronly/util/Float8_e8m0fnu.h>
#include <torch/headeronly/util/Half.h>

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h>

#if defined(USE_CUDA) || defined(__CUDACC__)
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// c10 compatibility aliases
// ---------------------------------------------------------------------------
namespace c10 {

using ScalarType = torch::headeronly::ScalarType;
using DeviceType = torch::headeronly::DeviceType;
using Layout = torch::headeronly::Layout;
using MemoryFormat = torch::headeronly::MemoryFormat;
using Device = torch::stable::Device;
using StreamId = torch::stable::accelerator::StreamId;
using DeviceIndex = int32_t;
using Half = torch::headeronly::Half;
using BFloat16 = torch::headeronly::BFloat16;

template <class T>
using optional = std::optional<T>;

// Kernels written against libtorch spell the empty optional as c10::nullopt.
using std::nullopt;

}  // namespace c10

// ---------------------------------------------------------------------------
// at:: compatibility aliases
// ---------------------------------------------------------------------------
namespace at {

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using Device = torch::stable::Device;
using DeviceType = torch::headeronly::DeviceType;
using Layout = torch::headeronly::Layout;
using IntArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;
using IntHeaderOnlyArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;
using Half = torch::headeronly::Half;
using BFloat16 = torch::headeronly::BFloat16;
using Float8_e4m3fn = torch::headeronly::Float8_e4m3fn;
using Float8_e4m3fnuz = torch::headeronly::Float8_e4m3fnuz;
using Float8_e5m2 = torch::headeronly::Float8_e5m2;
using Float8_e5m2fnuz = torch::headeronly::Float8_e5m2fnuz;
using Float8_e8m0fnu = torch::headeronly::Float8_e8m0fnu;

inline constexpr ScalarType kByte = ScalarType::Byte;
inline constexpr ScalarType kChar = ScalarType::Char;
inline constexpr ScalarType kShort = ScalarType::Short;
inline constexpr ScalarType kInt = ScalarType::Int;
inline constexpr ScalarType kInt32 = ScalarType::Int;
inline constexpr ScalarType kInt16 = ScalarType::Short;
inline constexpr ScalarType kInt8 = ScalarType::Char;
inline constexpr ScalarType kLong = ScalarType::Long;
inline constexpr ScalarType kHalf = ScalarType::Half;
inline constexpr ScalarType kFloat = ScalarType::Float;
inline constexpr ScalarType kDouble = ScalarType::Double;
inline constexpr ScalarType kBool = ScalarType::Bool;
inline constexpr ScalarType kBFloat16 = ScalarType::BFloat16;

inline const Device kCPU = Device(DeviceType::CPU);
inline const Device kCUDA = Device(DeviceType::CUDA);

using torch::stable::get_num_threads;
using torch::stable::parallel_for;
using torch::stable::sum;

/// A minimal TensorOptions replacement. The stable API takes individual
/// dtype/device/layout arguments, so kernels keep the familiar builder style
/// and this struct forwards the resolved values.
class TensorOptions {
 public:
  TensorOptions() = default;

  TensorOptions& dtype(ScalarType value) {
    dtype_ = value;
    return *this;
  }
  TensorOptions& device(Device value) {
    device_ = value;
    return *this;
  }
  TensorOptions& device(DeviceType value) {
    device_ = Device(value);
    return *this;
  }
  TensorOptions& layout(Layout value) {
    layout_ = value;
    return *this;
  }
  TensorOptions& requires_grad(bool value) {
    requires_grad_ = value;
    return *this;
  }
  TensorOptions& pinned_memory(bool value) {
    pinned_memory_ = value;
    return *this;
  }

  std::optional<ScalarType> dtype() const { return dtype_; }
  std::optional<Device> device() const { return device_; }
  std::optional<Layout> layout() const { return layout_; }
  std::optional<bool> requires_grad() const { return requires_grad_; }
  std::optional<bool> pinned_memory() const { return pinned_memory_; }

 private:
  std::optional<ScalarType> dtype_;
  std::optional<Device> device_;
  std::optional<Layout> layout_;
  std::optional<bool> requires_grad_;
  std::optional<bool> pinned_memory_;
};

inline Device device_of(const Tensor& tensor) { return tensor.device(); }

inline bool is_meta(const Device& device) {
  return device.type() == DeviceType::Meta;
}

inline bool is_meta(const Tensor& tensor) { return is_meta(tensor.device()); }

inline bool is_alias_of(const Tensor& lhs, const Tensor& rhs) {
  if (!lhs.defined() || !rhs.defined()) {
    return false;
  }
  const auto* lhs_begin = static_cast<const char*>(lhs.data_ptr());
  const auto* rhs_begin = static_cast<const char*>(rhs.data_ptr());
  const size_t lhs_bytes = static_cast<size_t>(lhs.numel()) * lhs.element_size();
  const size_t rhs_bytes = static_cast<size_t>(rhs.numel()) * rhs.element_size();
  const auto* lhs_end = lhs_begin + lhs_bytes;
  const auto* rhs_end = rhs_begin + rhs_bytes;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

inline bool is_floating_point(const Tensor& tensor) {
  const ScalarType dtype = tensor.scalar_type();
  return dtype == ScalarType::Half || dtype == ScalarType::BFloat16 ||
      dtype == ScalarType::Float || dtype == ScalarType::Double;
}

inline Tensor sum(const Tensor& self, int64_t dim) {
  std::array<int64_t, 1> dims{dim};
  return torch::stable::sum(
      self,
      torch::headeronly::IntHeaderOnlyArrayRef(dims),
      false,
      std::nullopt);
}

inline Tensor repeat(const Tensor& self, IntHeaderOnlyArrayRef repeats) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(repeats)};
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::repeat", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

inline Tensor ones(
    IntHeaderOnlyArrayRef size,
    const TensorOptions& options = TensorOptions()) {
  return torch::stable::full(
      size,
      1.0,
      options.dtype(),
      options.layout(),
      options.device(),
      options.pinned_memory());
}

inline Tensor arange(int64_t end, const TensorOptions& options = TensorOptions()) {
  // aten::arange takes Scalar arguments, which the stable IValue bridge does
  // not expose. Build the (small, autotune-only) index tensor from a cached
  // host buffer instead and let the stable copy move it to the target device.
  struct Cache {
    std::mutex mutex;
    std::unordered_map<int64_t, std::shared_ptr<std::vector<int64_t>>> values;
  };
  static Cache cache;
  std::shared_ptr<std::vector<int64_t>> values;
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto& entry = cache.values[end];
    if (!entry) {
      entry = std::make_shared<std::vector<int64_t>>(static_cast<size_t>(end));
      for (int64_t index = 0; index < end; ++index) {
        (*entry)[static_cast<size_t>(index)] = index;
      }
    }
    values = entry;
  }
  const int64_t sizes[1] = {end};
  const int64_t strides[1] = {1};
  Tensor host = torch::stable::from_blob(
      values->data(),
      torch::headeronly::IntHeaderOnlyArrayRef(sizes, 1),
      torch::headeronly::IntHeaderOnlyArrayRef(strides, 1),
      Device(DeviceType::CPU),
      ScalarType::Long);
  return torch::stable::to(
      host,
      options.dtype().value_or(ScalarType::Long),
      std::nullopt,
      options.device());
}

inline Tensor empty(
    IntHeaderOnlyArrayRef size,
    const TensorOptions& options = TensorOptions()) {
  return torch::stable::empty(
      size,
      options.dtype(),
      options.layout(),
      options.device(),
      options.pinned_memory());
}

inline Tensor zeros(
    IntHeaderOnlyArrayRef size,
    const TensorOptions& options = TensorOptions()) {
  Tensor out = empty(size, options);
  return torch::stable::zero_(out);
}

inline Tensor empty_like(const Tensor& self) {
  return torch::stable::empty_like(self);
}

inline Tensor empty_like(const Tensor& self, const TensorOptions& options) {
  return torch::stable::empty(
      self.sizes(),
      options.dtype().value_or(self.scalar_type()),
      options.layout(),
      options.device().value_or(self.device()),
      options.pinned_memory());
}

/// Contiguous equivalent of `empty_like(self, memory_format=Contiguous)`.
inline Tensor empty_like_contiguous(const Tensor& self) {
  return torch::stable::empty(
      self.sizes(), self.scalar_type(), std::nullopt, self.device());
}

inline Tensor zeros_like(const Tensor& self) {
  Tensor out = torch::stable::empty_like(self);
  return torch::stable::zero_(out);
}

inline Tensor contiguous(const Tensor& self) {
  return torch::stable::contiguous(self);
}

inline Tensor to(const Tensor& self, ScalarType dtype) {
  return torch::stable::to(self, dtype);
}

inline Tensor mul(const Tensor& lhs, const Tensor& rhs) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(lhs), torch::stable::detail::from(rhs)};
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::mul", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

inline Tensor add(const Tensor& lhs, const Tensor& rhs) {
  // aten::add.Tensor carries a Scalar alpha argument, and the stable IValue
  // bridge cannot marshal Scalar. Use the AOTI per-device shims (also stable
  // ABI) which take alpha as a plain double.
  AtenTensorHandle out = nullptr;
#if defined(USE_CUDA) || defined(__CUDACC__)
  if (lhs.device().is_cuda()) {
    STABLE_TORCH_ERROR_CODE_CHECK(
        aoti_torch_cuda_add_Tensor(lhs.get(), rhs.get(), 1.0, &out));
    return Tensor(out);
  }
#endif
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_cpu_add_Tensor(lhs.get(), rhs.get(), 1.0, &out));
  return Tensor(out);
}

inline Tensor& add_(Tensor& self, const Tensor& other) {
  const Tensor out = add(self, other);
  torch::stable::copy_(self, out);
  return self;
}

inline Tensor div(const Tensor& lhs, const Tensor& rhs) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(lhs), torch::stable::detail::from(rhs)};
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::div", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

/// aten::t: reverse the dimensions of a (2-D) tensor.
inline Tensor t(const Tensor& self) {
  std::array<StableIValue, 1> stack{torch::stable::detail::from(self)};
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::t", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

inline Tensor view(const Tensor& self, IntHeaderOnlyArrayRef sizes) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(sizes)};
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::view", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

#if defined(USE_CUDA) || defined(__CUDACC__)
namespace cuda {

class CompatibleCUDAStream {
 public:
  explicit CompatibleCUDAStream(cudaStream_t stream) : stream_(stream) {}
  cudaStream_t stream() const { return stream_; }
  operator cudaStream_t() const { return stream_; }

 private:
  cudaStream_t stream_;
};

inline CompatibleCUDAStream getCurrentCUDAStream(int32_t device_index = -1) {
  if (device_index < 0) {
    device_index = torch::stable::accelerator::getCurrentDeviceIndex();
  }
  void* stream = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream));
  return CompatibleCUDAStream(static_cast<cudaStream_t>(stream));
}

inline int32_t current_device_index() {
  return torch::stable::accelerator::getCurrentDeviceIndex();
}

inline void record_stream(const Tensor&, cudaStream_t) {
  // The stable ABI does not expose Tensor::record_stream. Callers keep the
  // workspace alive for the duration of the op; this is a lifetime hint, not
  // a correctness requirement.
}

inline cublasHandle_t getCurrentCUDABlasHandle() {
  void* handle = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_get_current_cuda_blas_handle(&handle));
  return static_cast<cublasHandle_t>(handle);
}

inline const cudaDeviceProp* getDeviceProperties(int32_t device_index) {
  static thread_local cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device_index) != cudaSuccess) {
    return nullptr;
  }
  return &properties;
}

class CUDAGuard {
 public:
  explicit CUDAGuard(int32_t device_index) : guard_(device_index) {}
  explicit CUDAGuard(const Device& device)
      : guard_(device.index() < 0 ? current_device_index() : device.index()) {}
  explicit CUDAGuard(const Tensor& tensor)
      : guard_(tensor.get_device_index()) {}

  void set_index(int32_t device_index) { guard_.set_index(device_index); }

 private:
  torch::stable::accelerator::DeviceGuard guard_;
};

using OptionalCUDAGuard = CUDAGuard;

}  // namespace cuda
#endif  // USE_CUDA || __CUDACC__

}  // namespace at

namespace torch {

using Tensor = torch::stable::Tensor;
using TensorOptions = at::TensorOptions;
using ScalarType = torch::headeronly::ScalarType;
using IntArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;
using Device = torch::stable::Device;
using DeviceType = torch::headeronly::DeviceType;

inline constexpr ScalarType kFloat32 = ScalarType::Float;
inline constexpr ScalarType kFloat16 = ScalarType::Half;
inline constexpr ScalarType kDouble = ScalarType::Double;
inline constexpr ScalarType kFloat = ScalarType::Float;
inline constexpr ScalarType kHalf = ScalarType::Half;
inline constexpr ScalarType kInt = ScalarType::Int;
inline constexpr ScalarType kInt32 = ScalarType::Int;
inline constexpr ScalarType kInt16 = ScalarType::Short;
inline constexpr ScalarType kInt8 = ScalarType::Char;
inline constexpr ScalarType kLong = ScalarType::Long;
inline constexpr ScalarType kShort = ScalarType::Short;
inline constexpr ScalarType kByte = ScalarType::Byte;
inline constexpr ScalarType kU8 = ScalarType::Byte;
inline constexpr ScalarType kChar = ScalarType::Char;
inline constexpr ScalarType kBool = ScalarType::Bool;
inline constexpr DeviceType kCPU = DeviceType::CPU;
inline constexpr DeviceType kCUDA = DeviceType::CUDA;

using at::empty;
using at::empty_like;
using at::zeros;
using at::zeros_like;

}  // namespace torch

#if defined(USE_CUDA) || defined(__CUDACC__)
namespace c10 {
namespace cuda {
using CUDAGuard = at::cuda::CUDAGuard;
using OptionalCUDAGuard = at::cuda::OptionalCUDAGuard;
using CUDAStream = at::cuda::CompatibleCUDAStream;
inline int32_t current_device() {
  return torch::stable::accelerator::getCurrentDeviceIndex();
}
}  // namespace cuda
}  // namespace c10
#endif  // USE_CUDA || __CUDACC__

// The original ATen kernels rely on ADL to find at::device_of because their
// tensor type lived in namespace at. torch::stable::Tensor does not, so make
// the helper visible at global scope for the migrated sources.
using at::device_of;

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
#define TORCH_CHECK(...) STD_TORCH_CHECK(__VA_ARGS__)
#define TORCH_INTERNAL_ASSERT(...) STD_TORCH_CHECK(__VA_ARGS__)
#define TORCH_CHECK_NOT_IMPLEMENTED(...) STD_TORCH_CHECK(false, __VA_ARGS__)
#define AT_ERROR(...) STD_TORCH_CHECK(false, __VA_ARGS__)
#define TORCH_WARN(...) \
  do {                  \
  } while (0)

#if defined(USE_CUDA) || defined(__CUDACC__)
#define GPTQMODEL_CUDA_CHECK(EXPR)                                       \
  do {                                                                   \
    cudaError_t _gptqmodel_cuda_status = (EXPR);                         \
    STD_TORCH_CHECK(                                                     \
        _gptqmodel_cuda_status == cudaSuccess,                           \
        "CUDA error: ",                                                  \
        cudaGetErrorString(_gptqmodel_cuda_status));                     \
  } while (0)

#define C10_CUDA_CHECK(EXPR) GPTQMODEL_CUDA_CHECK(EXPR)
#define AT_CUDA_CHECK(EXPR) GPTQMODEL_CUDA_CHECK(EXPR)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() GPTQMODEL_CUDA_CHECK(cudaGetLastError())
#endif  // USE_CUDA || __CUDACC__
