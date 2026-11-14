// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Torch-version-specific adapter for the Marlin GEMM.
//
// The heavy Marlin core is built against the LibTorch stable ABI, so one core
// binary serves every newer torch release. That surface does not expose
// `Tensor::record_stream`, which keeps caller-owned output and scratch buffers
// alive while a kernel on another stream is still writing them. This small
// adapter is rebuilt for the installed torch, keeps the public operator name
// and schema, records those buffers with the real allocator hook, and forwards
// to the stable core through the dispatcher. Python callers and direct
// `torch.ops` / native callers therefore get identical lifetime protection.
//
// `record_stream` only registers allocator usage; the completion event is
// created when the block is freed, so recording before the launch is safe and
// needs no post-launch cleanup.

#include <torch/library.h>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>
#include <vector>

#if MARLIN_ADAPTER_BF16
static constexpr const char* kCoreOp =
    "gptqmodel_marlin_core_bf16::gptq_marlin_gemm_bf16";
static constexpr const char* kPublicOp = "gptq_marlin_gemm_bf16";
static constexpr const char* kSchema =
    "gptq_marlin_gemm_bf16(Tensor a, Tensor(a!)? c, Tensor b_q_weight, Tensor? b_bias, Tensor b_scales, "
    "Tensor? global_scale, Tensor? b_zeros, Tensor? g_idx, Tensor? perm, Tensor(b!) workspace, int b_q_type_id, "
    "int size_m, int size_n, int size_k, bool is_k_full=True, bool use_atomic_add=False, "
    "bool use_fp32_reduce=False, bool is_zp_float=False, Tensor(c!)? c_tmp=None, "
    "Tensor(d!)? a_tmp=None) -> Tensor(a!)";
#else
static constexpr const char* kCoreOp =
    "gptqmodel_marlin_core_fp16::gptq_marlin_gemm_fp16";
static constexpr const char* kPublicOp = "gptq_marlin_gemm_fp16";
static constexpr const char* kSchema =
    "gptq_marlin_gemm_fp16(Tensor a, Tensor(a!)? c, Tensor b_q_weight, Tensor? b_bias, Tensor b_scales, "
    "Tensor? global_scale, Tensor? b_zeros, Tensor? g_idx, Tensor? perm, Tensor(b!) workspace, int b_q_type_id, "
    "int size_m, int size_n, int size_k, bool is_k_full=True, bool use_atomic_add=False, "
    "bool use_fp32_reduce=False, bool is_zp_float=False, Tensor(c!)? c_tmp=None, "
    "Tensor(d!)? a_tmp=None) -> Tensor(a!)";
#endif

namespace {

const c10::OperatorHandle& core_gemm_handle() {
  // Resolved lazily: the adapter is loaded after its stable core, but keeping
  // the lookup here avoids any static-initialization order dependency.
  static const c10::OperatorHandle handle =
      c10::Dispatcher::singleton().findSchemaOrThrow(kCoreOp, "");
  return handle;
}

c10::IValue optional_tensor(const std::optional<at::Tensor>& value) {
  return value.has_value() ? c10::IValue(*value) : c10::IValue();
}

at::Tensor record_and_forward(
    at::Tensor a, std::optional<at::Tensor> c_or_none, at::Tensor b_q_weight,
    std::optional<at::Tensor> const& b_bias_or_none, at::Tensor b_scales,
    std::optional<at::Tensor> const& global_scale_or_none,
    std::optional<at::Tensor> const& b_zeros_or_none,
    std::optional<at::Tensor> const& g_idx_or_none,
    std::optional<at::Tensor> const& perm_or_none, at::Tensor workspace,
    int64_t b_q_type_id, int64_t size_m, int64_t size_n, int64_t size_k,
    bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float,
    std::optional<at::Tensor> c_tmp_or_none,
    std::optional<at::Tensor> a_tmp_or_none) {
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  const auto stream = at::cuda::getCurrentCUDAStream(
      static_cast<c10::DeviceIndex>(a.get_device()));
  // These are the buffers the kernel writes. Recording them on the executing
  // stream keeps the allocator from handing their storage to another stream
  // before the launch completes.
  if (c_or_none.has_value()) c_or_none->record_stream(stream);
  if (c_tmp_or_none.has_value()) c_tmp_or_none->record_stream(stream);
  if (a_tmp_or_none.has_value()) a_tmp_or_none->record_stream(stream);

  std::vector<c10::IValue> stack = {
      a,
      optional_tensor(c_or_none),
      b_q_weight,
      optional_tensor(b_bias_or_none),
      b_scales,
      optional_tensor(global_scale_or_none),
      optional_tensor(b_zeros_or_none),
      optional_tensor(g_idx_or_none),
      optional_tensor(perm_or_none),
      workspace,
      b_q_type_id,
      size_m,
      size_n,
      size_k,
      is_k_full,
      use_atomic_add,
      use_fp32_reduce,
      is_zp_float,
      optional_tensor(c_tmp_or_none),
      optional_tensor(a_tmp_or_none),
  };
  core_gemm_handle().callBoxed(stack);
  return stack[0].toTensor();
}

}  // namespace

// The stable core already registered a DEF for this namespace; a fragment adds
// the public schema without competing for the single-DEF slot.
#if MARLIN_ADAPTER_BF16
TORCH_LIBRARY_FRAGMENT(gptqmodel_marlin_bf16, m) {
  m.def(kSchema);
}

TORCH_LIBRARY_IMPL(gptqmodel_marlin_bf16, CUDA, m) {
  m.impl(kPublicOp, &record_and_forward);
}
#else
TORCH_LIBRARY_FRAGMENT(gptqmodel_marlin_fp16, m) {
  m.def(kSchema);
}

TORCH_LIBRARY_IMPL(gptqmodel_marlin_fp16, CUDA, m) {
  m.impl(kPublicOp, &record_and_forward);
}
#endif
