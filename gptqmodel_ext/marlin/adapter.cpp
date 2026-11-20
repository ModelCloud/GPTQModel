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
// the same stack to the stable core through the dispatcher. Python callers and
// direct `torch.ops` / native callers therefore get identical lifetime
// protection.
//
// `record_stream` only registers allocator usage; the completion event is
// created when the block is freed, so recording before the launch is safe and
// needs no post-launch cleanup.

#include <torch/library.h>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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

// Argument positions in `kSchema` of the buffers the kernel writes.
constexpr size_t kCArg = 1;
constexpr size_t kCTmpArg = 18;
constexpr size_t kATmpArg = 19;

namespace {

const c10::OperatorHandle& core_gemm_handle() {
  // Resolved lazily: the adapter is loaded after its stable core, but keeping
  // the lookup here avoids any static-initialization order dependency.
  static const c10::OperatorHandle handle =
      c10::Dispatcher::singleton().findSchemaOrThrow(kCoreOp, "");
  return handle;
}

void record_stream_if_present(c10::Stack& stack, size_t index, at::Stream stream) {
  if (index >= stack.size() || stack[index].isNone()) {
    return;
  }
  stack[index].toTensor().record_stream(stream);
}

// Boxed on purpose: the stack already holds the caller's arguments, so the
// adapter can record streams and hand the same stack to the core without an
// extra unbox/box round trip.
void record_and_forward(const c10::OperatorHandle&, c10::Stack* stack) {
  const at::Tensor& a = stack->at(0).toTensor();
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  const auto stream = at::cuda::getCurrentCUDAStream(
      static_cast<c10::DeviceIndex>(a.get_device()));
  // These are the buffers the kernel writes. Recording them on the executing
  // stream keeps the allocator from handing their storage to another stream
  // before the launch completes.
  record_stream_if_present(*stack, kCArg, stream);
  record_stream_if_present(*stack, kCTmpArg, stream);
  record_stream_if_present(*stack, kATmpArg, stream);

  core_gemm_handle().callBoxed(*stack);
}

}  // namespace

// The stable core already registered a DEF for this namespace; a fragment adds
// the public schema without competing for the single-DEF slot.
#if MARLIN_ADAPTER_BF16
TORCH_LIBRARY_FRAGMENT(gptqmodel_marlin_bf16, m) {
  m.def(kSchema);
}

TORCH_LIBRARY_IMPL(gptqmodel_marlin_bf16, CUDA, m) {
  m.impl(kPublicOp, torch::CppFunction::makeFromBoxedFunction<&record_and_forward>());
}
#else
TORCH_LIBRARY_FRAGMENT(gptqmodel_marlin_fp16, m) {
  m.def(kSchema);
}

TORCH_LIBRARY_IMPL(gptqmodel_marlin_fp16, CUDA, m) {
  m.impl(kPublicOp, torch::CppFunction::makeFromBoxedFunction<&record_and_forward>());
}
#endif
