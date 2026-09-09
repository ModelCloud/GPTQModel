// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#include "quant_abi.h"
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

static_assert(sizeof(QvqQuantTensor) == 56);
static_assert(sizeof(QvqQuantConfig) == 112);
static_assert(offsetof(QvqQuantConfig, schedule) == 104);

namespace {
void check(bool ok, const char* why) { if (!ok) throw std::invalid_argument(why); }
template<class F> int boundary(char* error, uint64_t cap, F fn) noexcept {
  if (error && cap) error[0] = 0;
  try { fn(); return 0; }
  catch (const std::exception& e) {
    if (error && cap) std::snprintf(error, cap, "%s", e.what());
  } catch (...) { if (error && cap) std::snprintf(error, cap, "unknown native exception"); }
  return -1;
}
at::ScalarType dtype(uint32_t d) {
  switch(d) {
    case QVQ_F16: return at::kHalf; case QVQ_BF16: return at::kBFloat16;
    case QVQ_F32: return at::kFloat; case QVQ_I32: return at::kInt;
    case QVQ_U8: return at::kByte; default: throw std::invalid_argument("invalid ABI dtype");
  }
}
c10::IValue optional_dtype(uint32_t d) { return d ? c10::IValue(dtype(d)) : c10::IValue(); }
c10::IValue invoke(const char* name, std::vector<c10::IValue> args) {
  c10::Dispatcher::singleton().findSchemaOrThrow(name, "").callBoxed(&args);
  check(args.size() == 1, "operator must return exactly one value");
  return args[0];
}
at::Tensor tensor(const QvqQuantTensor& t, int device) {
  if (!t.data) {
    check(!t.bytes && !t.rank && !t.dtype && !t.shape[0] && !t.shape[1] &&
          !t.shape[2] && !t.shape[3], "absent tensor must be zero initialized");
    return {};
  }
  check(t.rank >= 1 && t.rank <= 4, "invalid tensor rank");
  auto type = dtype(t.dtype);
  uint64_t bytes = c10::elementSize(type);
  for (uint32_t i = 0; i < 4; ++i) {
    if (i >= t.rank) { check(t.shape[i] == 0, "unused dimensions must be zero"); continue; }
    check(t.shape[i] > 0 && uint64_t(t.shape[i]) <= uint64_t(INT64_MAX) / bytes,
          "invalid or overflowing tensor dimension");
    bytes *= uint64_t(t.shape[i]);
  }
  check(bytes == t.bytes, "tensor byte count mismatch");
  check(reinterpret_cast<uintptr_t>(t.data) % 16 == 0, "tensor requires 16-byte alignment");
  cudaPointerAttributes attr{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attr, t.data));
  check(attr.type == cudaMemoryTypeDevice && attr.device == device, "tensor is on wrong device");
  check(uintptr_t(t.data) <= UINTPTR_MAX - t.bytes, "tensor address overflow");
  return at::from_blob(t.data, at::IntArrayRef(t.shape, t.rank),
                      at::TensorOptions().dtype(type).device(at::kCUDA, device));
}
void outside_capture(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  check(status == cudaStreamCaptureStatusNone, "operation forbidden during CUDA capture");
}
void config_check(const QvqQuantConfig& c) {
  check(c.struct_bytes == sizeof(c) && c.abi_version == 1 && !c.reserved,
        "unsupported quant ABI configuration");
  check(c.operation >= 1 && c.operation <= 6, "unknown operation");
  check(c.device >= 0, "invalid device ordinal");
  if (c.operation == QVQ_MACHETE_MM)
    check(c.schedule && c.schedule[0], "Machete requires an explicit compiled schedule");
  else check(!c.schedule, "schedule is only valid for Machete MM");
  if (c.operation != QVQ_SWORDFISH_DECODE)
    check(!(c.mode || c.m_tiles || c.split_k || c.ctas || c.cta_quad || c.threads || c.stages),
          "decode tuning is only valid for Swordfish decode");
  check(c.transpose <= 1 && (!c.transpose || c.operation == QVQ_SWORDFISH_DEQUANT),
        "transpose is only valid for Swordfish dequantization");
  check(c.cta_quad == 0 || c.cta_quad == 1, "cta_quad must be zero or one");
  if (c.operation != QVQ_SWORDFISH_PREFILL)
    check(!c.tile_n && !c.chunk_m, "prefill tuning only applies to Swordfish prefill");
}
} // namespace

struct QvqQuantPlan {
  at::cuda::CUDAGraph graph{true};
  cudaStream_t stream;
  int device;
  std::mutex mutex;
};
extern "C" uint32_t qvq_quant_abi_version() { return 1; }
extern "C" int qvq_machete_schedules(uint32_t a, int64_t b, const uint32_t* ds,
    char* buffer, uint64_t cap, uint64_t* required, char* error, uint64_t ec) {
  return boundary(error, ec, [&] {
    check(ds && required && (buffer || !cap), "null schedule query argument");
    auto result = invoke("gptqmodel_machete::machete_supported_schedules",
        {dtype(a), b, optional_dtype(ds[0]), optional_dtype(ds[1]),
         optional_dtype(ds[2]), optional_dtype(ds[3]), optional_dtype(ds[4])});
    std::string names;
    for (const auto& entry : result.toListRef()) { names += entry.toStringRef(); names += '\n'; }
    *required = names.size() + 1;
    if (!buffer && !cap) return;
    check(cap >= *required, "schedule buffer too small");
    std::memcpy(buffer, names.c_str(), *required);
  });
}
extern "C" int qvq_quant_prepare(const QvqQuantConfig* cp, const QvqQuantTensor* inputs,
    uint32_t count, QvqQuantTensor output, void* raw_stream, QvqQuantPlan** handle,
    char* error, uint64_t ec) {
  if (handle) *handle = nullptr;
  return boundary(error, ec, [&] {
    check(cp && handle && inputs && raw_stream, "null prepare argument/nondefault stream required");
    check(cp->struct_bytes == sizeof(*cp) && cp->abi_version == 1, "unsupported quant ABI header");
    const auto c = *cp;
    config_check(c);
    c10::cuda::CUDAGuard device_guard(c.device);
    auto stream = static_cast<cudaStream_t>(raw_stream);
    outside_capture(stream);
    int stream_device = -1;
    C10_CUDA_CHECK(cudaStreamGetDevice(stream, &stream_device));
    check(stream_device == c.device, "stream belongs to a different device");
    c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromExternal(stream, c.device));
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, c.device));
    const int sm = prop.major * 10 + prop.minor;
    check(c.operation <= 2 ? sm == 90 : (sm == 100 || sm == 103 || sm == 110),
          "Machete requires SM90; Swordfish requires SM100/103/110");
    const uint32_t expected[] = {0, 6, 2, 4, 4, 2, 3};
    check(count == expected[c.operation], "incorrect input count");
    auto out = tensor(output, c.device);
    check(out.defined(), "output is required");
    std::vector<c10::IValue> ts;
    for (uint32_t i = 0; i < count; ++i) {
      auto t = tensor(inputs[i], c.device);
      if (t.defined()) {
        const auto lo = uintptr_t(inputs[i].data), hi = lo + inputs[i].bytes;
        const auto olo = uintptr_t(output.data), ohi = olo + output.bytes;
        check(hi <= olo || ohi <= lo, "input/output aliasing is forbidden");
      }
      ts.push_back(t.defined() ? c10::IValue(t) : c10::IValue());
    }
    const char* op = nullptr;
    std::vector<c10::IValue> args;
    switch (c.operation) {
      case QVQ_MACHETE_MM:
        op = "gptqmodel_machete::machete_mm";
        args = {ts[0], ts[1], c.weight_type, optional_dtype(c.output_dtype), ts[2], ts[3],
                c.group_size, ts[4], ts[5], std::string(c.schedule)}; break;
      case QVQ_MACHETE_PREPACK:
        op = "gptqmodel_machete::machete_prepack_B";
        args = {ts[0], dtype(c.activation_dtype), c.weight_type,
                ts[1].isNone() ? c10::IValue() : c10::IValue(ts[1].toTensor().scalar_type())}; break;
      case QVQ_SWORDFISH_DECODE:
        op = "gptqmodel_swordfish::swordfish_decode_explicit";
        args = {ts[0], ts[1], ts[2], ts[3], int64_t(c.num_bits), c.group_size, c.k, c.n,
                int64_t(c.mode), int64_t(c.m_tiles), int64_t(c.split_k), int64_t(c.ctas),
                bool(c.cta_quad), int64_t(c.threads), int64_t(c.stages)}; break;
      case QVQ_SWORDFISH_PREFILL:
        op = "gptqmodel_swordfish::swordfish_prefill_explicit";
        args = {ts[0], ts[1], ts[2], ts[3], int64_t(c.num_bits), c.group_size, c.k, c.n,
                int64_t(c.tile_n), int64_t(c.chunk_m)}; break;
      case QVQ_SWORDFISH_PREPACK:
        op = "gptqmodel_swordfish::swordfish_prepack_B";
        args = {ts[0], ts[1], c.k, c.n, int64_t(c.num_bits)}; break;
      case QVQ_SWORDFISH_DEQUANT:
        op = "gptqmodel_swordfish::swordfish_dequant_dense";
        args = {ts[0], ts[1], ts[2], int64_t(c.num_bits), c.group_size, c.k, c.n, bool(c.transpose)}; break;
    }
    auto run = [&] {
      auto result = invoke(op, args).toTensor();
      check(result.sizes() == out.sizes() && result.scalar_type() == out.scalar_type(),
            "output shape/dtype differs from native operator result");
      out.copy_(result);
    };
    for (int i = 0; i < 3; ++i) run();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    auto plan = std::make_unique<QvqQuantPlan>();
    plan->device = c.device; plan->stream = stream;
    plan->graph.capture_begin();
    try { run(); plan->graph.capture_end(); plan->graph.instantiate(); }
    catch (...) {
      cudaStreamCaptureStatus status;
      if (cudaStreamIsCapturing(stream, &status) == cudaSuccess && status != cudaStreamCaptureStatusNone) {
        try { plan->graph.capture_end(); } catch (...) {}
      }
      throw;
    }
    *handle = plan.release();
  });
}
extern "C" int qvq_quant_launch(QvqQuantPlan* p, void* raw_stream, char* error, uint64_t ec) {
  return boundary(error, ec, [&] {
    check(p && raw_stream == p->stream, "plan requires its preparation stream");
    std::lock_guard<std::mutex> lock(p->mutex);
    c10::cuda::CUDAGuard guard(p->device);
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p->stream, p->device));
    cudaStreamCaptureStatus status;
    cudaGraph_t parent;
    const cudaGraphNode_t* deps;
    const cudaGraphEdgeData* edges;
    size_t count;
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo(p->stream, &status, nullptr, &parent, &deps, &edges, &count));
    if (status == cudaStreamCaptureStatusNone) { p->graph.replay(); return; }
    check(status == cudaStreamCaptureStatusActive, "caller capture is invalidated");
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeGraph;
    params.graph.graph = p->graph.raw_cuda_graph();
    cudaGraphNode_t child;
    C10_CUDA_CHECK(cudaGraphAddNode(&child, parent, deps, edges, count, &params));
    C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(p->stream, &child, nullptr, 1,
                                                       cudaStreamSetCaptureDependencies));
  });
}
extern "C" int qvq_quant_destroy(QvqQuantPlan* p, char* error, uint64_t ec) {
  return boundary(error, ec, [&] {
    if (!p) return;
    c10::cuda::CUDAGuard guard(p->device);
    outside_capture(p->stream);
    C10_CUDA_CHECK(cudaStreamSynchronize(p->stream));
    delete p;
  });
}
