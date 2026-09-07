// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#include "qvq_window_abi.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <memory>
#include <mutex>

extern "C" void qvq_rank8_epilogue_no_hadamard(
    const at::Tensor& hidden,
    const at::Tensor& rank8_b,
    const at::Tensor& base,
    const at::Tensor& scale_v,
    const at::Tensor& bias,
    at::Tensor& output,
    cudaStream_t stream);

namespace {
thread_local bool owned_capture = false;
void check(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
bool power2(uint32_t value) { return value && !(value & (value - 1)); }
at::Tensor invoke(const char* name, std::vector<c10::IValue> args) {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
  op.callBoxed(&args);
  return args.at(0).toTensor();
}
at::Tensor tensor(QvqWindowBuffer buffer, at::IntArrayRef shape,
                  at::ScalarType dtype, int device) {
  uint64_t count = 1;
  for (auto dim : shape) count *= dim;
  check(buffer.data && buffer.bytes == count * c10::elementSize(dtype),
        "window ABI buffer byte count does not match its tensor contract");
  check(reinterpret_cast<uintptr_t>(buffer.data) % c10::elementSize(dtype) == 0,
        "window ABI buffer alignment is invalid");
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, buffer.data));
  check(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
        "native window buffers must reside on the input CUDA device");
  return at::from_blob(buffer.data, shape, at::TensorOptions().dtype(dtype).device(at::kCUDA, device));
}
at::ScalarType floating_storage_dtype(QvqWindowBuffer buffer, uint64_t count) {
  check(buffer.data && (buffer.bytes == count * sizeof(at::Half) ||
                        buffer.bytes == count * sizeof(float)),
        "window ABI floating buffer must contain contiguous FP16 or FP32 values");
  return buffer.bytes == count * sizeof(at::Half) ? at::kHalf : at::kFloat;
}
at::Tensor hadamard(const at::Tensor& x, c10::IValue pre,
                    c10::IValue post, c10::IValue bias, int64_t mode) {
  return invoke("gptqmodel_qvq::hadamard",
                {x, pre, post, bias, mode, false, false, false, c10::IValue(), int64_t(0)});
}
} // namespace

static int qvq_p32_window_linear_impl(
    QvqWindowBuffer x, QvqWindowBuffer window, QvqWindowBuffer banks,
    QvqWindowBuffer levels, QvqWindowBuffer su, QvqWindowBuffer sv,
    QvqWindowBuffer bias, QvqWindowBuffer rank8_a, QvqWindowBuffer rank8_b,
    QvqWindowBuffer y, const QvqP32WindowConfig* config, void* cuda_stream,
    char* error, uint64_t error_capacity, const at::Tensor& prepared_rank8_a,
    const at::Tensor& prepared_rank8_b, cudaStream_t recovery_stream,
    cudaEvent_t recovery_ready, cudaEvent_t recovery_done) {
  try {
    check(config && config->abi_version == 3 &&
              config->struct_bytes >= offsetof(QvqP32WindowConfig, recovery_kernel) &&
              config->struct_bytes <= sizeof(*config),
          "unsupported window ABI version or configuration size");
    const auto& c = *config;
    const bool has_recovery_policy =
        c.struct_bytes >= offsetof(QvqP32WindowConfig, recovery_kernel) +
            2 * sizeof(uint32_t);
    const uint32_t recovery_kernel = has_recovery_policy ? c.recovery_kernel : 0;
    const uint32_t recovery_projection =
        has_recovery_policy ? c.recovery_projection : 0;
    // The WGMMA decoder consumes 256-wide K/N tiles.  Hadamard transforms
    // additionally require a power-of-two width, but the transform-free
    // composite path must admit production projections such as Qwen's
    // K=5120/N=17408 (both are valid 256-wide tile shapes).
    check(c.m >= 1 && c.m <= 8192 && c.k >= 2048 && c.k <= 16384 && c.k % 256 == 0
          && c.n >= 256 && c.n <= 17408 && c.n % 256 == 0,
          "native window ABI requires M1..8192 and 256-wide K2048..16384/N256..17408");
    check(c.transition_bits >= 4 && c.transition_bits <= 7 && c.bank_alt_id <= 3,
          "invalid P32 transition bits or alternative bank id");
    check(c.min_m >= 1 && c.min_m <= c.m && c.max_m >= c.m && c.max_m <= 8192,
          "M is outside the native window policy");
    check(c.block_k == 256 && c.pipeline_stages == 2 && c.split_k == 1,
          "native reference ABI requires BK256, stages2, split1");
    check(c.input_hadamard <= 1 && c.output_hadamard <= 1 && c.rank8_enabled <= 1,
          "native window flags must be zero or one");
    check(recovery_kernel <= 1 && recovery_projection <= 1,
          "native rank8 policy supports separate_reference or fused_epilogue with separate_reference or concurrent_reference projection");
    check(recovery_kernel != 1 || !c.output_hadamard,
          "native fused rank8 epilogue requires output_hadamard=false");
    check(!c.input_hadamard || power2(c.k),
          "native input Hadamard requires power-of-two K");
    check(!c.output_hadamard || power2(c.n),
          "native output Hadamard requires power-of-two N");
    check((c.algorithm == 1 && c.block_m == 0 && c.block_n == 0 && c.warp_groups == 0)
          || (c.algorithm == 2 && (c.block_m == 32 || c.block_m == 64 || c.block_m == 128)
              && (c.block_n == 64 || c.block_n == 128)
              && (c.warp_groups == 0 || c.warp_groups == c.block_n / 64)),
          "unsupported native Hopper geometry");
    check(x.data, "native window input pointer is null");
    cudaPointerAttributes attributes{};
    C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, x.data));
    check(attributes.type == cudaMemoryTypeDevice, "native window input must be CUDA device memory");
    const int device = attributes.device;
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(cuda_stream), device));
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(static_cast<cudaStream_t>(cuda_stream), &capture));
    check(capture == cudaStreamCaptureStatusNone || owned_capture,
          "native reference ABI does not own external CUDA capture workspace");
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    check(properties.major == 9 && properties.minor == 0, "native window ABI requires SM90");
    at::NoGradGuard no_grad;
    check(!c.rank8_enabled || !at::globalContext().allowTF32CuBLAS(),
          "native rank8 reference requires TF32 disabled");
    const int64_t tiles = int64_t(c.k) * c.n / 256;
    auto input = tensor(x, {c.m, c.k}, at::kHalf, device);
    auto words = tensor(window, {tiles, 4 * c.transition_bits}, at::kInt, device);
    auto selectors = tensor(banks, {tiles}, at::kByte, device);
    auto codebook = tensor(levels, {256}, at::kHalf, device);
    auto scale_u_raw = tensor(su, {c.k}, floating_storage_dtype(su, c.k), device);
    auto scale_v_raw = tensor(sv, {c.n}, floating_storage_dtype(sv, c.n), device);
    auto output = tensor(y, {c.m, c.n}, at::kHalf, device);
    check(bias.data || bias.bytes == 0, "absent bias must have zero bytes");
    auto output_bias_raw = bias.data
        ? tensor(bias, {c.n}, floating_storage_dtype(bias, c.n), device)
        : at::Tensor();
    at::Tensor a, b;
    if (c.rank8_enabled) {
      a = tensor(rank8_a, {c.k, 8}, at::kHalf, device);
      b = tensor(rank8_b, {8, c.n}, at::kHalf, device);
      if (prepared_rank8_a.defined() || prepared_rank8_b.defined()) {
        check(prepared_rank8_a.defined() && prepared_rank8_b.defined(),
              "native prepared rank8 factors must be provided as a pair");
        check(prepared_rank8_a.scalar_type() == at::kFloat &&
                  prepared_rank8_b.scalar_type() == at::kFloat &&
                  prepared_rank8_a.device() == a.device() &&
                  prepared_rank8_b.device() == b.device() &&
                  prepared_rank8_a.sizes() == a.sizes() &&
                  prepared_rank8_b.sizes() == b.sizes(),
              "native prepared rank8 factor cache has an invalid layout");
      }
    }
    std::vector<QvqWindowBuffer> sources{x, window, banks, levels, su, sv, bias};
    if (c.rank8_enabled) { sources.push_back(rank8_a); sources.push_back(rank8_b); }
    const auto output_address = reinterpret_cast<uintptr_t>(y.data);
    for (auto source : sources) {
      if (!source.data) continue;
      const auto address = reinterpret_cast<uintptr_t>(source.data);
      check(output_address >= address + source.bytes || address >= output_address + y.bytes,
            "native window output must not overlap its inputs or artifact");
    }
    // All structural/device checks precede the first device computation.
    auto scale_u = scale_u_raw.to(at::kHalf);
    auto scale_v = scale_v_raw.to(at::kFloat);
    auto output_bias = output_bias_raw.defined() ? output_bias_raw.to(at::kFloat) : at::Tensor();
    auto transformed = c.input_hadamard
        ? hadamard(input, scale_u, c10::IValue(), c10::IValue(), 2)
        : input * scale_u;
    at::Tensor inner;
    if (c.algorithm == 1) {
      std::vector<at::Tensor> rows;
      for (int64_t start = 0; start < c.m; start += 16) {
        auto tile = transformed.slice(0, start, std::min<int64_t>(start + 16, c.m));
        const int64_t count = tile.size(0);
        if (count < 16) tile = at::constant_pad_nd(tile, {0, 0, 0, 16 - count});
        rows.push_back(invoke("gptqmodel_qvq_wgmma::p32_window_m16_tma",
            {tile, words, codebook, selectors, int64_t(c.transition_bits), int64_t(c.n),
             int64_t(c.bank_alt_id), int64_t(1)}).slice(0, 0, count));
      }
      inner = rows.size() == 1 ? rows[0] : at::cat(rows, 0);
    } else {
      const int64_t padded_m = ((c.m + c.block_m - 1) / c.block_m) * c.block_m;
      auto padded = at::constant_pad_nd(transformed, {0, 0, 0, padded_m - c.m});
      inner = invoke("gptqmodel_qvq_wgmma::p32_window_tuned",
          {padded, words, codebook, selectors, int64_t(c.transition_bits), int64_t(c.n),
           int64_t(c.bank_alt_id), int64_t(c.block_m), int64_t(c.block_n)})
          .view({padded_m, c.n}).slice(0, 0, c.m);
    }
    bool wrote_output = false;
    const bool concurrent_projection = c.rank8_enabled && recovery_projection == 1;
    if (concurrent_projection && recovery_kernel == 1 && !c.output_hadamard) {
      // Prepared graph execution supplies a dedicated stream and event pair.
      // Raw linear calls intentionally leave these null and execute the same
      // reference projection on the caller stream.
      if (recovery_stream != nullptr) {
        C10_CUDA_CHECK(cudaEventRecord(recovery_ready, static_cast<cudaStream_t>(cuda_stream)));
        C10_CUDA_CHECK(cudaStreamWaitEvent(recovery_stream, recovery_ready, 0));
      }
      const auto& a_float = prepared_rank8_a.defined() ? prepared_rank8_a : a.to(at::kFloat);
      // Keep projection on the existing captured FP32 GEMM path, then use a
      // direct-store epilogue. This avoids the uncompetitive serial custom
      // projection while preserving the explicit FP16 hidden boundary.
      at::Tensor hidden;
      if (recovery_stream != nullptr) {
        c10::cuda::CUDAStreamGuard recovery_guard(c10::cuda::getStreamFromExternal(
            recovery_stream, device));
        hidden = at::mm(transformed.to(at::kFloat), a_float).to(at::kHalf);
        C10_CUDA_CHECK(cudaEventRecord(recovery_done, recovery_stream));
      } else {
        hidden = at::mm(transformed.to(at::kFloat), a_float).to(at::kHalf);
      }
      if (recovery_stream != nullptr) {
        C10_CUDA_CHECK(cudaStreamWaitEvent(static_cast<cudaStream_t>(cuda_stream), recovery_done, 0));
      }
      // The transform-free composite path uses one graph-safe CUDA epilogue:
      // it consumes X'A at the explicit FP16 hidden boundary, expands in
      // FP32, adds the decoded base, and stores the final FP16 result directly
      // into the caller-owned output buffer.
      qvq_rank8_epilogue_no_hadamard(
          hidden, b, inner, scale_v, output_bias, output,
          static_cast<cudaStream_t>(cuda_stream));
      wrote_output = true;
    } else if (c.rank8_enabled) {
      const auto& a_float = prepared_rank8_a.defined() ? prepared_rank8_a : a.to(at::kFloat);
      const auto& b_float = prepared_rank8_b.defined() ? prepared_rank8_b : b.to(at::kFloat);
      at::Tensor hidden;
      if (concurrent_projection && recovery_stream != nullptr) {
        C10_CUDA_CHECK(cudaEventRecord(recovery_ready, static_cast<cudaStream_t>(cuda_stream)));
        C10_CUDA_CHECK(cudaStreamWaitEvent(recovery_stream, recovery_ready, 0));
        {
          c10::cuda::CUDAStreamGuard recovery_guard(c10::cuda::getStreamFromExternal(
              recovery_stream, device));
          hidden = at::mm(transformed.to(at::kFloat), a_float).to(at::kHalf);
          C10_CUDA_CHECK(cudaEventRecord(recovery_done, recovery_stream));
        }
        C10_CUDA_CHECK(cudaStreamWaitEvent(static_cast<cudaStream_t>(cuda_stream), recovery_done, 0));
      } else {
        hidden = at::mm(transformed.to(at::kFloat), a_float).to(at::kHalf);
      }
      // Preserve the explicit FP16 hidden boundary while combining the FP32
      // rank expansion with the decoded base in one BLAS epilogue.  ``addmm``
      // retains FP32 inputs/accumulation for the output-Hadamard reference
      // path, whose transform still owns the final store.
      inner = at::addmm(inner, hidden.to(at::kFloat), b_float);
    }
    if (!wrote_output && c.output_hadamard) {
      inner = hadamard(inner.contiguous(), c10::IValue(), scale_v,
                      output_bias.defined() ? c10::IValue(output_bias) : c10::IValue(), c.n >= 2048 ? 3 : 4);
    } else if (!wrote_output) {
      inner = inner * scale_v;
      if (output_bias.defined()) inner = inner + output_bias;
    }
    if (!wrote_output) output.copy_(inner);
    if (error && error_capacity) error[0] = '\0';
    return 0;
  } catch (const std::exception& exception) {
    if (error && error_capacity) std::snprintf(error, error_capacity, "%s", exception.what());
    return 1;
  } catch (...) {
    if (error && error_capacity) std::snprintf(error, error_capacity, "unknown native window failure");
    return 2;
  }
}

extern "C" int qvq_p32_window_linear(
    QvqWindowBuffer x, QvqWindowBuffer window, QvqWindowBuffer banks,
    QvqWindowBuffer levels, QvqWindowBuffer su, QvqWindowBuffer sv,
    QvqWindowBuffer bias, QvqWindowBuffer rank8_a, QvqWindowBuffer rank8_b,
    QvqWindowBuffer y, const QvqP32WindowConfig* config, void* cuda_stream,
    char* error, uint64_t error_capacity) {
  return qvq_p32_window_linear_impl(
      x, window, banks, levels, su, sv, bias, rank8_a, rank8_b, y,
      config, cuda_stream, error, error_capacity, at::Tensor(), at::Tensor(),
      nullptr, nullptr, nullptr);
}

namespace {
struct PreparedWindow {
  at::cuda::CUDAGraph graph{true};
  cudaStream_t stream;
  int device;
  at::Tensor rank8_a_float;
  at::Tensor rank8_b_float;
  cudaStream_t recovery_stream{nullptr};
  cudaEvent_t recovery_ready{nullptr};
  cudaEvent_t recovery_done{nullptr};
  std::mutex mutex;

  ~PreparedWindow() {
    if (recovery_ready != nullptr) cudaEventDestroy(recovery_ready);
    if (recovery_done != nullptr) cudaEventDestroy(recovery_done);
    if (recovery_stream != nullptr) cudaStreamDestroy(recovery_stream);
  }
};
template <typename Function>
int graph_boundary(Function&& function, char* error, uint64_t capacity) {
  try {
    function();
    if (error && capacity) error[0] = '\0';
    return 0;
  } catch (const std::exception& exception) {
    if (error && capacity) std::snprintf(error, capacity, "%s", exception.what());
    return 1;
  } catch (...) {
    if (error && capacity) std::snprintf(error, capacity, "unknown native window graph failure");
    return 2;
  }
}
} // namespace

extern "C" int qvq_p32_window_graph_create(
    const QvqWindowBuffer* buffers, const QvqP32WindowConfig* config,
    void* cuda_stream, void** handle, char* error, uint64_t error_capacity) {
  return graph_boundary([&] {
    check(handle, "native graph handle output is null");
    *handle = nullptr;
    check(buffers && cuda_stream, "native graph requires buffers and a non-default stream");
    auto stream = static_cast<cudaStream_t>(cuda_stream);
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    check(capture == cudaStreamCaptureStatusNone, "prepare the native graph outside capture");
    auto invoke_linear = [&] {
      char message[4096]{};
      const int status = qvq_p32_window_linear(
          buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
          buffers[5], buffers[6], buffers[7], buffers[8], buffers[9],
          config, cuda_stream, message, sizeof(message));
      if (status) throw std::runtime_error(message);
    };
    // Validation and lazy library initialization finish before capture begins.
    for (int i = 0; i < 3; ++i) invoke_linear();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaPointerAttributes attributes{};
    C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, buffers[0].data));
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, attributes.device));
    auto prepared = std::make_unique<PreparedWindow>();
    prepared->stream = stream;
    prepared->device = attributes.device;
    if (config->rank8_enabled) {
      prepared->rank8_a_float = tensor(
          buffers[7], {config->k, 8}, at::kHalf, prepared->device).to(at::kFloat);
      prepared->rank8_b_float = tensor(
          buffers[8], {8, config->n}, at::kHalf, prepared->device).to(at::kFloat);
    }
    const bool has_recovery_policy =
        config->struct_bytes >= offsetof(QvqP32WindowConfig, recovery_kernel) +
            2 * sizeof(uint32_t);
    if (config->rank8_enabled && has_recovery_policy &&
        config->recovery_projection == 1) {
      // The auxiliary producer is owned by the prepared graph.  Creating it
      // here, before capture, keeps replay free of stream/event allocation.
      C10_CUDA_CHECK(cudaStreamCreateWithFlags(
          &prepared->recovery_stream, cudaStreamNonBlocking));
      C10_CUDA_CHECK(cudaEventCreateWithFlags(
          &prepared->recovery_ready, cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventCreateWithFlags(
          &prepared->recovery_done, cudaEventDisableTiming));
    }
    auto invoke_prepared = [&] {
      char message[4096]{};
      const int status = qvq_p32_window_linear_impl(
          buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
          buffers[5], buffers[6], buffers[7], buffers[8], buffers[9],
          config, cuda_stream, message, sizeof(message),
          prepared->rank8_a_float, prepared->rank8_b_float,
          prepared->recovery_stream, prepared->recovery_ready,
          prepared->recovery_done);
      if (status) throw std::runtime_error(message);
    };
    prepared->graph.capture_begin();
    try {
      owned_capture = true;
      invoke_prepared();
      owned_capture = false;
      prepared->graph.capture_end();
    } catch (...) {
      owned_capture = false;
      // End the CUDA stream capture as well as releasing allocator routing.
      // reset() alone must not leave the caller's stream inside capture.
      cudaStreamCaptureStatus failed_capture;
      if (cudaStreamIsCapturing(stream, &failed_capture) == cudaSuccess
          && failed_capture != cudaStreamCaptureStatusNone) {
        try { prepared->graph.capture_end(); } catch (...) {}
      }
      prepared->graph.reset();
      throw;
    }
    prepared->graph.instantiate();
    *handle = prepared.release();
  }, error, error_capacity);
}

extern "C" int qvq_p32_window_graph_run(
    void* handle, void* cuda_stream, char* error, uint64_t error_capacity) {
  return graph_boundary([&] {
    check(handle, "native graph handle is null");
    auto& prepared = *static_cast<PreparedWindow*>(handle);
    std::lock_guard<std::mutex> lock(prepared.mutex);
    auto stream = static_cast<cudaStream_t>(cuda_stream);
    check(stream == prepared.stream, "native graph must run on its owning stream");
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, prepared.device));
    cudaStreamCaptureStatus capture;
    cudaGraph_t parent = nullptr;
    const cudaGraphNode_t* dependencies = nullptr;
    const cudaGraphEdgeData* edge_data = nullptr;
    size_t count = 0;
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo(
        stream, &capture, nullptr, &parent, &dependencies, &edge_data, &count));
    check(capture != cudaStreamCaptureStatusInvalidated, "enclosing CUDA capture is invalidated");
    if (capture == cudaStreamCaptureStatusNone) {
      prepared.graph.replay();
    } else {
      cudaGraphNode_t child;
      cudaGraphNodeParams params{};
      params.type = cudaGraphNodeTypeGraph;
      params.graph.graph = prepared.graph.raw_cuda_graph();
      C10_CUDA_CHECK(cudaGraphAddNode(&child, parent, dependencies, edge_data, count, &params));
      C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
          stream, &child, nullptr, 1, cudaStreamSetCaptureDependencies));
    }
  }, error, error_capacity);
}

extern "C" int qvq_p32_window_graph_destroy(
    void* handle, char* error, uint64_t error_capacity) {
  return graph_boundary([&] {
    if (!handle) return;
    auto* prepared = static_cast<PreparedWindow*>(handle);
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(prepared->stream, prepared->device));
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(prepared->stream, &capture));
    check(capture == cudaStreamCaptureStatusNone, "cannot destroy native graph during capture");
    C10_CUDA_CHECK(cudaStreamSynchronize(prepared->stream));
    delete prepared;
  }, error, error_capacity);
}
TORCH_LIBRARY(gptqmodel_qvq_window_abi, m) {
  m.def("version() -> int", []() -> int64_t { return 3; });
}
