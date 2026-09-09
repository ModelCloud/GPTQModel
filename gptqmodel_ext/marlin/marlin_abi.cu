// SPDX-License-Identifier: Apache-2.0
#ifndef QVQ_MARLIN_STANDALONE
#define QVQ_MARLIN_STANDALONE
#endif
#include "gptq_marlin.cu"
#include "marlin_abi.h"
#include <cstdio>
#include <memory>
#include <limits>

struct QvqMarlinPlan {
  QvqMarlinProblem problem;
  QvqMarlinConfig config;
  QvqMarlinResources resources{};
  marlin::MarlinFuncPtr kernel;
};
namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class F> int boundary(F&& fn, char* error, uint64_t capacity) noexcept {
  try {
    fn();
    if (error && capacity) error[0] = 0;
    return 0;
  } catch (const std::exception& e) {
    if (error && capacity) std::snprintf(error, capacity, "%s", e.what());
    return 1;
  } catch (...) {
    if (error && capacity) std::snprintf(error, capacity, "unknown Marlin ABI error");
    return 2;
  }
}
void buffer(QvqMarlinBuffer b, uint64_t bytes) {
  check(b.bytes >= bytes && (!bytes || b.data), "missing or undersized Marlin buffer");
  check(!bytes || reinterpret_cast<uintptr_t>(b.data) % 16 == 0,
        "Marlin device buffers require 16-byte alignment");
}
template<class T> marlin::MarlinFuncPtr select(const QvqMarlinProblem& p,
                                             const QvqMarlinConfig& c) {
  if (c.prefill) return marlin::get_marlin_packed_prefill_kernel<T>(c.prefill);
  const bool act = p.has_act_order && !p.is_k_full;
  return marlin::get_marlin_kernel<T>(vllm::ScalarType::from_id(p.weight_type),
      c.tile_m == 8 ? 1 : c.tile_m / 16, c.tile_n / 16, c.tile_k / 16,
      c.tile_m == 8, act, p.has_zero_points,
      act ? 0 : (p.group_size == -1 ? -1 : p.group_size / 16),
      c.threads, p.zero_points_float, c.stages);
}
}
extern "C" int qvq_marlin_abi_version() { return QVQ_MARLIN_ABI_VERSION; }
extern "C" void qvq_marlin_destroy(QvqMarlinPlan* plan) { delete plan; }

extern "C" int qvq_marlin_prepare(const QvqMarlinProblem* problem,
    const QvqMarlinConfig* config, QvqMarlinPlan** output,
    QvqMarlinResources* resources, char* error, uint64_t capacity) {
  if (output) *output = nullptr;
  return boundary([&] {
    check(problem && config && output && resources, "null Marlin preparation argument");
    const auto& p = *problem;
    const auto& c = *config;
    check(p.struct_bytes == sizeof(p) && c.struct_bytes == sizeof(c) &&
          p.abi_version == QVQ_MARLIN_ABI_VERSION && c.abi_version == QVQ_MARLIN_ABI_VERSION,
          "Marlin ABI version/structure mismatch");
    check(p.m > 0 && p.k > 0 && p.n > 0 && p.e == 1 && p.lda >= p.k && p.lda % 8 == 0,
          "invalid Marlin M/K/N/E/stride; dense ABI requires E=1");
    check(p.dtype == 0 || p.dtype == 1, "Marlin requires FP16 or BF16");
    for (int flag : {p.has_bias, p.has_act_order, p.is_k_full, p.has_zero_points,
                     p.zero_points_float, c.atomic_add, c.fp32_reduce})
      check(flag == 0 || flag == 1, "Marlin flags must be 0 or 1");
    check(c.tile_m == 8 || c.tile_m == 16 || c.tile_m == 32 || c.tile_m == 48 || c.tile_m == 64,
          "unsupported Marlin M tile");
    check(c.tile_m != 8 || p.m <= 8, "M8 specialization requires M<=8");
    check(c.tile_n >= 64 && c.tile_n <= 512 && c.tile_n % 16 == 0 &&
          c.tile_k >= 64 && c.tile_k <= 128 && c.tile_k % 16 == 0 &&
          p.n % c.tile_n == 0 && p.k % c.tile_k == 0,
          "Marlin K/N must divide the requested tiles");
    check(c.threads == 64 || c.threads == 128 || c.threads == 256,
          "unsupported Marlin thread count");
    check(c.stages == 2 || c.stages == 4, "Marlin has only stage-2/stage-4 specializations");
    check(c.max_parallel >= 1 && c.max_parallel <= 128 && c.blocks_per_sm >= 1 &&
          c.blocks_per_sm <= 4, "invalid Marlin grid/reduction parallelism");
    int device;
    cuda_check(cudaGetDevice(&device));
    check(device == p.device, "set the caller CUDA device before Marlin preparation");
    cudaStreamCaptureStatus capture;
    cuda_check(cudaStreamIsCapturing(nullptr, &capture));
    check(capture == cudaStreamCaptureStatusNone, "prepare Marlin outside CUDA capture");
    const auto info = marlin::query_marlin_device_info(device);
    check(info.major_capability == 8 && info.minor_capability == 0,
          "this Marlin ABI build targets SM80");
    check(c.sm_count >= 1 && c.sm_count <= info.sms, "invalid requested Marlin SM count");
    check(c.shared_memory_bytes > 0 && c.shared_memory_bytes <= info.max_shared_mem,
          "invalid requested Marlin shared memory");
    check(p.groups > 0, "Marlin requires positive scale group count");
    if (p.has_act_order && !p.is_k_full)
      check(p.group_size == 0, "partial act-order requires group_size=0");
    else
      check((p.group_size == -1 && p.groups == 1) ||
            (p.group_size >= 16 && p.group_size % 16 == 0 &&
             p.k % p.group_size == 0 && p.groups == p.k / p.group_size),
            "scale groups do not match Marlin K");
    const auto type = vllm::ScalarType::from_id(p.weight_type);
    check(type == vllm::kU4 || type == vllm::kU4B8 || type == vllm::kU8B128,
          "initial Marlin ABI supports GPTQ/AWQ integer weights only");
    check((type == vllm::kU4) == bool(p.has_zero_points), "weight type/zero-point mismatch");
    check(!p.zero_points_float || (p.has_zero_points && p.dtype == 0),
          "floating zero points require FP16 and explicit zero points");
    check(!p.has_act_order || !p.has_zero_points, "act-order with zero points is unsupported");
    check(c.prefill >= 0 && c.prefill <= 4, "invalid packed-prefill specialization");
    if (c.prefill) {
      static const int tiles[4][4] = {{64,128,64,128}, {64,256,64,128},
                                    {32,512,64,256}, {64,128,64,64}};
      const auto& tile = tiles[c.prefill - 1];
      check(type == vllm::kU4B8 && !p.has_act_order && p.is_k_full && p.group_size == 128 &&
            p.m >= 16 && c.tile_m == tile[0] && c.tile_n == tile[1] &&
            c.tile_k == tile[2] && c.threads == tile[3] && c.stages == 4 && !c.atomic_add &&
            c.blocks_per_sm == 1 && c.max_parallel == 1 && c.sm_count == info.sms,
            "packed-prefill geometry/quantization differs from compiled specialization");
    } else {
      check(marlin::is_valid_config({c.tile_k,c.tile_n,c.threads},
            c.tile_m == 8 ? 1 : c.tile_m/16, p.m,p.n,p.k,type.size_bits(),p.group_size,
            p.has_act_order && !p.is_k_full,p.is_k_full,p.has_zero_points,p.zero_points_float,
            c.stages,c.shared_memory_bytes), "invalid Marlin specialization resource contract");
    }
    check(marlin::get_kernel_cache_size({c.tile_k,c.tile_n,c.threads},
          c.tile_m == 8 ? 1 : c.tile_m/16,p.m,p.n,p.k,type.size_bits(),p.group_size,
          p.has_act_order && !p.is_k_full,p.is_k_full,p.has_zero_points,p.zero_points_float,
          c.stages) <= c.shared_memory_bytes, "Marlin shared memory is smaller than the kernel requires");
    check(!c.prefill || ((uint64_t(p.m)+c.tile_m-1)/c.tile_m)*(p.n/c.tile_n) <= INT32_MAX,
          "Marlin prefill grid exceeds CUDA grid limit");
    auto plan = std::make_unique<QvqMarlinPlan>();
    plan->problem = p;
    plan->config = c;
    plan->kernel = p.dtype == 0 ? select<half>(p,c) : select<nv_bfloat16>(p,c);
    check(plan->kernel != marlin::MarlinDefault, "requested Marlin specialization was not compiled");
    marlin::ensure_marlin_max_dynamic_shared_memory(plan->kernel, device, c.shared_memory_bytes);
    const uint64_t columns = std::max<uint64_t>(p.n, uint64_t(c.sm_count)*c.blocks_per_sm*512);
    plan->resources = {c.fp32_reduce ? uint64_t(64)*columns*4 : 0,
      p.has_act_order ? uint64_t(p.m)*p.k*2 : 0,
      uint64_t(c.sm_count)*c.blocks_per_sm*4,
      c.prefill ? 1 : int((uint64_t(p.m)+uint64_t(c.tile_m)*c.max_parallel-1)/(uint64_t(c.tile_m)*c.max_parallel)),
      80, QVQ_MARLIN_KERNEL_VERSION};
    if (p.has_act_order) ++plan->resources.launch_count;
    ++plan->resources.launch_count; // scratch lock initialization
    *resources = plan->resources;
    *output = plan.release();
  }, error, capacity);
}

extern "C" int qvq_marlin_launch(const QvqMarlinPlan* plan,
    const QvqMarlinBuffers* buffers, void* raw_stream, char* error, uint64_t capacity) {
  return boundary([&] {
    check(plan && buffers, "null Marlin execution argument");
    const auto& p = plan->problem;
    const auto& c = plan->config;
    const auto& b = *buffers;
    int device;
    cuda_check(cudaGetDevice(&device));
    check(device == p.device, "Marlin plan belongs to a different CUDA device");
    const auto type = vllm::ScalarType::from_id(p.weight_type);
    buffer(b.a, (uint64_t(p.m-1)*p.lda+p.k)*2);
    buffer(b.weight, uint64_t(p.k)*p.n*type.size_bits()/8);
    buffer(b.scales, uint64_t(p.groups)*p.n*2);
    buffer(b.output, uint64_t(p.m)*p.n*2);
    buffer(b.bias, p.has_bias ? uint64_t(p.n)*2 : 0);
    buffer(b.zeros, p.has_zero_points ? uint64_t(p.groups)*p.n*(p.zero_points_float ? 16 : type.size_bits())/8 : 0);
    buffer(b.group_index, p.has_act_order ? uint64_t(p.k)*4 : 0);
    buffer(b.permutation, p.has_act_order ? uint64_t(p.k)*4 : 0);
    buffer(b.reduction, plan->resources.reduction_bytes);
    buffer(b.permuted_a, plan->resources.permuted_a_bytes);
    buffer(b.locks, plan->resources.locks_bytes);
    auto stream = static_cast<cudaStream_t>(raw_stream);
    cuda_check(cudaMemsetAsync(b.locks.data, 0, plan->resources.locks_bytes, stream));
    const int4* a = static_cast<const int4*>(b.a.data);
    int lda = p.lda;
    if (p.has_act_order) {
      int blocks = std::min(c.sm_count,p.m);
      marlin::permute_cols_kernel<<<blocks,marlin::default_threads,0,stream>>>(
        a, static_cast<const int*>(b.permutation.data),static_cast<int4*>(b.permuted_a.data),
        p.m,p.k,lda,marlin::div_ceil(p.m,blocks));
      a = static_cast<const int4*>(b.permuted_a.data);
      lda = p.k;
    }
    int4* out = static_cast<int4*>(b.output.data);
    for (int offset = 0; offset < p.m;) {
      int rows = c.prefill ? p.m : std::min(p.m-offset,c.tile_m*c.max_parallel);
      int grid = c.prefill ? marlin::div_ceil(rows,c.tile_m)*(p.n/c.tile_n) : c.sm_count*c.blocks_per_sm;
      const int4* weight = static_cast<const int4*>(b.weight.data);
      int4* reduction = static_cast<int4*>(b.reduction.data);
      const int4* bias = p.has_bias ? static_cast<const int4*>(b.bias.data) : nullptr;
      const int4* scales = static_cast<const int4*>(b.scales.data);
      const float* global = nullptr;
      const int4* zeros = p.has_zero_points ? static_cast<const int4*>(b.zeros.data) : nullptr;
      const int* groups = p.has_act_order ? static_cast<const int*>(b.group_index.data) : nullptr;
      int* locks = static_cast<int*>(b.locks.data);
      bool bias_flag = p.has_bias, atomic = c.atomic_add, fp32 = c.fp32_reduce;
      void* args[] = {&a,&weight,&out,&reduction,&bias,&scales,&global,&zeros,&groups,
        const_cast<int*>(&p.groups),&rows,const_cast<int*>(&p.n),const_cast<int*>(&p.k),&lda,&locks,
        &bias_flag,&atomic,&fp32,const_cast<int*>(&c.shared_memory_bytes)};
      cuda_check(cudaLaunchKernel(reinterpret_cast<const void*>(plan->kernel),dim3(grid),dim3(c.threads),
        args,c.shared_memory_bytes,stream));
      offset += rows;
      a += uint64_t(rows)*lda/8;
      out += uint64_t(rows)*p.n/8;
    }
  }, error, capacity);
}
