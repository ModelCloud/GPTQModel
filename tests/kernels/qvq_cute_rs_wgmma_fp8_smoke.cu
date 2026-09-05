// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>

namespace {

using Element = cutlass::float_e4m3_t;
using TileShape = cute::Shape<cute::_64, cute::_16, cute::_32>;
using TiledMma = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<
        Element,
        Element,
        float,
        TileShape,
        cute::GMMA::Major::K,
        cute::GMMA::Major::K>()));
using SmemLayoutAtomB = decltype(
    cutlass::gemm::collective::detail::rs_smem_selector<
        cute::GMMA::Major::K,
        Element,
        cute::_32,
        cute::_32,
        false>());
using SmemLayoutB = decltype(cute::tile_to_shape(
    SmemLayoutAtomB{}, cute::make_shape(cute::_16{}, cute::_32{})));

static_assert(cute::size(TiledMma{}) == 128);

__global__ __launch_bounds__(128) void rs_wgmma_fp8_smoke_kernel(
    const Element* __restrict__ a,
    const Element* __restrict__ b,
    float* __restrict__ c,
    int* __restrict__ values_per_thread) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  __shared__ __align__(128) Element shared_b[cute::cosize_v<SmemLayoutB>];
  auto sB = cute::make_tensor(cute::make_smem_ptr(shared_b), SmemLayoutB{});
  for (int index = static_cast<int>(threadIdx.x); index < 16 * 32; index += 128) {
    const int n = index / 32;
    const int k = index - n * 32;
    sB(n, k) = b[index];
  }
  __syncthreads();

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(static_cast<int>(threadIdx.x));
  auto coordinate_a = cute::make_identity_tensor(
      cute::make_shape(cute::_64{}, cute::_32{}));
  auto thread_coordinate_a = thread_mma.partition_A(coordinate_a);
  auto fragment_a = cute::make_tensor<Element>(thread_coordinate_a.shape());
  constexpr int kAValuesPerThread = cute::size(decltype(thread_coordinate_a){});
  static_assert(kAValuesPerThread == 16);
  if (threadIdx.x == 0) {
    *values_per_thread = kAValuesPerThread;
  }
#pragma unroll
  for (int index = 0; index < kAValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_a(index);
    const int row = static_cast<int>(cute::get<0>(coordinate));
    const int column = static_cast<int>(cute::get<1>(coordinate));
    fragment_a(index) = a[row * 32 + column];
  }

  auto thread_shared_b = thread_mma.partition_B(sB);
  auto fragment_b = thread_mma.make_fragment_B(thread_shared_b);
  auto coordinate_c = cute::make_identity_tensor(
      cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_c = thread_mma.partition_C(coordinate_c);
  auto accumulator = cute::make_tensor<float>(thread_coordinate_c.shape());
  cute::clear(accumulator);

  cute::warpgroup_fence_operand(fragment_a);
  cute::warpgroup_fence_operand(accumulator);
  cute::warpgroup_arrive();
  cute::gemm(
      tiled_mma,
      fragment_a(cute::_, cute::_, cute::_0{}),
      fragment_b(cute::_, cute::_, cute::_0{}),
      accumulator);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
  cute::warpgroup_fence_operand(accumulator);

  constexpr int kCValuesPerThread = cute::size(decltype(thread_coordinate_c){});
#pragma unroll
  for (int index = 0; index < kCValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_c(index);
    const int row = static_cast<int>(cute::get<0>(coordinate));
    const int column = static_cast<int>(cute::get<1>(coordinate));
    c[row * 16 + column] = accumulator(index);
  }
#endif
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

}  // namespace

int main() {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  if (properties.major != 9 || properties.minor != 0 ||
      std::string(properties.name).find("H200") == std::string::npos) {
    std::fprintf(stderr, "exclusive H200 required, found %s\n", properties.name);
    return 2;
  }

  std::vector<Element> host_a(64 * 32, Element(1.0f));
  std::vector<Element> host_b(16 * 32, Element(1.0f));
  std::vector<float> host_c(64 * 16, 0.0f);
  Element* device_a = nullptr;
  Element* device_b = nullptr;
  float* device_c = nullptr;
  int* device_values_per_thread = nullptr;
  check_cuda(cudaMalloc(&device_a, sizeof(Element) * host_a.size()), "cudaMalloc A");
  check_cuda(cudaMalloc(&device_b, sizeof(Element) * host_b.size()), "cudaMalloc B");
  check_cuda(cudaMalloc(&device_c, sizeof(float) * host_c.size()), "cudaMalloc C");
  check_cuda(cudaMalloc(&device_values_per_thread, sizeof(int)), "cudaMalloc metadata");
  check_cuda(cudaMemcpy(device_a, host_a.data(), sizeof(Element) * host_a.size(),
                        cudaMemcpyHostToDevice), "cudaMemcpy A");
  check_cuda(cudaMemcpy(device_b, host_b.data(), sizeof(Element) * host_b.size(),
                        cudaMemcpyHostToDevice), "cudaMemcpy B");

  rs_wgmma_fp8_smoke_kernel<<<1, 128>>>(
      device_a, device_b, device_c, device_values_per_thread);
  check_cuda(cudaGetLastError(), "rs_wgmma_fp8_smoke_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "rs_wgmma_fp8_smoke_kernel synchronize");
  check_cuda(cudaMemcpy(host_c.data(), device_c, sizeof(float) * host_c.size(),
                        cudaMemcpyDeviceToHost), "cudaMemcpy C");
  int values_per_thread = 0;
  check_cuda(cudaMemcpy(&values_per_thread, device_values_per_thread, sizeof(int),
                        cudaMemcpyDeviceToHost), "cudaMemcpy metadata");

  for (size_t index = 0; index < host_c.size(); ++index) {
    if (std::abs(host_c[index] - 32.0f) > 1.0e-4f) {
      std::fprintf(stderr, "incorrect C[%zu]: %.8f (expected 32)\n",
                   index, host_c[index]);
      return 3;
    }
  }
  std::printf(
      "PASS device=%s mma=m64n16k32 operands=e4m3xe4m3 accumulator=f32 "
      "A_values_per_lane=%d\n",
      properties.name,
      values_per_thread);
  return values_per_thread == 16 ? 0 : 4;
}
