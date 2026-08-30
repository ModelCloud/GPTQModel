// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdio>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>
#include <cutlass/version.h>

namespace {

using Element = cutlass::half_t;
using TileShape = cute::Shape<cute::_64, cute::_16, cute::_16>;
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
        cute::_16,
        cute::_16,
        false>());
using SmemLayoutB = decltype(cute::tile_to_shape(
    SmemLayoutAtomB{}, cute::make_shape(cute::_16{}, cute::_16{})));

static_assert(cute::size(TiledMma{}) == 128);

__global__ __launch_bounds__(128) void rs_wgmma_smoke_kernel(
    const Element* __restrict__ a,
    const Element* __restrict__ b,
    float* __restrict__ c,
    int* __restrict__ a_rows,
    int* __restrict__ a_columns,
    int* __restrict__ a_values_per_thread) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  __shared__ __align__(128) Element shared_b[cute::cosize_v<SmemLayoutB>];
  auto sB = cute::make_tensor(cute::make_smem_ptr(shared_b), SmemLayoutB{});

  for (int index = static_cast<int>(threadIdx.x); index < 16 * 16; index += 128) {
    const int n = index / 16;
    const int k = index - n * 16;
    sB(n, k) = b[index];
  }
  __syncthreads();

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(static_cast<int>(threadIdx.x));

  auto coordinate_a = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_a = thread_mma.partition_A(coordinate_a);
  auto fragment_a = cute::make_tensor<Element>(thread_coordinate_a.shape());

  constexpr int kAValuesPerThread = cute::size(decltype(thread_coordinate_a){});
  static_assert(kAValuesPerThread == 8);
  if (threadIdx.x == 0) {
    *a_values_per_thread = kAValuesPerThread;
  }
#pragma unroll
  for (int index = 0; index < kAValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_a(index);
    const int row = static_cast<int>(cute::get<0>(coordinate));
    const int column = static_cast<int>(cute::get<1>(coordinate));
    fragment_a(index) = a[row * 16 + column];
    a_rows[static_cast<int>(threadIdx.x) * kAValuesPerThread + index] = row;
    a_columns[static_cast<int>(threadIdx.x) * kAValuesPerThread + index] = column;
  }

  auto thread_shared_b = thread_mma.partition_B(sB);
  auto fragment_b = thread_mma.make_fragment_B(thread_shared_b);

  auto coordinate_c = cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_16{}));
  auto thread_coordinate_c = thread_mma.partition_C(coordinate_c);
  auto fragment_c = cute::make_tensor<float>(thread_coordinate_c.shape());
  cute::clear(fragment_c);

  cute::warpgroup_fence_operand(fragment_a);
  cute::warpgroup_fence_operand(fragment_c);
  cute::warpgroup_arrive();
  cute::gemm(
      tiled_mma,
      fragment_a(cute::_, cute::_, cute::_0{}),
      fragment_b(cute::_, cute::_, cute::_0{}),
      fragment_c);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
  cute::warpgroup_fence_operand(fragment_c);

  constexpr int kCValuesPerThread = cute::size(decltype(thread_coordinate_c){});
#pragma unroll
  for (int index = 0; index < kCValuesPerThread; ++index) {
    const auto coordinate = thread_coordinate_c(index);
    const int row = static_cast<int>(cute::get<0>(coordinate));
    const int column = static_cast<int>(cute::get<1>(coordinate));
    c[row * 16 + column] = fragment_c(index);
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

int main(int argc, char** argv) {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  if (properties.major != 9 || properties.minor != 0) {
    std::fprintf(stderr, "SM90 H100/H200 required, found %s (SM%d%d)\n",
                 properties.name, properties.major, properties.minor);
    return 2;
  }

  constexpr int kAValues = 64 * 16;
  constexpr int kBValues = 16 * 16;
  constexpr int kCValues = 64 * 16;
  constexpr int kAFragmentCoordinates = 128 * 8;

  std::vector<Element> host_a(kAValues, Element(1.0f));
  std::vector<Element> host_b(kBValues, Element(1.0f));
  std::vector<float> host_c(kCValues, 0.0f);
  std::vector<int> host_rows(kAFragmentCoordinates, -1);
  std::vector<int> host_columns(kAFragmentCoordinates, -1);

  Element* device_a = nullptr;
  Element* device_b = nullptr;
  float* device_c = nullptr;
  int* device_rows = nullptr;
  int* device_columns = nullptr;
  int* device_values_per_thread = nullptr;
  check_cuda(cudaMalloc(&device_a, sizeof(Element) * kAValues), "cudaMalloc A");
  check_cuda(cudaMalloc(&device_b, sizeof(Element) * kBValues), "cudaMalloc B");
  check_cuda(cudaMalloc(&device_c, sizeof(float) * kCValues), "cudaMalloc C");
  check_cuda(cudaMalloc(&device_rows, sizeof(int) * kAFragmentCoordinates), "cudaMalloc rows");
  check_cuda(cudaMalloc(&device_columns, sizeof(int) * kAFragmentCoordinates), "cudaMalloc columns");
  check_cuda(cudaMalloc(&device_values_per_thread, sizeof(int)), "cudaMalloc fragment size");
  check_cuda(cudaMemcpy(device_a, host_a.data(), sizeof(Element) * kAValues, cudaMemcpyHostToDevice),
             "cudaMemcpy A");
  check_cuda(cudaMemcpy(device_b, host_b.data(), sizeof(Element) * kBValues, cudaMemcpyHostToDevice),
             "cudaMemcpy B");

  rs_wgmma_smoke_kernel<<<1, 128>>>(
      device_a,
      device_b,
      device_c,
      device_rows,
      device_columns,
      device_values_per_thread);
  check_cuda(cudaGetLastError(), "rs_wgmma_smoke_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "rs_wgmma_smoke_kernel synchronize");
  check_cuda(cudaMemcpy(host_c.data(), device_c, sizeof(float) * kCValues, cudaMemcpyDeviceToHost),
             "cudaMemcpy C");
  check_cuda(cudaMemcpy(host_rows.data(), device_rows,
                        sizeof(int) * kAFragmentCoordinates, cudaMemcpyDeviceToHost),
             "cudaMemcpy rows");
  check_cuda(cudaMemcpy(host_columns.data(), device_columns,
                        sizeof(int) * kAFragmentCoordinates, cudaMemcpyDeviceToHost),
             "cudaMemcpy columns");
  int values_per_thread = 0;
  check_cuda(cudaMemcpy(&values_per_thread, device_values_per_thread,
                        sizeof(int), cudaMemcpyDeviceToHost),
             "cudaMemcpy fragment size");

  if (argc == 2 && std::string_view(argv[1]) == "--dump-layout") {
    for (int thread = 0; thread < 128; ++thread) {
      std::printf("thread=%03d", thread);
      for (int value = 0; value < values_per_thread; ++value) {
        const int index = thread * values_per_thread + value;
        std::printf(" (%d,%d)", host_rows[index], host_columns[index]);
      }
      std::printf("\n");
    }
  }

  for (int index = 0; index < kCValues; ++index) {
    if (std::abs(host_c[index] - 16.0f) > 1.0e-4f) {
      std::fprintf(stderr, "incorrect C[%d]: %.8f (expected 16)\n", index, host_c[index]);
      return 3;
    }
  }

  int row_lane_counts[64] = {};
  bool coordinate_seen[kAValues] = {};
  for (int thread = 0; thread < 128; ++thread) {
    int owned_rows[2] = {-1, -1};
    int values_per_row[2] = {};
    int owned_row_count = 0;
    for (int value = 0; value < values_per_thread; ++value) {
      const int index = thread * values_per_thread + value;
      const int row = host_rows[index];
      const int column = host_columns[index];
      if (row < 0 || row >= 64 || column < 0 || column >= 16) {
        std::fprintf(stderr, "invalid A coordinate from thread %d: (%d,%d)\n", thread, row, column);
        return 4;
      }
      int row_slot = -1;
      for (int slot = 0; slot < owned_row_count; ++slot) {
        if (owned_rows[slot] == row) {
          row_slot = slot;
        }
      }
      if (row_slot < 0) {
        if (owned_row_count >= 2) {
          std::fprintf(stderr, "thread %d owns more than two A rows\n", thread);
          return 5;
        }
        row_slot = owned_row_count++;
        owned_rows[row_slot] = row;
        ++row_lane_counts[row];
      }
      ++values_per_row[row_slot];
      const int coordinate_index = row * 16 + column;
      if (coordinate_seen[coordinate_index]) {
        std::fprintf(stderr, "duplicate A coordinate (%d,%d)\n", row, column);
        return 6;
      }
      coordinate_seen[coordinate_index] = true;
    }
    if (owned_row_count != 2 || values_per_row[0] != 4 || values_per_row[1] != 4) {
      std::fprintf(stderr,
                   "thread %d owns %d A rows with %d/%d values (expected 2 rows, 4 each)\n",
                   thread, owned_row_count, values_per_row[0], values_per_row[1]);
      return 7;
    }
  }
  for (int row = 0; row < 64; ++row) {
    if (row_lane_counts[row] != 4) {
      std::fprintf(stderr, "A row %d is owned by %d lanes (expected 4)\n", row, row_lane_counts[row]);
      return 8;
    }
  }

  std::printf(
      "PASS device=%s cutlass=%d.%d.%d mma=m64n16k16 "
      "A_values_per_lane=%d A_rows_per_lane=2 A_values_per_lane_row=4 lanes_per_A_row=4\n",
      properties.name,
      CUTLASS_MAJOR,
      CUTLASS_MINOR,
      CUTLASS_PATCH,
      values_per_thread);
  return 0;
}
