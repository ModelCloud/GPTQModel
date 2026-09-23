// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#define QVQ_WGMMA_DEVICE_ONLY 1
#include "qvq_wgmma_cuda.cu"
#include "qvq_wgmma_raw_abi.h"

#include <cstdio>
#include <cstring>

// A transition-rate shard compiles only one template family and publishes a
// private entry point. The small dispatcher TU retains the stable public ABI.
#if QVQ_WGMMA_BITS_ONLY == 4
#define qvq_p32_wgmma_raw_launch qvq_p32_wgmma_raw_launch_w4
#define qvq_p32_wgmma_raw_launch_plan qvq_p32_wgmma_raw_launch_plan_w4
#elif QVQ_WGMMA_BITS_ONLY == 5
#define qvq_p32_wgmma_raw_launch qvq_p32_wgmma_raw_launch_w5
#define qvq_p32_wgmma_raw_launch_plan qvq_p32_wgmma_raw_launch_plan_w5
#elif QVQ_WGMMA_BITS_ONLY == 6
#define qvq_p32_wgmma_raw_launch qvq_p32_wgmma_raw_launch_w6
#define qvq_p32_wgmma_raw_launch_plan qvq_p32_wgmma_raw_launch_plan_w6
#elif QVQ_WGMMA_BITS_ONLY == 7
#define qvq_p32_wgmma_raw_launch qvq_p32_wgmma_raw_launch_w7
#define qvq_p32_wgmma_raw_launch_plan qvq_p32_wgmma_raw_launch_plan_w7
#endif

namespace {

constexpr uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

__global__ void pad_m16(
    const Element* input, Element* padded, int m, int k) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int count = 16 * k;
  if (index >= count) return;
  padded[index] = index < m * k ? input[index] : Element{};
}

template <int TransitionBits>
cudaError_t launch_ordered(
    const Element* input, const uint32_t* trellis, const uint8_t* bank_ids,
    const Element* levels, const uint8_t* bank_alt_id, float* workspace,
    int m, int k, int n, int split_count, cudaStream_t stream) {
  constexpr int kWords = 4 * TransitionBits;
  using TrellisLayout = P32TrellisTmaSmemLayoutFor<TransitionBits>;
  const int k_tiles = k / kP32TileRows;
  const int n_tiles = n / kP32TileColumns;
  auto input_tensor = cute::make_tensor(
      input, cute::make_shape(m, k),
      cute::make_stride(static_cast<int64_t>(k), cute::_1{}));
  auto trellis_tensor = cute::make_tensor(
      trellis, cute::make_shape(kWords, n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, cute::Int<kWords>{},
                        static_cast<int64_t>(n_tiles) * kWords));
  auto bank_tensor = cute::make_tensor(
      bank_ids, cute::make_shape(n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, static_cast<int64_t>(n_tiles)));
  auto input_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, input_tensor,
      WgmmaTmaSmemLayoutB{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_256{}));
  auto trellis_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, trellis_tensor,
      TrellisLayout{}(cute::_, cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::Int<kWords>{},
                       cute::Int<kP32N16TilesPerBlock>{},
                       cute::Int<kP32K16TilesPerStage>{}));
  auto bank_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, bank_tensor,
      P32BankTmaSmemLayout{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_16{}));
  const dim3 grid(static_cast<unsigned>(n / kOutputColumns), 1,
                  static_cast<unsigned>(split_count));
  HopperGroupedP32LaunchParams grouped{};
  grouped.launch_bank_alt_ids = bank_alt_id;
  if (k == 8192 && n == 2048 && split_count == 16) {
    qvq_p32_window_wgmma_m16_tma_kernel<
        TransitionBits, false, true, false, true>
        <<<grid, kTmaThreads, 0, stream>>>(
            input_tma, trellis_tma, bank_tma, levels, workspace, grouped,
            m, k, n, split_count, 0);
  } else {
    qvq_p32_window_wgmma_m16_tma_kernel<TransitionBits, false, true>
        <<<grid, kTmaThreads, 0, stream>>>(
            input_tma, trellis_tma, bank_tma, levels, workspace, grouped,
            m, k, n, split_count, 0);
  }
  return cudaGetLastError();
}

template <int TransitionBits, int Rows, int RowTiles>
cudaError_t launch_direct_rows(
    const Element* input, const uint32_t* trellis, const uint8_t* bank_ids,
    const Element* levels, const uint8_t* bank_alt_id, float* output,
    int k, int n, cudaStream_t stream) {
  constexpr int kWords = 4 * TransitionBits;
  using TrellisLayout = P32TrellisTmaSmemLayoutFor<TransitionBits>;
  using SharedStorage =
      P32WgmmaTmaSharedStorageFor<TransitionBits, 1, RowTiles>;
  const int k_tiles = k / kP32TileRows;
  const int n_tiles = n / kP32TileColumns;
  auto input_tensor = cute::make_tensor(
      input, cute::make_shape(Rows, k),
      cute::make_stride(static_cast<int64_t>(k), cute::_1{}));
  auto trellis_tensor = cute::make_tensor(
      trellis, cute::make_shape(kWords, n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, cute::Int<kWords>{},
                        static_cast<int64_t>(n_tiles) * kWords));
  auto bank_tensor = cute::make_tensor(
      bank_ids, cute::make_shape(n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, static_cast<int64_t>(n_tiles)));
  auto input_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, input_tensor,
      WgmmaTmaSmemLayoutB{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_256{}));
  auto trellis_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, trellis_tensor,
      TrellisLayout{}(cute::_, cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::Int<kWords>{},
                       cute::Int<kP32N16TilesPerBlock>{},
                       cute::Int<kP32K16TilesPerStage>{}));
  auto bank_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, bank_tensor,
      P32BankTmaSmemLayout{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_16{}));
  HopperGroupedP32LaunchParams grouped{};
  grouped.launch_bank_alt_ids = bank_alt_id;
  auto kernel = qvq_p32_window_wgmma_m16_tma_kernel<
      TransitionBits, false, false, false, true, 1, 0, RowTiles,
      decltype(input_tma), decltype(trellis_tma), decltype(bank_tma),
      HopperGroupedP32LaunchParams>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(SharedStorage)));
  if (status != cudaSuccess) return status;
  const dim3 grid(
      static_cast<unsigned>(n / kOutputColumns),
      static_cast<unsigned>(Rows / (RowTiles * kRows)), 1);
  kernel<<<grid, kTmaThreads, sizeof(SharedStorage), stream>>>(
      input_tma, trellis_tma, bank_tma, levels, output, grouped,
      Rows, k, n, 1, 0);
  return cudaGetLastError();
}

template <int TransitionBits>
cudaError_t launch_direct(
    const Element* input, const uint32_t* trellis, const uint8_t* bank_ids,
    const Element* levels, const uint8_t* bank_alt_id, float* output,
    int rows, int k, int n, int block_m, cudaStream_t stream) {
  return rows == 64
      ? launch_direct_rows<TransitionBits, 64, 4>(
            input, trellis, bank_ids, levels, bank_alt_id, output, k, n, stream)
      : n <= 512
          ? launch_direct_rows<TransitionBits, 128, 1>(
                input, trellis, bank_ids, levels, bank_alt_id, output, k, n, stream)
          : n <= 2048
              ? k == 8192
                  ? block_m == 64
                      ? launch_direct_rows<TransitionBits, 128, 4>(
                            input, trellis, bank_ids, levels, bank_alt_id, output, k, n, stream)
                      : launch_direct_rows<TransitionBits, 128, 8>(
                            input, trellis, bank_ids, levels, bank_alt_id, output, k, n, stream)
                  : launch_direct_rows<TransitionBits, 128, 2>(
                        input, trellis, bank_ids, levels, bank_alt_id, output, k, n, stream)
              : launch_direct_rows<TransitionBits, 128, 8>(
                    input, trellis, bank_ids, levels, bank_alt_id, output, k, n, stream);
}

struct RawGroupedGateUpParams {
  const uint8_t* bank_alt_id;
};

template <int TransitionBits>
cudaError_t launch_direct_grouped_gate_up(
    const Element* input, const uint32_t* trellis, const uint8_t* bank_ids,
    const Element* levels, const uint8_t* bank_alt_ids, float* output,
    int k, int n, cudaStream_t stream) {
  constexpr int Rows = 128;
  constexpr int RowTiles = 8;
  constexpr int N64BlocksPerCta = 2;
  constexpr int kWords = 4 * TransitionBits;
  using TrellisLayout =
      P32TrellisTmaSmemLayoutFor<TransitionBits, N64BlocksPerCta>;
  using SharedStorage = P32WgmmaTmaSharedStorageFor<
      TransitionBits, N64BlocksPerCta, RowTiles>;
  const int k_tiles = k / kP32TileRows;
  const int n_tiles = n / kP32TileColumns;
  const int child_n = n / 2;
  auto input_tensor = cute::make_tensor(
      input, cute::make_shape(Rows, k),
      cute::make_stride(static_cast<int64_t>(k), cute::_1{}));
  auto trellis_tensor = cute::make_tensor(
      trellis, cute::make_shape(kWords, n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, cute::Int<kWords>{},
                        static_cast<int64_t>(n_tiles) * kWords));
  auto bank_tensor = cute::make_tensor(
      bank_ids, cute::make_shape(n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, static_cast<int64_t>(n_tiles)));
  auto input_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, input_tensor,
      WgmmaTmaSmemLayoutB{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_256{}));
  auto trellis_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, trellis_tensor,
      TrellisLayout{}(cute::_, cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::Int<kWords>{},
                       cute::Int<2 * kP32N16TilesPerBlock>{},
                       cute::Int<kP32K16TilesPerStage>{}));
  auto bank_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, bank_tensor,
      P32BankTmaSmemLayout{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_16{}));
  const RawGroupedGateUpParams grouped{bank_alt_ids};
  auto kernel = qvq_p32_window_wgmma_m16_tma_kernel<
      TransitionBits, true, false, true, true, N64BlocksPerCta, false,
      RowTiles, decltype(input_tma), decltype(trellis_tma),
      decltype(bank_tma), RawGroupedGateUpParams>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(SharedStorage)));
  if (status != cudaSuccess) return status;
  const dim3 grid(static_cast<unsigned>(child_n / (N64BlocksPerCta * kOutputColumns)), 2, 1);
  kernel<<<grid, kTmaThreads + kThreads, sizeof(SharedStorage), stream>>>(
      input_tma, trellis_tma, bank_tma, levels, output, grouped,
      Rows, k, n, 1, 0);
  return cudaGetLastError();
}

__global__ void reduce_ordered_rows(
    const float* partials, float* output, int values, int plane_stride,
    int split_count) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= values) return;
  float value = 0.0f;
  for (int split = 0; split < split_count; ++split) {
    value += partials[static_cast<int64_t>(split) * plane_stride + index];
  }
  output[index] = value;
}

int fail(char* error, uint64_t capacity, const char* message) {
  if (error != nullptr && capacity != 0) {
    std::snprintf(error, static_cast<size_t>(capacity), "%s", message);
  }
  return 1;
}

struct PlanStorageWriter {
  uint8_t* data;
  uint64_t capacity;
  uint64_t offset = 0;

  template <typename T>
  T* store(const T& value) {
    const uint64_t aligned = align_up(offset, alignof(T));
    if (aligned + sizeof(T) > capacity) return nullptr;
    auto* destination = reinterpret_cast<T*>(data + aligned);
    std::memcpy(destination, &value, sizeof(T));
    offset = aligned + sizeof(T);
    return destination;
  }
};

void set_device_arg(
    qvq_p32_launch_descriptor* launch, int index, const void* address) {
  launch->args[index].address = address;
  launch->args[index].size = sizeof(void*);
  launch->args[index].type = QVQ_P32_LAUNCH_ARG_DEVICE_POINTER;
}

template <typename T>
bool set_host_arg(
    qvq_p32_launch_descriptor* launch, int index, PlanStorageWriter* storage,
    const T& value) {
  const T* stored = storage->store(value);
  if (stored == nullptr) return false;
  launch->args[index].address = stored;
  launch->args[index].size = sizeof(T);
  launch->args[index].type = QVQ_P32_LAUNCH_ARG_HOST_VALUE;
  return true;
}

template <int TransitionBits, int Rows, int RowTiles>
cudaError_t build_direct_plan_rows(
    const Element* input, const uint32_t* trellis, const uint8_t* bank_ids,
    const Element* levels, const uint8_t* bank_alt_id, float* output,
    int k, int n, qvq_p32_launch_plan* plan) {
  constexpr int kWords = 4 * TransitionBits;
  using TrellisLayout = P32TrellisTmaSmemLayoutFor<TransitionBits>;
  using SharedStorage =
      P32WgmmaTmaSharedStorageFor<TransitionBits, 1, RowTiles>;
  const int k_tiles = k / kP32TileRows;
  const int n_tiles = n / kP32TileColumns;
  auto input_tensor = cute::make_tensor(
      input, cute::make_shape(Rows, k),
      cute::make_stride(static_cast<int64_t>(k), cute::_1{}));
  auto trellis_tensor = cute::make_tensor(
      trellis, cute::make_shape(kWords, n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, cute::Int<kWords>{},
                        static_cast<int64_t>(n_tiles) * kWords));
  auto bank_tensor = cute::make_tensor(
      bank_ids, cute::make_shape(n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, static_cast<int64_t>(n_tiles)));
  auto input_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, input_tensor,
      WgmmaTmaSmemLayoutB{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_256{}));
  auto trellis_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, trellis_tensor,
      TrellisLayout{}(cute::_, cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::Int<kWords>{},
                       cute::Int<kP32N16TilesPerBlock>{},
                       cute::Int<kP32K16TilesPerStage>{}));
  auto bank_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, bank_tensor,
      P32BankTmaSmemLayout{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_16{}));
  HopperGroupedP32LaunchParams grouped{};
  grouped.launch_bank_alt_ids = bank_alt_id;
  auto kernel = qvq_p32_window_wgmma_m16_tma_kernel<
      TransitionBits, false, false, false, true, 1, 0, RowTiles,
      decltype(input_tma), decltype(trellis_tma), decltype(bank_tma),
      HopperGroupedP32LaunchParams>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(SharedStorage)));
  if (status != cudaSuccess) return status;

  std::memset(plan, 0, sizeof(*plan));
  PlanStorageWriter storage{
      reinterpret_cast<uint8_t*>(plan->host_storage),
      sizeof(plan->host_storage)};
  auto* launch = &plan->launches[0];
  launch->kernel_symbol = reinterpret_cast<const void*>(kernel);
  launch->kernel_name = "qvq_p32_wgmma_direct_rows";
  launch->grid_x = static_cast<unsigned>(n / kOutputColumns);
  launch->grid_y = static_cast<unsigned>(Rows / (RowTiles * kRows));
  launch->grid_z = 1;
  launch->block_x = kTmaThreads;
  launch->block_y = 1;
  launch->block_z = 1;
  launch->shared_memory_bytes = sizeof(SharedStorage);
  int arg = 0;
  if (!set_host_arg(launch, arg++, &storage, input_tma) ||
      !set_host_arg(launch, arg++, &storage, trellis_tma) ||
      !set_host_arg(launch, arg++, &storage, bank_tma)) {
    return cudaErrorInvalidValue;
  }
  set_device_arg(launch, arg++, levels);
  set_device_arg(launch, arg++, output);
  if (!set_host_arg(launch, arg++, &storage, grouped) ||
      !set_host_arg(launch, arg++, &storage, Rows) ||
      !set_host_arg(launch, arg++, &storage, k) ||
      !set_host_arg(launch, arg++, &storage, n)) {
    return cudaErrorInvalidValue;
  }
  constexpr int split_count = 1;
  constexpr int launch_bank_alt_id = 0;
  if (!set_host_arg(launch, arg++, &storage, split_count) ||
      !set_host_arg(launch, arg++, &storage, launch_bank_alt_id)) {
    return cudaErrorInvalidValue;
  }
  launch->arg_count = arg;
  plan->launch_count = 1;
  return cudaSuccess;
}

template <int TransitionBits>
cudaError_t build_direct_plan(
    const Element* input, const uint32_t* trellis, const uint8_t* bank_ids,
    const Element* levels, const uint8_t* bank_alt_id, float* output,
    int rows, int k, int n, int block_m, qvq_p32_launch_plan* plan) {
  return rows == 64
      ? build_direct_plan_rows<TransitionBits, 64, 4>(
            input, trellis, bank_ids, levels, bank_alt_id, output, k, n, plan)
      : n <= 512
          ? build_direct_plan_rows<TransitionBits, 128, 1>(
                input, trellis, bank_ids, levels, bank_alt_id, output, k, n, plan)
          : n <= 2048
              ? k == 8192
                  ? block_m == 64
                      ? build_direct_plan_rows<TransitionBits, 128, 4>(
                            input, trellis, bank_ids, levels, bank_alt_id, output, k, n, plan)
                      : build_direct_plan_rows<TransitionBits, 128, 8>(
                            input, trellis, bank_ids, levels, bank_alt_id, output, k, n, plan)
                  : build_direct_plan_rows<TransitionBits, 128, 2>(
                        input, trellis, bank_ids, levels, bank_alt_id, output, k, n, plan)
              : build_direct_plan_rows<TransitionBits, 128, 8>(
                    input, trellis, bank_ids, levels, bank_alt_id, output, k, n, plan);
}

template <int TransitionBits>
cudaError_t build_direct_grouped_gate_up_plan(
    const Element* input, const uint32_t* trellis, const uint8_t* bank_ids,
    const Element* levels, const uint8_t* bank_alt_ids, float* output,
    int k, int n, qvq_p32_launch_plan* plan) {
  constexpr int Rows = 128;
  constexpr int RowTiles = 8;
  constexpr int N64BlocksPerCta = 2;
  constexpr int kWords = 4 * TransitionBits;
  using TrellisLayout =
      P32TrellisTmaSmemLayoutFor<TransitionBits, N64BlocksPerCta>;
  using SharedStorage = P32WgmmaTmaSharedStorageFor<
      TransitionBits, N64BlocksPerCta, RowTiles>;
  const int k_tiles = k / kP32TileRows;
  const int n_tiles = n / kP32TileColumns;
  const int child_n = n / 2;
  auto input_tensor = cute::make_tensor(
      input, cute::make_shape(Rows, k),
      cute::make_stride(static_cast<int64_t>(k), cute::_1{}));
  auto trellis_tensor = cute::make_tensor(
      trellis, cute::make_shape(kWords, n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, cute::Int<kWords>{},
                        static_cast<int64_t>(n_tiles) * kWords));
  auto bank_tensor = cute::make_tensor(
      bank_ids, cute::make_shape(n_tiles, k_tiles),
      cute::make_stride(cute::_1{}, static_cast<int64_t>(n_tiles)));
  auto input_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, input_tensor,
      WgmmaTmaSmemLayoutB{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_256{}));
  auto trellis_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, trellis_tensor,
      TrellisLayout{}(cute::_, cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::Int<kWords>{},
                       cute::Int<2 * kP32N16TilesPerBlock>{},
                       cute::Int<kP32K16TilesPerStage>{}));
  auto bank_tma = cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, bank_tensor,
      P32BankTmaSmemLayout{}(cute::_, cute::_, cute::_0{}),
      cute::make_shape(cute::_16{}, cute::_16{}));
  const RawGroupedGateUpParams grouped{bank_alt_ids};
  auto kernel = qvq_p32_window_wgmma_m16_tma_kernel<
      TransitionBits, true, false, true, true, N64BlocksPerCta, false,
      RowTiles, decltype(input_tma), decltype(trellis_tma),
      decltype(bank_tma), RawGroupedGateUpParams>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(SharedStorage)));
  if (status != cudaSuccess) return status;
  std::memset(plan, 0, sizeof(*plan));
  PlanStorageWriter storage{reinterpret_cast<uint8_t*>(plan->host_storage), sizeof(plan->host_storage)};
  auto* launch = &plan->launches[0];
  launch->kernel_symbol = reinterpret_cast<const void*>(kernel);
  launch->kernel_name = "qvq_p32_wgmma_direct_grouped_gate_up";
  launch->grid_x = static_cast<unsigned>(child_n / (N64BlocksPerCta * kOutputColumns));
  launch->grid_y = 2;
  launch->grid_z = 1;
  launch->block_x = kTmaThreads + kThreads;
  launch->block_y = 1;
  launch->block_z = 1;
  launch->shared_memory_bytes = sizeof(SharedStorage);
  int arg = 0;
  if (!set_host_arg(launch, arg++, &storage, input_tma) ||
      !set_host_arg(launch, arg++, &storage, trellis_tma) ||
      !set_host_arg(launch, arg++, &storage, bank_tma)) return cudaErrorInvalidValue;
  set_device_arg(launch, arg++, levels);
  set_device_arg(launch, arg++, output);
  if (!set_host_arg(launch, arg++, &storage, grouped) ||
      !set_host_arg(launch, arg++, &storage, Rows) ||
      !set_host_arg(launch, arg++, &storage, k) ||
      !set_host_arg(launch, arg++, &storage, n)) return cudaErrorInvalidValue;
  constexpr int split_count = 1;
  constexpr int launch_bank_alt_id = 0;
  if (!set_host_arg(launch, arg++, &storage, split_count) ||
      !set_host_arg(launch, arg++, &storage, launch_bank_alt_id)) return cudaErrorInvalidValue;
  launch->arg_count = arg;
  plan->launch_count = 1;
  return cudaSuccess;
}

}  // namespace

#if !defined(QVQ_WGMMA_BITS_ONLY)
extern "C" uint32_t qvq_p32_wgmma_raw_abi_version(void) {
  return QVQ_WGMMA_RAW_ABI_VERSION;
}

extern "C" uint64_t qvq_p32_wgmma_raw_workspace_bytes(
    const QvqP32WgmmaRawConfig* c) {
  if (c == nullptr) return 0;
  if (c->algorithm == 2 || c->algorithm == 3 || c->algorithm == 4) return 0;
  const uint64_t padded_input = align_up(16ull * c->k * sizeof(Element), 256);
  const uint64_t partials =
      static_cast<uint64_t>(c->split_count) * 16ull * c->n * sizeof(float);
  return padded_input + partials;
}
#endif

static uint64_t raw_workspace_bytes(const QvqP32WgmmaRawConfig* c) {
  if (c == nullptr) return 0;
  if (c->algorithm == 2 || c->algorithm == 3 || c->algorithm == 4) return 0;
  const uint64_t padded_input = align_up(16ull * c->k * sizeof(Element), 256);
  const uint64_t partials =
      static_cast<uint64_t>(c->split_count) * 16ull * c->n * sizeof(float);
  return padded_input + partials;
}

extern "C" int qvq_p32_wgmma_raw_launch(
    const void* activation, const void* window, const void* bank_ids,
    const void* levels, const void* bank_alt_id, void* output,
    void* workspace, uint64_t workspace_bytes,
    const QvqP32WgmmaRawConfig* c, void* cuda_stream,
    char* error, uint64_t error_capacity) {
  if (c == nullptr || c->abi_version != QVQ_WGMMA_RAW_ABI_VERSION ||
      c->struct_bytes != sizeof(*c)) {
    return fail(error, error_capacity, "invalid QVQ WGMMA raw ABI config");
  }
  const bool ordered_m16 = c->algorithm == 1 && c->block_m == 0 &&
      c->block_n == 0 && c->m >= 1 && c->m <= 16;
  const bool direct_m64 = c->algorithm == 2 && c->block_m == 64 &&
      c->block_n == 64 && c->m == 64 && c->split_count == 1;
  const bool direct_m128 = c->algorithm == 3 &&
      (c->block_m == 128 || (c->block_m == 64 && c->k == 8192 && c->n == 2048)) &&
      c->block_n == 64 && c->m == 128 && c->split_count == 1;
  const bool grouped_gate_up = c->algorithm == 4 && c->block_m == 128 &&
      c->block_n == 128 && c->m == 128 && c->k == 2048 && c->n == 16384 &&
      c->split_count == 1;
  if ((!ordered_m16 && !direct_m64 && !direct_m128 && !grouped_gate_up) ||
      c->k < 256 || c->k % 256 != 0 ||
      c->n < 256 || c->n % 256 != 0 || c->transition_bits < 4 ||
      c->transition_bits > 7 || c->split_count < 1 || c->split_count > 64 ||
      (c->k / 16) % c->split_count != 0 ||
      ((c->k / 16) / c->split_count) % 16 != 0) {
    return fail(error, error_capacity, "unsupported QVQ WGMMA raw geometry");
  }
  const uint64_t required = raw_workspace_bytes(c);
  if (required != 0 && (workspace == nullptr || workspace_bytes < required)) {
    return fail(error, error_capacity, "QVQ WGMMA raw workspace is too small");
  }
  auto stream = static_cast<cudaStream_t>(cuda_stream);
  if (direct_m64 || direct_m128 || grouped_gate_up) {
    cudaError_t status = cudaSuccess;
#define QVQ_LAUNCH_DIRECT(BITS) launch_direct<BITS>(                        \
    static_cast<const Element*>(activation),                               \
    static_cast<const uint32_t*>(window),                                  \
    static_cast<const uint8_t*>(bank_ids),                                 \
    static_cast<const Element*>(levels),                                   \
    static_cast<const uint8_t*>(bank_alt_id), static_cast<float*>(output), \
    c->m, c->k, c->n, c->block_m, stream)
    if (grouped_gate_up) {
#define QVQ_LAUNCH_GROUPED(BITS) launch_direct_grouped_gate_up<BITS>(       \
    static_cast<const Element*>(activation),                               \
    static_cast<const uint32_t*>(window),                                  \
    static_cast<const uint8_t*>(bank_ids),                                 \
    static_cast<const Element*>(levels),                                   \
    static_cast<const uint8_t*>(bank_alt_id), static_cast<float*>(output), \
    c->k, c->n, stream)
      switch (c->transition_bits) {
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 4
        case 4: status = QVQ_LAUNCH_GROUPED(4); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 5
        case 5: status = QVQ_LAUNCH_GROUPED(5); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 6
        case 6: status = QVQ_LAUNCH_GROUPED(6); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 7
        case 7: status = QVQ_LAUNCH_GROUPED(7); break;
#endif
      }
#undef QVQ_LAUNCH_GROUPED
    } else switch (c->transition_bits) {
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 4
      case 4: status = QVQ_LAUNCH_DIRECT(4); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 5
      case 5: status = QVQ_LAUNCH_DIRECT(5); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 6
      case 6: status = QVQ_LAUNCH_DIRECT(6); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 7
      case 7: status = QVQ_LAUNCH_DIRECT(7); break;
#endif
    }
#undef QVQ_LAUNCH_DIRECT
    return status == cudaSuccess ? 0
        : fail(error, error_capacity, cudaGetErrorString(status));
  }
  auto* padded_input = static_cast<Element*>(workspace);
  const uint64_t partial_offset = align_up(16ull * c->k * sizeof(Element), 256);
  auto* partials = reinterpret_cast<float*>(
      static_cast<uint8_t*>(workspace) + partial_offset);
  constexpr int pad_threads = 256;
  const int padded_values = static_cast<int>(16 * c->k);
  pad_m16<<<(padded_values + pad_threads - 1) / pad_threads, pad_threads, 0, stream>>>(
      static_cast<const Element*>(activation), padded_input, c->m, c->k);
  cudaError_t status = cudaSuccess;
  switch (c->transition_bits) {
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 4
    case 4: status = launch_ordered<4>(
        padded_input, static_cast<const uint32_t*>(window),
        static_cast<const uint8_t*>(bank_ids), static_cast<const Element*>(levels),
        static_cast<const uint8_t*>(bank_alt_id), partials,
        16, c->k, c->n, c->split_count, stream); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 5
    case 5: status = launch_ordered<5>(
        padded_input, static_cast<const uint32_t*>(window),
        static_cast<const uint8_t*>(bank_ids), static_cast<const Element*>(levels),
        static_cast<const uint8_t*>(bank_alt_id), partials,
        16, c->k, c->n, c->split_count, stream); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 6
    case 6: status = launch_ordered<6>(
        padded_input, static_cast<const uint32_t*>(window),
        static_cast<const uint8_t*>(bank_ids), static_cast<const Element*>(levels),
        static_cast<const uint8_t*>(bank_alt_id), partials,
        16, c->k, c->n, c->split_count, stream); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 7
    case 7: status = launch_ordered<7>(
        padded_input, static_cast<const uint32_t*>(window),
        static_cast<const uint8_t*>(bank_ids), static_cast<const Element*>(levels),
        static_cast<const uint8_t*>(bank_alt_id), partials,
        16, c->k, c->n, c->split_count, stream); break;
#endif
  }
  if (status != cudaSuccess) return fail(error, error_capacity, cudaGetErrorString(status));
  constexpr int threads = 256;
  const int values = static_cast<int>(c->m * c->n);
  const int plane_stride = static_cast<int>(16 * c->n);
  reduce_ordered_rows<<<(values + threads - 1) / threads, threads, 0, stream>>>(
      partials, static_cast<float*>(output),
      values, plane_stride, c->split_count);
  status = cudaGetLastError();
  return status == cudaSuccess ? 0
      : fail(error, error_capacity, cudaGetErrorString(status));
}

extern "C" int qvq_p32_wgmma_raw_launch_plan(
    const void* activation, const void* window, const void* bank_ids,
    const void* levels, const void* bank_alt_id, void* output,
    const QvqP32WgmmaRawConfig* c, qvq_p32_launch_plan* plan,
    char* error, uint64_t error_capacity) {
  if (c == nullptr || plan == nullptr ||
      c->abi_version != QVQ_WGMMA_RAW_ABI_VERSION ||
      c->struct_bytes != sizeof(*c)) {
    return fail(error, error_capacity, "invalid QVQ WGMMA launch-plan config");
  }
  const bool direct_m64 = c->algorithm == 2 && c->block_m == 64 &&
      c->block_n == 64 && c->m == 64 && c->split_count == 1;
  const bool direct_m128 = c->algorithm == 3 &&
      (c->block_m == 128 || (c->block_m == 64 && c->k == 8192 && c->n == 2048)) &&
      c->block_n == 64 && c->m == 128 && c->split_count == 1;
  const bool grouped_gate_up = c->algorithm == 4 && c->block_m == 128 &&
      c->block_n == 128 && c->m == 128 && c->k == 2048 && c->n == 16384 &&
      c->split_count == 1;
  if ((!direct_m64 && !direct_m128 && !grouped_gate_up) || c->k < 256 || c->k % 256 != 0 ||
      c->n < 256 || c->n % 256 != 0 || c->transition_bits < 4 ||
      c->transition_bits > 7) {
    return fail(error, error_capacity, "unsupported QVQ WGMMA launch-plan geometry");
  }
  cudaError_t status = cudaSuccess;
#define QVQ_BUILD_DIRECT_PLAN(BITS) build_direct_plan<BITS>(                 \
    static_cast<const Element*>(activation),                                \
    static_cast<const uint32_t*>(window),                                   \
    static_cast<const uint8_t*>(bank_ids),                                  \
    static_cast<const Element*>(levels),                                    \
    static_cast<const uint8_t*>(bank_alt_id), static_cast<float*>(output),  \
    c->m, c->k, c->n, c->block_m, plan)
  if (grouped_gate_up) {
#define QVQ_BUILD_GROUPED_PLAN(BITS) build_direct_grouped_gate_up_plan<BITS>( \
    static_cast<const Element*>(activation),                                 \
    static_cast<const uint32_t*>(window),                                    \
    static_cast<const uint8_t*>(bank_ids),                                   \
    static_cast<const Element*>(levels),                                     \
    static_cast<const uint8_t*>(bank_alt_id), static_cast<float*>(output),   \
    c->k, c->n, plan)
    switch (c->transition_bits) {
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 4
      case 4: status = QVQ_BUILD_GROUPED_PLAN(4); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 5
      case 5: status = QVQ_BUILD_GROUPED_PLAN(5); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 6
      case 6: status = QVQ_BUILD_GROUPED_PLAN(6); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 7
      case 7: status = QVQ_BUILD_GROUPED_PLAN(7); break;
#endif
    }
#undef QVQ_BUILD_GROUPED_PLAN
  } else switch (c->transition_bits) {
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 4
    case 4: status = QVQ_BUILD_DIRECT_PLAN(4); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 5
    case 5: status = QVQ_BUILD_DIRECT_PLAN(5); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 6
    case 6: status = QVQ_BUILD_DIRECT_PLAN(6); break;
#endif
#if !defined(QVQ_WGMMA_BITS_ONLY) || QVQ_WGMMA_BITS_ONLY == 7
    case 7: status = QVQ_BUILD_DIRECT_PLAN(7); break;
#endif
  }
#undef QVQ_BUILD_DIRECT_PLAN
  return status == cudaSuccess ? 0
      : fail(error, error_capacity, cudaGetErrorString(status));
}
