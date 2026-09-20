// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#define QVQ_WGMMA_DEVICE_ONLY 1
#include "qvq_wgmma_cuda.cu"
#include "qvq_wgmma_raw_abi.h"

#include <cstdio>

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

}  // namespace

extern "C" uint32_t qvq_p32_wgmma_raw_abi_version(void) {
  return QVQ_WGMMA_RAW_ABI_VERSION;
}

extern "C" uint64_t qvq_p32_wgmma_raw_workspace_bytes(
    const QvqP32WgmmaRawConfig* c) {
  if (c == nullptr) return 0;
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
  if (c->algorithm != 1 || c->block_m != 0 || c->block_n != 0 ||
      c->m < 1 || c->m > 16 || c->k < 256 || c->k % 256 != 0 ||
      c->n < 256 || c->n % 256 != 0 || c->transition_bits < 4 ||
      c->transition_bits > 7 || c->split_count < 1 || c->split_count > 64 ||
      (c->k / 16) % c->split_count != 0 ||
      ((c->k / 16) / c->split_count) % 16 != 0) {
    return fail(error, error_capacity, "unsupported QVQ WGMMA raw geometry");
  }
  const uint64_t required = qvq_p32_wgmma_raw_workspace_bytes(c);
  if (workspace == nullptr || workspace_bytes < required) {
    return fail(error, error_capacity, "QVQ WGMMA raw workspace is too small");
  }
  auto stream = static_cast<cudaStream_t>(cuda_stream);
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
    case 4: status = launch_ordered<4>(
        padded_input, static_cast<const uint32_t*>(window),
        static_cast<const uint8_t*>(bank_ids), static_cast<const Element*>(levels),
        static_cast<const uint8_t*>(bank_alt_id), partials,
        16, c->k, c->n, c->split_count, stream); break;
    case 5: status = launch_ordered<5>(
        padded_input, static_cast<const uint32_t*>(window),
        static_cast<const uint8_t*>(bank_ids), static_cast<const Element*>(levels),
        static_cast<const uint8_t*>(bank_alt_id), partials,
        16, c->k, c->n, c->split_count, stream); break;
    case 6: status = launch_ordered<6>(
        padded_input, static_cast<const uint32_t*>(window),
        static_cast<const uint8_t*>(bank_ids), static_cast<const Element*>(levels),
        static_cast<const uint8_t*>(bank_alt_id), partials,
        16, c->k, c->n, c->split_count, stream); break;
    case 7: status = launch_ordered<7>(
        padded_input, static_cast<const uint32_t*>(window),
        static_cast<const uint8_t*>(bank_ids), static_cast<const Element*>(levels),
        static_cast<const uint8_t*>(bank_alt_id), partials,
        16, c->k, c->n, c->split_count, stream); break;
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
