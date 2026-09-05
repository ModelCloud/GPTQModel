// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_p32_abi.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

bool close_enough(float actual, float expected) {
  const float tolerance = 2.0e-4f + 2.0e-3f * std::fabs(expected);
  return std::fabs(actual - expected) <= tolerance;
}

struct DeviceBuffer {
  void* pointer = nullptr;
  ~DeviceBuffer() { cudaFree(pointer); }
  bool allocate(size_t bytes) {
    return check_cuda(cudaMalloc(&pointer, bytes), "allocate launch-plan buffer");
  }
};

// Launch exactly what the public descriptor says, without duplicating QvQ's
// specialization selection or interpreting its private host-value payload.
bool execute_plan(const qvq_p32_launch_plan& plan, cudaStream_t stream) {
  if (plan.launch_count < 1 ||
      plan.launch_count > QVQ_P32_LAUNCH_PLAN_MAX_LAUNCHES) return false;
  for (int index = 0; index < plan.launch_count; ++index) {
    const auto& launch = plan.launches[index];
    if (launch.kernel_symbol == nullptr || launch.kernel_name == nullptr ||
        launch.arg_count < 1 || launch.arg_count > QVQ_P32_LAUNCH_PLAN_MAX_ARGS ||
        launch.dependency_count != (index == 0 ? 0 : 1) ||
        (index != 0 && launch.dependencies[0] != 0)) return false;
    void* args[QVQ_P32_LAUNCH_PLAN_MAX_ARGS];
    for (int arg = 0; arg < launch.arg_count; ++arg) {
      const auto& value = launch.args[arg];
      if (value.address == nullptr) return false;
      if (value.type == QVQ_P32_LAUNCH_ARG_DEVICE_POINTER) {
        args[arg] = const_cast<void*>(static_cast<const void*>(&value.address));
      } else if (value.type == QVQ_P32_LAUNCH_ARG_HOST_VALUE) {
        const auto address = reinterpret_cast<uintptr_t>(value.address);
        const auto start = reinterpret_cast<uintptr_t>(plan.host_storage);
        if (value.size <= 0 || address < start ||
            address + value.size > start + sizeof(plan.host_storage)) return false;
        args[arg] = const_cast<void*>(value.address);
      } else {
        return false;
      }
    }
    if (!check_cuda(cudaLaunchKernel(
            launch.kernel_symbol, dim3(launch.grid_x, launch.grid_y, launch.grid_z),
            dim3(launch.block_x, launch.block_y, launch.block_z), args,
            launch.shared_memory_bytes, stream), "launch descriptor")) return false;
  }
  return true;
}

bool test_grouped_launch_plans() {
  constexpr int kK = 64;
  constexpr int kN = 80;
  constexpr int kMaxM = 16;
  constexpr int kTiles = (kK / 16) * (kN / 16);
  std::vector<half> input(kMaxM * kK);
  std::vector<half> levels(QVQ_P32_LEVEL_COUNT);
  std::vector<uint32_t> trellis(kTiles * 4 * QVQ_P32_TRANSITION_BITS_MAX);
  std::vector<uint8_t> banks(kTiles);
  const uint8_t alts[] = {1, 3, 2};
  for (int i = 0; i < input.size(); ++i)
    input[i] = __float2half((static_cast<float>(i % 31) - 15.0f) / 32.0f);
  for (int i = 0; i < levels.size(); ++i)
    levels[i] = __float2half((static_cast<float>(i) - 127.5f) / 128.0f);
  for (int i = 0; i < trellis.size(); ++i)
    trellis[i] = 0x13579bdfu ^ (0x9e3779b9u * static_cast<uint32_t>(i + 1));
  for (int i = 0; i < banks.size(); ++i) banks[i] = (i % 2) ? 0xa5 : 0x5a;

  DeviceBuffer x, t, l, b, a, native, recorded, scratch;
  const size_t bytes = kMaxM * kN * sizeof(float);
  if (!x.allocate(input.size() * sizeof(half)) ||
      !t.allocate(trellis.size() * sizeof(uint32_t)) ||
      !l.allocate(levels.size() * sizeof(half)) ||
      !b.allocate(banks.size()) || !a.allocate(sizeof(alts)) ||
      !native.allocate(bytes) || !recorded.allocate(bytes) ||
      !scratch.allocate(4 * bytes)) return false;
  if (!check_cuda(cudaMemcpy(x.pointer, input.data(), input.size() * sizeof(half), cudaMemcpyHostToDevice), "copy plan input") ||
      !check_cuda(cudaMemcpy(t.pointer, trellis.data(), trellis.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy plan trellis") ||
      !check_cuda(cudaMemcpy(l.pointer, levels.data(), levels.size() * sizeof(half), cudaMemcpyHostToDevice), "copy plan levels") ||
      !check_cuda(cudaMemcpy(b.pointer, banks.data(), banks.size(), cudaMemcpyHostToDevice), "copy plan banks") ||
      !check_cuda(cudaMemcpy(a.pointer, alts, sizeof(alts), cudaMemcpyHostToDevice), "copy plan alts")) return false;

  cudaStream_t stream = nullptr;
  if (!check_cuda(cudaStreamCreate(&stream), "create plan stream")) return false;
  struct StreamCleanup {
    cudaStream_t stream;
    ~StreamCleanup() { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
  } cleanup{stream};
  std::vector<float> expected(kMaxM * kN), actual(kMaxM * kN);
  int rejected = 0;
  // Each row violates one boundary shared by native launch and plan creation.
  constexpr int invalid_cases[][6] = {
      {0, 1, 1, 3, 1, 3}, {5, 1, 1, 3, 1, 3},
      {1, 129, 1, 3, 1, 3}, {1, 1, 5, 3, 1, 3},
      {1, 1, 1, 1, 1, 3}, {1, 1, 1, 4, 1, 3},
      {1, 1, 1, 3, 0, 3}, {1, 1, 1, 3, 3, 2},
      {1, 1, 1, 3, 1, 6}, {1, 1, 1, 2, 1, 4},
  };
  for (const auto& invalid : invalid_cases) {
    qvq_p32_launch_plan plan;
    const int plan_status = qvq_p32_grouped_launch_plan(
        x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
        static_cast<float*>(recorded.pointer), static_cast<float*>(scratch.pointer),
        1, kK, kN, 4, 4, invalid[0], invalid[1], invalid[2],
        QVQ_P32_VARIANT_SCALAR, 128, 2, 0, QVQ_P32_REDUCTION_NATIVE,
        invalid[3], invalid[4], invalid[5], &plan);
    const int launch_status = qvq_p32_grouped_window(
        x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
        static_cast<float*>(native.pointer), static_cast<float*>(scratch.pointer),
        1, kK, kN, 4, 4, invalid[0], invalid[1], invalid[2],
        QVQ_P32_VARIANT_SCALAR, 128, 2, 0, QVQ_P32_REDUCTION_NATIVE,
        invalid[3], invalid[4], invalid[5], stream);
    if (plan_status == 0 || launch_status == 0) {
      std::fprintf(stderr, "invalid grouped case %d was accepted\n", rejected);
      return false;
    }
    ++rejected;
  }
  int cases = 0;
  float max_absolute = 0;
  for (int bits = 4; bits <= 7; ++bits)
  for (int m : {1, 2, 3, 4, 5, 8, 15, 16})
  for (int threads : {64, 128, 256})
  for (int stage = 1; stage <= 4; ++stage)
  for (int groups : {2, 3})
  for (int pattern = 0; pattern < 3; ++pattern) {
    const int split0 = pattern == 0 ? 1 : 2;
    const int split1 = pattern == 2 ? 4 : 1;
    const int split2 = pattern == 0 ? 1 : 2;
    const int variant = m <= 4 ? QVQ_P32_VARIANT_SCALAR : QVQ_P32_VARIANT_BLOCK;
    const int end1 = groups == 2 ? kN / 16 : 3;
    qvq_p32_launch_plan plan;
    const int status = qvq_p32_grouped_launch_plan(
        x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
        static_cast<float*>(recorded.pointer), static_cast<float*>(scratch.pointer),
        m, kK, kN, bits, 4, split0, split1, split2, variant, threads, stage,
        0, QVQ_P32_REDUCTION_NATIVE, groups, 1, end1, &plan);
    const int expected_launches = 1 + (split0 > 1) + (split1 > 1) +
        (groups == 3 && split2 > 1);
    if (status != 0 || plan.launch_count != expected_launches ||
        qvq_p32_grouped_window(
            x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
            static_cast<float*>(native.pointer), static_cast<float*>(scratch.pointer),
            m, kK, kN, bits, 4, split0, split1, split2, variant, threads, stage,
            0, QVQ_P32_REDUCTION_NATIVE, groups, 1, end1, stream) != 0 ||
        !execute_plan(plan, stream) ||
        !check_cuda(cudaMemcpyAsync(expected.data(), native.pointer, m * kN * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy native plan reference") ||
        !check_cuda(cudaMemcpyAsync(actual.data(), recorded.pointer, m * kN * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy plan output") ||
        !check_cuda(cudaStreamSynchronize(stream), "synchronize plan")) {
      std::fprintf(stderr, "plan execution failed M=%d bits=%d threads=%d stage=%d groups=%d pattern=%d: %s\n",
                   m, bits, threads, stage, groups, pattern, qvq_last_error());
      return false;
    }
    for (int i = 0; i < m * kN; ++i) {
      const float error = std::fabs(actual[i] - expected[i]);
      max_absolute = std::max(max_absolute, error);
      // Both paths select the same kernel, arguments and reduction order.
      if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || error != 0) {
        std::fprintf(stderr, "plan mismatch M=%d bits=%d threads=%d stage=%d groups=%d pattern=%d index=%d drift=%g\n",
                     m, bits, threads, stage, groups, pattern, i, error);
        return false;
      }
    }
    ++cases;
  }
  std::printf("qvq_p32_launch_plans=PASS cases=%d rejected=%d max_absolute=%g K=%d N=%d dtype=f16/f32\n",
              cases, rejected, max_absolute, kK, kN);
  return true;
}

}  // namespace

int main() {
  if (qvq_p32_abi_version() != QVQ_P32_ABI_VERSION ||
      qvq_p32_kernel_version() != QVQ_P32_KERNEL_VERSION ||
      qvq_compiled_sm() != QVQ_P32_COMPILED_SM) {
    std::fprintf(stderr, "QVQ P32 runtime contract mismatch\n");
    return 1;
  }

  int device = 0;
  if (!check_cuda(cudaGetDevice(&device), "cudaGetDevice") ||
      !qvq_device_is_supported(device)) {
    std::fprintf(stderr, "QVQ P32 runtime smoke requires an SM80 device\n");
    return 1;
  }

  constexpr int kM = 1;
  constexpr int kK = 32;
  constexpr int kN = 16;
  constexpr int kTransitionBits = 4;
  constexpr int kSplitCount = 2;
  constexpr int kWordsPerTile = 4 * kTransitionBits;
  constexpr int kTileCount =
      (kK / QVQ_P32_TILE_SIZE) * (kN / QVQ_P32_TILE_SIZE);

  std::vector<half> input(kM * kK);
  for (int index = 0; index < input.size(); ++index) {
    input[index] = __float2half((static_cast<float>(index % 11) - 5.0f) / 16.0f);
  }
  std::vector<uint32_t> trellis(kTileCount * kWordsPerTile);
  for (int index = 0; index < trellis.size(); ++index) {
    trellis[index] = 0x13579bdfu ^ (0x9e3779b9u * static_cast<uint32_t>(index + 1));
  }
  std::vector<half> levels(QVQ_P32_LEVEL_COUNT);
  for (int index = 0; index < levels.size(); ++index) {
    levels[index] = __float2half((static_cast<float>(index) - 127.5f) / 128.0f);
  }
  const std::vector<uint8_t> bank_ids = {0xa5u, 0x5au};
  const std::vector<uint8_t> bank_alt_id = {3u};

  void* device_input = nullptr;
  void* device_trellis = nullptr;
  void* device_levels = nullptr;
  void* device_bank_ids = nullptr;
  void* device_bank_alt_id = nullptr;
  float* device_native_output = nullptr;
  float* device_partials_output = nullptr;
  float* device_native_workspace = nullptr;
  float* device_partials_workspace = nullptr;
  cudaStream_t stream = nullptr;

  const size_t output_bytes = kM * kN * sizeof(float);
  const size_t workspace_bytes = kSplitCount * output_bytes;
  bool ok = check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate") &&
      check_cuda(cudaMalloc(&device_input, input.size() * sizeof(half)),
                 "cudaMalloc(input)") &&
      check_cuda(cudaMalloc(&device_trellis,
                            trellis.size() * sizeof(uint32_t)),
                 "cudaMalloc(trellis)") &&
      check_cuda(cudaMalloc(&device_levels, levels.size() * sizeof(half)),
                 "cudaMalloc(levels)") &&
      check_cuda(cudaMalloc(&device_bank_ids, bank_ids.size()),
                 "cudaMalloc(bank_ids)") &&
      check_cuda(cudaMalloc(&device_bank_alt_id, bank_alt_id.size()),
                 "cudaMalloc(bank_alt_id)") &&
      check_cuda(cudaMalloc(&device_native_output, output_bytes),
                 "cudaMalloc(native_output)") &&
      check_cuda(cudaMalloc(&device_partials_output, output_bytes),
                 "cudaMalloc(partials_output)") &&
      check_cuda(cudaMalloc(&device_native_workspace, workspace_bytes),
                 "cudaMalloc(native_workspace)") &&
      check_cuda(cudaMalloc(&device_partials_workspace, workspace_bytes),
                 "cudaMalloc(partials_workspace)");

  if (ok) {
    ok = check_cuda(cudaMemcpyAsync(device_input, input.data(),
                                    input.size() * sizeof(half),
                                    cudaMemcpyHostToDevice, stream),
                    "copy input") &&
        check_cuda(cudaMemcpyAsync(device_trellis, trellis.data(),
                                   trellis.size() * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "copy trellis") &&
        check_cuda(cudaMemcpyAsync(device_levels, levels.data(),
                                   levels.size() * sizeof(half),
                                   cudaMemcpyHostToDevice, stream),
                   "copy levels") &&
        check_cuda(cudaMemcpyAsync(device_bank_ids, bank_ids.data(),
                                   bank_ids.size(), cudaMemcpyHostToDevice,
                                   stream),
                   "copy bank_ids") &&
        check_cuda(cudaMemcpyAsync(device_bank_alt_id, bank_alt_id.data(),
                                   bank_alt_id.size(), cudaMemcpyHostToDevice,
                                   stream),
                   "copy bank_alt_id");
  }

  if (ok && qvq_p32_window_with_row_groups(
                device_input, device_trellis, device_levels, device_bank_ids,
                device_bank_alt_id, device_native_output,
                device_native_workspace, kM, kK, kN, kTransitionBits,
                kSplitCount, QVQ_P32_VARIANT_SCALAR, 128, 2, 0,
                QVQ_P32_REDUCTION_NATIVE, 1, stream) != 0) {
    std::fprintf(stderr, "native reduction failed: %s\n", qvq_last_error());
    ok = false;
  }
  if (ok && qvq_p32_window(
                device_input, device_trellis, device_levels, device_bank_ids,
                device_bank_alt_id, device_partials_output,
                device_partials_workspace, kM, kK, kN, kTransitionBits,
                kSplitCount, QVQ_P32_VARIANT_SCALAR, 128, 2, 0,
                QVQ_P32_REDUCTION_PARTIALS, stream) != 0) {
    std::fprintf(stderr, "partials reduction failed: %s\n", qvq_last_error());
    ok = false;
  }

  std::vector<float> native_output(kM * kN);
  std::vector<float> partials(kSplitCount * kM * kN);
  if (ok) {
    ok = check_cuda(cudaMemcpyAsync(native_output.data(), device_native_output,
                                    output_bytes, cudaMemcpyDeviceToHost,
                                    stream),
                    "copy native output") &&
        check_cuda(cudaMemcpyAsync(partials.data(), device_partials_workspace,
                                   workspace_bytes, cudaMemcpyDeviceToHost,
                                   stream),
                   "copy partials") &&
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }

  if (ok) {
    for (int column = 0; column < kN; ++column) {
      const float expected = partials[column] + partials[kN + column];
      if (!close_enough(native_output[column], expected)) {
        std::fprintf(stderr,
                     "reduction mismatch at N=%d: native=%g partials=%g\n",
                     column, native_output[column], expected);
        ok = false;
        break;
      }
    }
  }

  if (stream != nullptr) cudaStreamSynchronize(stream);
  cudaFree(device_partials_workspace);
  cudaFree(device_native_workspace);
  cudaFree(device_partials_output);
  cudaFree(device_native_output);
  cudaFree(device_bank_alt_id);
  cudaFree(device_bank_ids);
  cudaFree(device_levels);
  cudaFree(device_trellis);
  cudaFree(device_input);
  if (stream != nullptr) cudaStreamDestroy(stream);

  if (!ok || !test_grouped_launch_plans()) return 1;
  std::printf("qvq_p32_runtime_smoke=PASS sm=%d split=%d reduction=native+partials\n",
              qvq_device_sm(device), kSplitCount);
  return 0;
}
