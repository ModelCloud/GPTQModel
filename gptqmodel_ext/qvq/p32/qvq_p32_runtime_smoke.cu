// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_p32_abi.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

bool close_enough(float actual, float expected) {
  return std::isfinite(actual) && std::isfinite(expected) &&
      std::fabs(actual - expected) <= 2.0e-3f;
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
  DeviceBuffer group_t, group_b, independent;
  const size_t bytes = kMaxM * kN * sizeof(float);
  if (!x.allocate(input.size() * sizeof(half)) ||
      !t.allocate(trellis.size() * sizeof(uint32_t)) ||
      !l.allocate(levels.size() * sizeof(half)) ||
      !b.allocate(banks.size()) || !a.allocate(sizeof(alts)) ||
      !native.allocate(bytes) || !recorded.allocate(bytes) ||
      !scratch.allocate(4 * bytes) ||
      !group_t.allocate(trellis.size() * sizeof(uint32_t)) ||
      !group_b.allocate(banks.size()) || !independent.allocate(bytes)) return false;
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
  for (int invalid_mode : {0, 3}) {
    qvq_p32_launch_plan plan;
    if (qvq_p32_grouped_launch_plan(
            x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
            static_cast<float*>(recorded.pointer), static_cast<float*>(scratch.pointer),
            1, kK, kN, 4, 4, 2, 1, 2, QVQ_P32_VARIANT_SCALAR, 128, 2,
            0, invalid_mode, 3, 1, 3, &plan) == 0 ||
        qvq_p32_grouped_window(
            x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
            static_cast<float*>(recorded.pointer), static_cast<float*>(scratch.pointer),
            1, kK, kN, 4, 4, 2, 1, 2, QVQ_P32_VARIANT_SCALAR, 128, 2,
            0, invalid_mode, 3, 1, 3, stream) == 0) {
      std::fprintf(stderr, "invalid grouped reduction mode %d was accepted\n", invalid_mode);
      return false;
    }
    ++rejected;
  }
  {
    qvq_p32_config invalid_tuning = {
        4, QVQ_P32_VARIANT_SCALAR, 128, 2, 0,
        QVQ_P32_REDUCTION_NATIVE, QVQ_P32_TUNING_EXTERNAL, 8, 4};
    qvq_p32_launch_plan plan;
    if (qvq_p32_grouped_launch_plan_tuned(
            x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
            static_cast<float*>(recorded.pointer), static_cast<float*>(scratch.pointer),
            1, kK, kN, 4, 4, 1, 1, 1, 2, 1, 3, &invalid_tuning, &plan) == 0) {
      std::fprintf(stderr, "unsupported external N geometry was accepted\n");
      return false;
    }
    ++rejected;
  }
  int cases = 0;
  float max_absolute = 0;
  float independent_max_absolute = 0;
  bool observed_nonzero_output = false;
  for (int bits = 4; bits <= 7; ++bits)
  for (int m = 1; m <= kMaxM; ++m)
  for (int threads : {64, 128, 256})
  for (int stage = 1; stage <= 4; ++stage)
  for (int groups : {2, 3})
  for (int pattern = 0; pattern < 4; ++pattern) {
    const int split0 = pattern == 0 ? 1 : (pattern == 3 ? 3 : 2);
    const int split1 = pattern == 2 ? 4 : (pattern == 3 ? 2 : 1);
    const int split2 = pattern == 0 ? 1 : 2;
    const int variant = m <= 4 ? QVQ_P32_VARIANT_SCALAR : QVQ_P32_VARIANT_BLOCK;
    const int end1 = groups == 2 ? kN / 16 : 3;
    const int warps = threads / 32;
    const qvq_p32_config tuned = {
        4,
        variant,
        threads,
        stage,
        0,
        QVQ_P32_REDUCTION_NATIVE,
        QVQ_P32_TUNING_EXTERNAL,
        variant == QVQ_P32_VARIANT_SCALAR ? 4 * warps : warps,
        warps,
    };
    qvq_p32_launch_plan plan;
    const int status = qvq_p32_grouped_launch_plan_tuned(
        x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
        static_cast<float*>(recorded.pointer), static_cast<float*>(scratch.pointer),
        m, kK, kN, bits, 4, split0, split1, split2, groups, 1, end1,
        &tuned, &plan);
    const int expected_launches = 1 + (split0 > 1) + (split1 > 1) +
        (groups == 3 && split2 > 1);
    if (status != 0 || plan.launch_count != expected_launches ||
        qvq_p32_grouped_window_tuned(
            x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
            static_cast<float*>(native.pointer), static_cast<float*>(scratch.pointer),
            m, kK, kN, bits, 4, split0, split1, split2, groups, 1, end1,
            &tuned, stream) != 0 ||
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
      observed_nonzero_output =
          observed_nonzero_output || std::fabs(expected[i]) > 1.0e-6f;
      // Both paths select the same kernel, arguments and reduction order.
      if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || error != 0) {
        std::fprintf(stderr, "plan mismatch M=%d bits=%d threads=%d stage=%d groups=%d pattern=%d index=%d drift=%g\n",
                     m, bits, threads, stage, groups, pattern, i, error);
        return false;
      }
    }
    // Independently pack and execute each group through the standard ABI.
    // Native/descriptor equality alone cannot detect a shared layout defect.
    const int splits[] = {split0, split1, split2};
    const int ends[] = {16, end1 * 16, kN};
    int group_begin = 0;
    for (int group = 0; group < groups; ++group) {
      const int width = ends[group] - group_begin;
      const int group_tiles = width / 16;
      std::vector<uint32_t> group_words((kK / 16) * group_tiles * 4 * bits);
      std::vector<uint8_t> group_banks((kK / 16) * group_tiles);
      for (int kt = 0; kt < kK / 16; ++kt)
      for (int nt = 0; nt < group_tiles; ++nt) {
        const int source_tile = kt * (kN / 16) + group_begin / 16 + nt;
        const int target_tile = kt * group_tiles + nt;
        std::copy_n(trellis.data() + source_tile * 4 * bits, 4 * bits,
                    group_words.data() + target_tile * 4 * bits);
        group_banks[target_tile] = banks[source_tile];
      }
      if (!check_cuda(cudaMemcpyAsync(group_t.pointer, group_words.data(), group_words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "copy independent trellis") ||
          !check_cuda(cudaMemcpyAsync(group_b.pointer, group_banks.data(), group_banks.size(), cudaMemcpyHostToDevice, stream), "copy independent banks") ||
          qvq_p32_window(
              x.pointer, group_t.pointer, l.pointer, group_b.pointer,
              static_cast<const uint8_t*>(a.pointer) + group,
              static_cast<float*>(independent.pointer), static_cast<float*>(scratch.pointer),
              m, kK, width, bits, splits[group], variant, threads, stage, 0,
              QVQ_P32_REDUCTION_NATIVE, stream) != 0 ||
          !check_cuda(cudaMemcpyAsync(actual.data(), independent.pointer, m * width * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy independent reference") ||
          !check_cuda(cudaStreamSynchronize(stream), "synchronize independent reference")) return false;
      for (int row = 0; row < m; ++row)
      for (int col = 0; col < width; ++col) {
        const float value = actual[row * width + col];
        const float error = std::fabs(value - expected[row * kN + group_begin + col]);
        independent_max_absolute = std::max(independent_max_absolute, error);
        if (!std::isfinite(value) || error > 2.0e-3f) {
          std::fprintf(stderr, "independent grouped mismatch M=%d bits=%d threads=%d stage=%d groups=%d pattern=%d group=%d row=%d col=%d drift=%g\n",
                       m, bits, threads, stage, groups, pattern, group, row, col, error);
          return false;
        }
      }
      group_begin = ends[group];
    }
    // Poison both destinations: split groups must not write final output,
    // and only packed split-group partials may be consumed by the caller.
    std::vector<float> partials(4 * m * kN);
    for (bool recorded_path : {false, true}) {
      if (!check_cuda(cudaMemsetAsync(recorded.pointer, 0xff, m * kN * sizeof(float), stream), "poison direct output") ||
          !check_cuda(cudaMemsetAsync(scratch.pointer, 0xff, partials.size() * sizeof(float), stream), "poison partial output")) return false;
      const int partial_status = recorded_path
          ? qvq_p32_grouped_launch_plan(
                x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
                static_cast<float*>(recorded.pointer), static_cast<float*>(scratch.pointer),
                m, kK, kN, bits, 4, split0, split1, split2, variant, threads, stage,
                0, QVQ_P32_REDUCTION_PARTIALS, groups, 1, end1, &plan)
          : qvq_p32_grouped_window(
                x.pointer, t.pointer, l.pointer, b.pointer, a.pointer,
                static_cast<float*>(recorded.pointer), static_cast<float*>(scratch.pointer),
                m, kK, kN, bits, 4, split0, split1, split2, variant, threads, stage,
                0, QVQ_P32_REDUCTION_PARTIALS, groups, 1, end1, stream);
      if (partial_status != 0 ||
          (recorded_path && (plan.launch_count != 1 || !execute_plan(plan, stream))) ||
          !check_cuda(cudaMemcpyAsync(actual.data(), recorded.pointer, m * kN * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy direct partial output") ||
          !check_cuda(cudaMemcpyAsync(partials.data(), scratch.pointer, partials.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy packed partial output") ||
          !check_cuda(cudaStreamSynchronize(stream), "synchronize partials")) return false;
      int offset = 0;
      int begin = 0;
      for (int group = 0; group < groups; ++group) {
        const int width = ends[group] - begin;
        for (int row = 0; row < m; ++row)
        for (int col = 0; col < width; ++col) {
          const int index = row * kN + begin + col;
          float sum = actual[index];
          if (splits[group] > 1) {
            if (!std::isnan(sum)) {
              std::fprintf(stderr, "partial mode wrote split-group final output\n");
              return false;
            }
            sum = 0;
            for (int split = 0; split < splits[group]; ++split)
              sum += partials[offset + split * m * width + row * width + col];
          }
          const float error = std::fabs(sum - expected[index]);
          if (!std::isfinite(sum) || error != 0) {
            std::fprintf(stderr, "grouped partial mismatch M=%d bits=%d threads=%d stage=%d groups=%d pattern=%d recorded=%d index=%d drift=%g\n",
                         m, bits, threads, stage, groups, pattern, recorded_path, index, error);
            return false;
          }
        }
        if (splits[group] > 1) offset += splits[group] * m * width;
        begin = ends[group];
      }
      for (int i = offset; i < partials.size(); ++i) {
        if (!std::isnan(partials[i])) {
          std::fprintf(stderr, "partial mode wrote outside packed split storage\n");
          return false;
        }
      }
    }
    ++cases;
  }
  if (!observed_nonzero_output) {
    std::fprintf(stderr,
                 "all P32 launch-plan outputs were zero; device kernel body may be unavailable for this SM\n");
    return false;
  }
  std::printf("qvq_p32_launch_plans=PASS cases=%d modes=native,partials_dispatch,partials_record rejected=%d max_absolute=%g K=%d N=%d dtype=f16/f32\n",
              cases, rejected, max_absolute, kK, kN);
  std::printf("qvq_p32_grouped_independent=PASS cases=%d max_absolute=%g limit=0.002\n",
              cases, independent_max_absolute);
  return true;
}

bool test_rank8_epilogue() {
  constexpr int kM = 3;
  constexpr int kN = 80;
  cudaStream_t stream = nullptr;
  if (!check_cuda(cudaStreamCreate(&stream), "create rank8 stream")) return false;
  for (int rank_count : {8, 16, 24}) {
    std::vector<float> base(kM * kN);
    std::vector<half> hidden(kM * rank_count);
    std::vector<float> rank8_b(rank_count * kN);
    for (int index = 0; index < base.size(); ++index) {
      base[index] = (static_cast<float>((index * 7) % 23) - 11.0f) / 17.0f;
    }
    for (int index = 0; index < hidden.size(); ++index) {
      hidden[index] = __float2half(
          (static_cast<float>((index * 5) % 19) - 9.0f) / 13.0f);
    }
    for (int index = 0; index < rank8_b.size(); ++index) {
      rank8_b[index] = (static_cast<float>((index * 3) % 29) - 14.0f) / 31.0f;
    }
    std::vector<float> expected = base;
    for (int row = 0; row < kM; ++row) {
      for (int column = 0; column < kN; ++column) {
        float correction = 0.0f;
        for (int rank = 0; rank < rank_count; ++rank) {
          correction += __half2float(hidden[row * rank_count + rank]) *
              rank8_b[rank * kN + column];
        }
        expected[row * kN + column] += correction;
      }
    }

    float* device_base = nullptr;
    void* device_hidden = nullptr;
    void* device_rank8_b = nullptr;
    bool ok = check_cuda(cudaMalloc(&device_base, base.size() * sizeof(float)),
                         "allocate rank8 base") &&
        check_cuda(cudaMalloc(&device_hidden, hidden.size() * sizeof(half)),
                   "allocate rank8 hidden") &&
        check_cuda(cudaMalloc(&device_rank8_b, rank8_b.size() * sizeof(float)),
                   "allocate rank8 B") &&
        check_cuda(cudaMemcpyAsync(device_base, base.data(),
                                   base.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "copy rank8 base") &&
        check_cuda(cudaMemcpyAsync(device_hidden, hidden.data(),
                                   hidden.size() * sizeof(half),
                                   cudaMemcpyHostToDevice, stream),
                   "copy rank8 hidden") &&
        check_cuda(cudaMemcpyAsync(device_rank8_b, rank8_b.data(),
                                   rank8_b.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "copy rank8 B");
    if (ok) {
      ok = qvq_p32_rank8_epilogue(
               device_base, device_hidden, device_rank8_b, device_base, kM, kN,
               rank_count, stream) == 0 &&
          check_cuda(cudaMemcpyAsync(base.data(), device_base,
                                     base.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy rank8 result") &&
          check_cuda(cudaStreamSynchronize(stream), "sync rank8 stream");
    }
    if (ok) {
      for (int index = 0; index < base.size(); ++index) {
        const float error = std::fabs(base[index] - expected[index]);
        if (error > 1.0e-5f || !std::isfinite(base[index])) {
          std::fprintf(stderr,
                       "rank8 epilogue mismatch rank_count=%d index=%d error=%g\n",
                       rank_count, index, error);
          ok = false;
          break;
        }
      }
    }
    cudaFree(device_rank8_b);
    cudaFree(device_hidden);
    cudaFree(device_base);
    if (!ok) {
      cudaStreamDestroy(stream);
      return false;
    }
  }
  cudaStreamDestroy(stream);
  std::printf("qvq_p32_rank8_epilogue=PASS M=%d N=%d rank_counts=8,16,24 alias=base\n",
              kM, kN);
  return true;
}

float round_half(float value) {
  return __half2float(__float2half_rn(value));
}

bool test_rank8_hadamard_epilogue() {
  constexpr int kM = 128;
  constexpr int kRank = 8;
  cudaStream_t stream = nullptr;
  if (!check_cuda(cudaStreamCreate(&stream), "create rank8 Hadamard stream")) return false;
  for (const int n : {16, 32, 64, 512, 2048, 8192}) {
    const bool normalize_first = n >= 2048;
    std::vector<float> base(kM * n);
    std::vector<half> hidden(kM * kRank);
    std::vector<float> rank8_b(kRank * n);
    std::vector<half> scale(n);
    std::vector<half> expected(kM * n);
    std::vector<half> actual(kM * n);
    std::vector<half> base_expected(kM * n);
    std::vector<half> base_actual(kM * n);
    for (int index = 0; index < base.size(); ++index) {
      base[index] = (static_cast<float>((index * 7) % 43) - 21.0f) / 64.0f;
    }
    for (int index = 0; index < hidden.size(); ++index) {
      hidden[index] = __float2half_rn(
          (static_cast<float>((index * 5) % 19) - 9.0f) / 32.0f);
    }
    for (int index = 0; index < rank8_b.size(); ++index) {
      rank8_b[index] = (static_cast<float>((index * 3) % 29) - 14.0f) / 128.0f;
    }
    for (int column = 0; column < n; ++column) {
      scale[column] = __float2half_rn(
          0.75f + static_cast<float>(column % 17) / 64.0f);
    }
    const float sqrt_n = std::sqrt(static_cast<float>(n));
    const float divisor = round_half(sqrt_n);
    const float reciprocal = 1.0f / sqrt_n;
    std::vector<float> row_values(n);
    for (int row = 0; row < kM; ++row) {
      for (int column = 0; column < n; ++column) {
        float correction = 0.0f;
        for (int rank = 0; rank < kRank; ++rank) {
          correction += __half2float(hidden[row * kRank + rank]) *
              rank8_b[rank * n + column];
        }
        float value = round_half(base[row * n + column] + correction);
        if (normalize_first) value = round_half(value / divisor);
        row_values[column] = value;
      }
      for (int bit = 1; bit < n; bit <<= 1) {
        for (int index = 0; index < n; ++index) {
          const int peer = index ^ bit;
          if (index < peer) {
            const float first = row_values[index];
            const float second = row_values[peer];
            row_values[index] = round_half(first + second);
            row_values[peer] = round_half(first - second);
          }
        }
      }
      for (int column = 0; column < n; ++column) {
        float value = row_values[column];
        if (!normalize_first) value = round_half(value * reciprocal);
        value = round_half(value * __half2float(scale[column]));
        expected[row * n + column] = __float2half_rn(value);
      }
    }
    for (int row = 0; row < kM; ++row) {
      for (int column = 0; column < n; ++column) {
        float value = round_half(base[row * n + column]);
        if (normalize_first) value = round_half(value / divisor);
        row_values[column] = value;
      }
      for (int bit = 1; bit < n; bit <<= 1) {
        for (int index = 0; index < n; ++index) {
          const int peer = index ^ bit;
          if (index < peer) {
            const float first = row_values[index];
            const float second = row_values[peer];
            row_values[index] = round_half(first + second);
            row_values[peer] = round_half(first - second);
          }
        }
      }
      for (int column = 0; column < n; ++column) {
        float value = row_values[column];
        if (!normalize_first) value = round_half(value * reciprocal);
        value = round_half(value * __half2float(scale[column]));
        base_expected[row * n + column] = __float2half_rn(value);
      }
    }

    float* device_base = nullptr;
    half* device_hidden = nullptr;
    float* device_rank8_b = nullptr;
    half* device_scale = nullptr;
    half* device_output = nullptr;
    bool ok = check_cuda(cudaMalloc(&device_base, base.size() * sizeof(float)), "allocate fused base") &&
        check_cuda(cudaMalloc(&device_hidden, hidden.size() * sizeof(half)), "allocate fused hidden") &&
        check_cuda(cudaMalloc(&device_rank8_b, rank8_b.size() * sizeof(float)), "allocate fused B") &&
        check_cuda(cudaMalloc(&device_scale, scale.size() * sizeof(half)), "allocate fused scale") &&
        check_cuda(cudaMalloc(&device_output, actual.size() * sizeof(half)), "allocate fused output") &&
        check_cuda(cudaMemcpyAsync(device_base, base.data(), base.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy fused base") &&
        check_cuda(cudaMemcpyAsync(device_hidden, hidden.data(), hidden.size() * sizeof(half), cudaMemcpyHostToDevice, stream), "copy fused hidden") &&
        check_cuda(cudaMemcpyAsync(device_rank8_b, rank8_b.data(), rank8_b.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy fused B") &&
        check_cuda(cudaMemcpyAsync(device_scale, scale.data(), scale.size() * sizeof(half), cudaMemcpyHostToDevice, stream), "copy fused scale");
    if (ok) {
      ok = qvq_p32_rank8_hadamard_epilogue(
               device_base, device_hidden, device_rank8_b, device_scale,
               device_output, kM, n, kRank, normalize_first, stream) == 0 &&
          check_cuda(cudaMemcpyAsync(actual.data(), device_output,
                                     actual.size() * sizeof(half),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy fused result") &&
          check_cuda(cudaStreamSynchronize(stream), "sync fused rank8 stream");
    }
    if (ok) {
      for (int index = 0; index < actual.size(); ++index) {
        if (__half_as_ushort(actual[index]) != __half_as_ushort(expected[index])) {
          std::fprintf(stderr,
                       "rank8 Hadamard mismatch N=%d index=%d actual=%g expected=%g\n",
                       n, index, __half2float(actual[index]),
                       __half2float(expected[index]));
          ok = false;
          break;
        }
      }
    }
    if (ok) {
      ok = qvq_p32_hadamard_epilogue(
               device_base, device_scale, device_output, kM, n,
               normalize_first, stream) == 0 &&
          check_cuda(cudaMemcpyAsync(base_actual.data(), device_output,
                                     base_actual.size() * sizeof(half),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy base fused result") &&
          check_cuda(cudaStreamSynchronize(stream), "sync base fused stream");
    }
    if (ok) {
      for (int index = 0; index < base_actual.size(); ++index) {
        if (__half_as_ushort(base_actual[index]) !=
            __half_as_ushort(base_expected[index])) {
          std::fprintf(stderr,
                       "base Hadamard mismatch N=%d index=%d actual=%g expected=%g\n",
                       n, index, __half2float(base_actual[index]),
                       __half2float(base_expected[index]));
          ok = false;
          break;
        }
      }
    }
    cudaFree(device_output);
    cudaFree(device_scale);
    cudaFree(device_rank8_b);
    cudaFree(device_hidden);
    cudaFree(device_base);
    if (!ok) {
      cudaStreamDestroy(stream);
      return false;
    }
  }
  cudaStreamDestroy(stream);
  std::printf(
      "qvq_p32_rank8_hadamard_epilogue=PASS qvq_p32_hadamard_epilogue=PASS M=%d N=16,32,64,512,2048,8192 bitwise_fp16\n",
      kM);
  return true;
}

bool test_rank8_project() {
  constexpr int kM = 3;
  constexpr int kK = 96;
  cudaStream_t stream = nullptr;
  if (!check_cuda(cudaStreamCreate(&stream), "create rank8 project stream")) return false;
  for (int rank_count : {8, 16, 24}) {
    std::vector<float> input(kM * kK);
    std::vector<half> rank8_a(kK * rank_count);
    std::vector<half> expected(kM * rank_count);
    std::vector<half> actual(kM * rank_count);
    for (int index = 0; index < input.size(); ++index) {
      input[index] = (static_cast<float>((index * 11) % 37) - 18.0f) / 23.0f;
    }
    for (int index = 0; index < rank8_a.size(); ++index) {
      rank8_a[index] = __float2half(
          (static_cast<float>((index * 7) % 31) - 15.0f) / 29.0f);
    }
    for (int row = 0; row < kM; ++row) {
      for (int rank = 0; rank < rank_count; ++rank) {
        float value = 0.0f;
        for (int k = 0; k < kK; ++k) {
          value += input[row * kK + k] *
              __half2float(rank8_a[k * rank_count + rank]);
        }
        expected[row * rank_count + rank] = __float2half_rn(value);
      }
    }

    float* device_input = nullptr;
    half* device_rank8_a = nullptr;
    half* device_hidden = nullptr;
    bool ok = check_cuda(cudaMalloc(&device_input, input.size() * sizeof(float)),
                         "allocate rank8 project input") &&
        check_cuda(cudaMalloc(&device_rank8_a, rank8_a.size() * sizeof(half)),
                   "allocate rank8 project A") &&
        check_cuda(cudaMalloc(&device_hidden, actual.size() * sizeof(half)),
                   "allocate rank8 project hidden") &&
        check_cuda(cudaMemcpyAsync(device_input, input.data(),
                                   input.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "copy rank8 project input") &&
        check_cuda(cudaMemcpyAsync(device_rank8_a, rank8_a.data(),
                                   rank8_a.size() * sizeof(half),
                                   cudaMemcpyHostToDevice, stream),
                   "copy rank8 project A");
    if (ok) {
      ok = qvq_p32_rank8_project(
               device_input, device_rank8_a, device_hidden, kM, kK,
               rank_count, stream) == 0 &&
          check_cuda(cudaMemcpyAsync(actual.data(), device_hidden,
                                     actual.size() * sizeof(half),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy rank8 project result") &&
          check_cuda(cudaStreamSynchronize(stream), "sync rank8 project stream");
    }
    if (ok) {
      for (int index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(__half2float(actual[index]) -
                                     __half2float(expected[index]));
        if (error > 2.0e-3f || !std::isfinite(__half2float(actual[index]))) {
          std::fprintf(stderr,
                       "rank8 project mismatch rank_count=%d index=%d error=%g\n",
                       rank_count, index, error);
          ok = false;
          break;
        }
      }
    }
    cudaFree(device_hidden);
    cudaFree(device_rank8_a);
    cudaFree(device_input);
    if (!ok) {
      cudaStreamDestroy(stream);
      return false;
    }
  }
  cudaStreamDestroy(stream);
  std::printf("qvq_p32_rank8_project=PASS M=%d K=%d rank_counts=8,16,24 output=f16\n",
              kM, kK);
  return true;
}

bool test_rank8_project_hadamard_fusion(int target_m = 0, int target_n = 0,
                                        int target_warps = 0,
                                        int target_threads = 0) {
  constexpr int kRank = 8;
  constexpr int kK = 2048;
  constexpr int kRepeats = 24;
  cudaStream_t stream = nullptr;
  if (!check_cuda(cudaStreamCreate(&stream), "create fused project/Hadamard stream"))
    return false;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!check_cuda(cudaEventCreate(&start), "create fused timing start") ||
      !check_cuda(cudaEventCreate(&stop), "create fused timing stop")) {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return false;
  }

  for (const int m : {1, 128, 960}) {
    if (target_m != 0 && m != target_m) continue;
    for (const int n : {512, 2048, 8192}) {
      if (target_n != 0 && n != target_n) continue;
      const bool normalize_first = n >= 2048;
      std::vector<float> input(m * kK);
      std::vector<half> rank8_a(kK * kRank);
      std::vector<float> base(m * n);
      std::vector<float> rank8_b(kRank * n);
      std::vector<half> scale(n);
      std::vector<half> hidden(m * kRank);
      std::vector<half> expected(m * n);
      std::vector<half> actual(m * n);
      for (int index = 0; index < input.size(); ++index)
        input[index] = (static_cast<float>((index * 17) % 101) - 50.0f) / 128.0f;
      for (int index = 0; index < rank8_a.size(); ++index)
        rank8_a[index] = __float2half_rn(
            (static_cast<float>((index * 13) % 67) - 33.0f) / 512.0f);
      for (int index = 0; index < base.size(); ++index)
        base[index] = (static_cast<float>((index * 7) % 91) - 45.0f) / 64.0f;
      for (int index = 0; index < rank8_b.size(); ++index)
        rank8_b[index] = (static_cast<float>((index * 3) % 59) - 29.0f) / 256.0f;
      for (int column = 0; column < n; ++column)
        scale[column] = __float2half_rn(
            0.75f + static_cast<float>(column % 17) / 64.0f);

      DeviceBuffer device_input, device_a, device_base, device_b, device_scale;
      DeviceBuffer device_hidden, device_expected, device_actual;
      bool ok = device_input.allocate(input.size() * sizeof(float)) &&
          device_a.allocate(rank8_a.size() * sizeof(half)) &&
          device_base.allocate(base.size() * sizeof(float)) &&
          device_b.allocate(rank8_b.size() * sizeof(float)) &&
          device_scale.allocate(scale.size() * sizeof(half)) &&
          device_hidden.allocate(hidden.size() * sizeof(half)) &&
          device_expected.allocate(expected.size() * sizeof(half)) &&
          device_actual.allocate(actual.size() * sizeof(half));
      if (ok) {
        ok = check_cuda(cudaMemcpyAsync(device_input.pointer, input.data(),
                                        input.size() * sizeof(float),
                                        cudaMemcpyHostToDevice, stream), "copy fused input") &&
            check_cuda(cudaMemcpyAsync(device_a.pointer, rank8_a.data(),
                                       rank8_a.size() * sizeof(half),
                                       cudaMemcpyHostToDevice, stream), "copy fused A") &&
            check_cuda(cudaMemcpyAsync(device_base.pointer, base.data(),
                                       base.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream), "copy fused base") &&
            check_cuda(cudaMemcpyAsync(device_b.pointer, rank8_b.data(),
                                       rank8_b.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream), "copy fused B") &&
            check_cuda(cudaMemcpyAsync(device_scale.pointer, scale.data(),
                                       scale.size() * sizeof(half),
                                       cudaMemcpyHostToDevice, stream), "copy fused scale");
      }
      if (ok) {
        ok = qvq_p32_rank8_project(
                 device_input.pointer, device_a.pointer, device_hidden.pointer,
                 m, kK, kRank, stream) == 0 &&
            qvq_p32_rank8_hadamard_epilogue(
                 static_cast<const float*>(device_base.pointer),
                 device_hidden.pointer, device_b.pointer, device_scale.pointer,
                 device_expected.pointer, m, n, kRank, normalize_first, stream) == 0 &&
            qvq_p32_rank8_hadamard_project(
                 device_input.pointer, device_a.pointer,
                 static_cast<const float*>(device_base.pointer), device_b.pointer,
                 device_scale.pointer, device_actual.pointer,
                 m, kK, n, kRank, normalize_first, stream) == 0 &&
            check_cuda(cudaMemcpyAsync(expected.data(), device_expected.pointer,
                                       expected.size() * sizeof(half),
                                       cudaMemcpyDeviceToHost, stream), "copy separate result") &&
            check_cuda(cudaMemcpyAsync(actual.data(), device_actual.pointer,
                                       actual.size() * sizeof(half),
                                       cudaMemcpyDeviceToHost, stream), "copy fused result") &&
            check_cuda(cudaStreamSynchronize(stream), "sync fused parity run");
      }
      if (ok) {
        for (int index = 0; index < actual.size(); ++index) {
          if (__half_as_ushort(actual[index]) != __half_as_ushort(expected[index])) {
            std::fprintf(stderr,
                         "fused rank8 mismatch M=%d K=%d N=%d index=%d actual=%g expected=%g\n",
                         m, kK, n, index, __half2float(actual[index]),
                         __half2float(expected[index]));
            ok = false;
            break;
          }
        }
      }
      const int max_threads = std::max(32, std::min(n / 2, 1024));
      std::vector<qvq_p32_rank8_hadamard_config> candidates;
      for (const int threads : {32, 64, 128, 256, 512, 1024}) {
        if (threads > max_threads) continue;
        for (const int projection_warps : {1, 2, 4, 6, 8}) {
          if (projection_warps > threads / 32) continue;
          candidates.push_back({
              QVQ_P32_RANK8_CONFIG_VERSION,
              sizeof(qvq_p32_rank8_hadamard_config),
              QVQ_P32_TUNING_EXTERNAL,
              projection_warps,
              threads,
              QVQ_P32_RANK8_ROWS_PER_CTA,
          });
        }
      }
      for (const auto& config : candidates) {
        if (target_warps != 0 &&
            config.projection_warps != target_warps) continue;
        if (target_threads != 0 && config.threads != target_threads) continue;
        if (ok) {
          ok = qvq_p32_rank8_hadamard_project_tuned(
                   device_input.pointer, device_a.pointer,
                   static_cast<const float*>(device_base.pointer), device_b.pointer,
                   device_scale.pointer, device_actual.pointer, m, kK, n, kRank,
                   normalize_first, &config, stream) == 0 &&
              check_cuda(cudaMemcpyAsync(actual.data(), device_actual.pointer,
                                         actual.size() * sizeof(half),
                                         cudaMemcpyDeviceToHost, stream),
                         "copy tuned fused result") &&
              check_cuda(cudaStreamSynchronize(stream), "sync tuned parity run");
        }
        if (ok) {
          for (int index = 0; index < actual.size(); ++index) {
            if (__half_as_ushort(actual[index]) != __half_as_ushort(expected[index])) {
              std::fprintf(stderr,
                           "tuned fused rank8 mismatch M=%d K=%d N=%d warps=%d threads=%d index=%d actual=%g expected=%g\n",
                           m, kK, n, config.projection_warps, config.threads,
                           index, __half2float(actual[index]),
                           __half2float(expected[index]));
              ok = false;
              break;
            }
          }
        }
        for (int warmup = 0; ok && warmup < 4; ++warmup) {
          ok = qvq_p32_rank8_hadamard_project_tuned(
              device_input.pointer, device_a.pointer,
              static_cast<const float*>(device_base.pointer), device_b.pointer,
              device_scale.pointer, device_actual.pointer, m, kK, n, kRank,
              normalize_first, &config, stream) == 0;
        }

        float separate_ms = 0.0f;
        float fused_ms = 0.0f;
        for (int round = 0; ok && round < 3; ++round) {
          const bool fused_first = (round & 1) != 0;
          for (int arm = 0; arm < 2; ++arm) {
            const bool fused = arm == 0 ? fused_first : !fused_first;
            if (!check_cuda(cudaEventRecord(start, stream), "record timing start")) {
              ok = false;
              break;
            }
            for (int repeat = 0; repeat < kRepeats; ++repeat) {
              if (fused) {
                ok = qvq_p32_rank8_hadamard_project_tuned(
                         device_input.pointer, device_a.pointer,
                         static_cast<const float*>(device_base.pointer),
                         device_b.pointer, device_scale.pointer, device_actual.pointer,
                         m, kK, n, kRank, normalize_first, &config, stream) == 0;
              } else {
                ok = qvq_p32_rank8_project(
                         device_input.pointer, device_a.pointer, device_hidden.pointer,
                         m, kK, kRank, stream) == 0 &&
                    qvq_p32_rank8_hadamard_epilogue(
                         static_cast<const float*>(device_base.pointer),
                         device_hidden.pointer, device_b.pointer, device_scale.pointer,
                         device_expected.pointer, m, n, kRank, normalize_first, stream) == 0;
              }
              if (!ok) break;
            }
            if (!ok || !check_cuda(cudaEventRecord(stop, stream), "record timing stop") ||
                !check_cuda(cudaEventSynchronize(stop), "sync timing stop")) {
              ok = false;
              break;
            }
            float elapsed_ms = 0.0f;
            if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                            "read event timing")) {
              ok = false;
              break;
            }
            if (fused) fused_ms += elapsed_ms / kRepeats / 3.0f;
            else separate_ms += elapsed_ms / kRepeats / 3.0f;
          }
        }
        if (ok) {
          std::printf("rank8_project_hadamard=PASS M=%d K=%d N=%d warps=%d threads=%d bitwise=1 separate_ms=%.6f fused_ms=%.6f speedup=%.4f\n",
                      m, kK, n, config.projection_warps, config.threads,
                      separate_ms, fused_ms, separate_ms / fused_ms);
        }
        if (!ok) break;
      }
      if (!ok) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return false;
      }
    }
  }
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return true;
}

bool test_rank8_project_full_context_rows() {
  constexpr int kM = 131070;
  constexpr int kK = 1;
  constexpr int kRankCount = 8;
  cudaStream_t stream = nullptr;
  float* input = nullptr;
  half* rank8_a = nullptr;
  half* hidden = nullptr;
  float* base = nullptr;
  float* rank8_b = nullptr;
  bool ok = check_cuda(cudaStreamCreate(&stream), "create full-context rank8 stream") &&
      check_cuda(cudaMalloc(&input, kM * sizeof(float)),
                 "allocate full-context rank8 input") &&
      check_cuda(cudaMalloc(&rank8_a, kRankCount * sizeof(half)),
                 "allocate full-context rank8 A") &&
      check_cuda(cudaMalloc(&hidden, kM * kRankCount * sizeof(half)),
                 "allocate full-context rank8 hidden") &&
      check_cuda(cudaMalloc(&base, kM * sizeof(float)),
                 "allocate full-context rank8 base") &&
      check_cuda(cudaMalloc(&rank8_b, kRankCount * sizeof(float)),
                 "allocate full-context rank8 B") &&
      check_cuda(cudaMemsetAsync(input, 0, kM * sizeof(float), stream),
                 "clear full-context rank8 input") &&
      check_cuda(cudaMemsetAsync(rank8_a, 0, kRankCount * sizeof(half), stream),
                 "clear full-context rank8 A") &&
      check_cuda(cudaMemsetAsync(base, 0, kM * sizeof(float), stream),
                 "clear full-context rank8 base") &&
      check_cuda(cudaMemsetAsync(rank8_b, 0, kRankCount * sizeof(float), stream),
                 "clear full-context rank8 B");
  if (ok) {
    ok = qvq_p32_rank8_project(
             input, rank8_a, hidden, kM, kK, kRankCount, stream) == 0 &&
        qvq_p32_rank8_epilogue(
             base, hidden, rank8_b, base, kM, 1, kRankCount, stream) == 0 &&
        check_cuda(cudaStreamSynchronize(stream),
                   "sync full-context rank8 recovery");
  }
  half last{};
  if (ok) {
    float last_base = 1.0f;
    ok = check_cuda(cudaMemcpy(
             &last, hidden + (kM - 1) * kRankCount + (kRankCount - 1),
             sizeof(last), cudaMemcpyDeviceToHost),
             "copy full-context rank8 project last row") &&
        check_cuda(cudaMemcpy(&last_base, base + kM - 1, sizeof(last_base),
                              cudaMemcpyDeviceToHost),
                   "copy full-context rank8 epilogue last row") &&
        __half2float(last) == 0.0f && last_base == 0.0f;
  }
  cudaFree(rank8_b);
  cudaFree(base);
  cudaFree(hidden);
  cudaFree(rank8_a);
  cudaFree(input);
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (ok) std::printf("qvq_p32_rank8_full_context=PASS M=%d project+epilogue\n", kM);
  return ok;
}

}  // namespace

int test_standard_partials(int rows, int transition_bits = 4,
                           int threads = 128, int stage = 2,
                           int row_groups = QVQ_P32_ROW_GROUPS_AUTO,
                           int size_k = 80, int size_n = 16) {
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

  const int kM = rows;
  const int kK = size_k;
  const int kN = size_n;
  const int kTransitionBits = transition_bits;
  constexpr int kSplitCount = 3;
  const int variant = rows <= 4 ? QVQ_P32_VARIANT_SCALAR : QVQ_P32_VARIANT_BLOCK;
  const int kWordsPerTile = 4 * kTransitionBits;
  const int kTileCount =
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
  std::vector<uint8_t> bank_ids(kTileCount);
  for (int index = 0; index < kTileCount; ++index) {
    bank_ids[index] = index % 2 == 0 ? 0xa5u : 0x5au;
  }
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
                kSplitCount, variant, threads, stage, 0,
                QVQ_P32_REDUCTION_NATIVE, 1, stream) != 0) {
    std::fprintf(stderr, "native reduction failed: %s\n", qvq_last_error());
    ok = false;
  }
  if (ok && qvq_p32_window_with_row_groups(
                device_input, device_trellis, device_levels, device_bank_ids,
                device_bank_alt_id, device_partials_output,
                device_partials_workspace, kM, kK, kN, kTransitionBits,
                kSplitCount, variant, threads, stage, 0,
                QVQ_P32_REDUCTION_PARTIALS, row_groups, stream) != 0) {
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
    for (int column = 0; column < kM * kN; ++column) {
      float expected = partials[column];
      for (int split = 1; split < kSplitCount; ++split) {
        expected += partials[split * kM * kN + column];
      }
      if (!close_enough(native_output[column], expected)) {
        std::fprintf(stderr,
                     "reduction mismatch M=%d rate=%d threads=%d stage=%d row_groups=%d at element=%d: native=%g partials=%g\n",
                     kM, transition_bits, threads, stage, row_groups,
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

  if (!ok) return 1;
  std::printf("qvq_p32_standard_partials=PASS sm=%d M=%d K=%d N=%d rate=%d split=%d threads=%d stage=%d row_groups=%d reduction=native+partials\n",
              qvq_device_sm(device), kM, kK, kN, transition_bits,
              kSplitCount, threads, stage, row_groups);
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--rank8") == 0) {
    return test_rank8_epilogue() && test_rank8_hadamard_epilogue() ? 0 : 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--rank8-project") == 0) {
    return test_rank8_project() && test_rank8_project_full_context_rows() ? 0 : 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--rank8-fused") == 0) {
    return test_rank8_project_hadamard_fusion() ? 0 : 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--rank8-fused-target") == 0) {
    return test_rank8_project_hadamard_fusion(128, 8192, 4, 1024) ? 0 : 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--rank8-fused-target-8") == 0) {
    return test_rank8_project_hadamard_fusion(128, 8192, 8, 1024) ? 0 : 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--rank8-fused-warp-sweep") == 0) {
    return test_rank8_project_hadamard_fusion(128, 8192, 0, 1024) ? 0 : 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--rank8-prefill-cta-2048") == 0) {
    return test_rank8_project_hadamard_fusion(960, 2048, 4, 1024) ? 0 : 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--rank8-prefill-cta-8192") == 0) {
    return test_rank8_project_hadamard_fusion(960, 8192, 4, 1024) ? 0 : 1;
  }
  for (int bits : {4, 5, 6, 7}) {
    for (int stage : {1, 2, 3, 4}) {
      for (int rows : {1, 17, 31, 32, 64, 128, 256}) {
        for (int threads : {64, 128, 256}) {
          if (test_standard_partials(rows, bits, threads, stage) != 0) return 1;
        }
        for (int groups : {2, 4, 8}) {
          if (rows % (16 * groups) != 0 || (groups == 8 && stage == 4)) continue;
          if (test_standard_partials(rows, bits, 128, stage, groups) != 0) return 1;
        }
      }
    }
  }
  // Exercise each restricted row-group-16 stage at a legal Qwen geometry.
  // These remain synthetic kernel/layout checks, not model-quality evidence.
  for (int bits : {4, 5, 6, 7}) {
    for (int stage : {2, 3}) {
      if (test_standard_partials(1024, bits, 128, stage, 16, 5120, 10240) != 0) return 1;
    }
  }
  if (test_standard_partials(1024, 5, 128, 3, 16, 5120, 1024) != 0) return 1;
  if (test_standard_partials(4096, 4, 128, 4, 16, 5120, 6144) != 0) return 1;
  return test_grouped_launch_plans() ? 0 : 1;
}
