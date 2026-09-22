// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_wgmma_raw_abi.h"

#include <cstdio>
#include <cstring>

#define QVQ_DECLARE_SHARD(BITS)                                                \
  extern "C" int qvq_p32_wgmma_raw_launch_w##BITS(                           \
      const void*, const void*, const void*, const void*, const void*, void*,  \
      void*, uint64_t, const QvqP32WgmmaRawConfig*, void*, char*, uint64_t);   \
  extern "C" int qvq_p32_wgmma_raw_launch_plan_w##BITS(                      \
      const void*, const void*, const void*, const void*, const void*, void*,  \
      const QvqP32WgmmaRawConfig*, qvq_p32_launch_plan*, char*, uint64_t)

QVQ_DECLARE_SHARD(4);
QVQ_DECLARE_SHARD(5);
QVQ_DECLARE_SHARD(6);
QVQ_DECLARE_SHARD(7);
#undef QVQ_DECLARE_SHARD

namespace {

constexpr uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

int invalid_bits(char* error, uint64_t capacity) {
  if (error != nullptr && capacity != 0) {
    std::snprintf(error, static_cast<size_t>(capacity),
                  "QVQ WGMMA transition_bits must be in [4,7]");
  }
  return -1;
}

}  // namespace

extern "C" uint32_t qvq_p32_wgmma_raw_abi_version(void) {
  return QVQ_WGMMA_RAW_ABI_VERSION;
}

extern "C" uint64_t qvq_p32_wgmma_raw_workspace_bytes(
    const QvqP32WgmmaRawConfig* c) {
  if (c == nullptr) return 0;
  if (c->algorithm == 2 || c->algorithm == 3 || c->algorithm == 4) return 0;
  const uint64_t padded_input = align_up(16ull * c->k * sizeof(uint16_t), 256);
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
  if (c == nullptr) return invalid_bits(error, error_capacity);
#define QVQ_CALL_SHARD(BITS) qvq_p32_wgmma_raw_launch_w##BITS(                 \
    activation, window, bank_ids, levels, bank_alt_id, output, workspace,      \
    workspace_bytes, c, cuda_stream, error, error_capacity)
  switch (c->transition_bits) {
    case 4: return QVQ_CALL_SHARD(4);
    case 5: return QVQ_CALL_SHARD(5);
    case 6: return QVQ_CALL_SHARD(6);
    case 7: return QVQ_CALL_SHARD(7);
    default: return invalid_bits(error, error_capacity);
  }
#undef QVQ_CALL_SHARD
}

extern "C" int qvq_p32_wgmma_raw_launch_plan(
    const void* activation, const void* window, const void* bank_ids,
    const void* levels, const void* bank_alt_id, void* output,
    const QvqP32WgmmaRawConfig* c, qvq_p32_launch_plan* plan,
    char* error, uint64_t error_capacity) {
  if (c == nullptr) return invalid_bits(error, error_capacity);
#define QVQ_CALL_PLAN_SHARD(BITS) qvq_p32_wgmma_raw_launch_plan_w##BITS(       \
    activation, window, bank_ids, levels, bank_alt_id, output, c, plan,       \
    error, error_capacity)
  switch (c->transition_bits) {
    case 4: return QVQ_CALL_PLAN_SHARD(4);
    case 5: return QVQ_CALL_PLAN_SHARD(5);
    case 6: return QVQ_CALL_PLAN_SHARD(6);
    case 7: return QVQ_CALL_PLAN_SHARD(7);
    default: return invalid_bits(error, error_capacity);
  }
#undef QVQ_CALL_PLAN_SHARD
}
