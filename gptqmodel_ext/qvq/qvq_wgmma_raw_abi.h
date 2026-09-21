// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>
#include "p32/qvq_p32_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QVQ_WGMMA_RAW_ABI_VERSION 3u

// Framework-neutral SM90 P32 core. All pointers are device pointers owned by
// the caller; stream is a cudaStream_t represented as void*. Workspace is FP32
// storage. Algorithm 1 requires the size returned by the workspace query;
// algorithms 2/3/4 write their unsplit result directly and require no workspace.
typedef struct {
  uint32_t abi_version, struct_bytes;
  uint32_t m, k, n, transition_bits;
  uint32_t split_count;
  uint32_t algorithm;  // 1 = ordered M16, 2 = direct M64, 3 = direct M128,
                       // 4 = direct M128 equal-width grouped gate/up.
  uint32_t block_m, block_n;
} QvqP32WgmmaRawConfig;

uint32_t qvq_p32_wgmma_raw_abi_version(void);
uint64_t qvq_p32_wgmma_raw_workspace_bytes(
    const QvqP32WgmmaRawConfig* config);
int qvq_p32_wgmma_raw_launch(
    const void* activation_f16,
    const void* continuous_window_i32,
    const void* bank_ids_u8,
    const void* levels_f16,
    const void* bank_alt_id_u8,
    void* output_f32,
    void* workspace_f32,
    uint64_t workspace_bytes,
    const QvqP32WgmmaRawConfig* config,
    void* cuda_stream,
    char* error,
    uint64_t error_capacity);

// Describe the direct M64/M128 launch as compiler-owned CUDA graph nodes.
// Host-value arguments point into `plan` and remain valid until the plan is
// rebuilt or destroyed. This avoids stream recapture on every graph replay.
int qvq_p32_wgmma_raw_launch_plan(
    const void* activation_f16,
    const void* continuous_window_i32,
    const void* bank_ids_u8,
    const void* levels_f16,
    const void* bank_alt_id_u8,
    void* output_f32,
    const QvqP32WgmmaRawConfig* config,
    struct qvq_p32_launch_plan* plan,
    char* error,
    uint64_t error_capacity);

#ifdef __cplusplus
}
#endif
