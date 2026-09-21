// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QVQ_WGMMA_RAW_ABI_VERSION 2u

// Framework-neutral SM90 P32 core. All pointers are device pointers owned by
// the caller; stream is a cudaStream_t represented as void*. Workspace is FP32
// storage. Algorithm 1 requires the size returned by the workspace query;
// algorithm 2 writes its unsplit result directly and requires no workspace.
typedef struct {
  uint32_t abi_version, struct_bytes;
  uint32_t m, k, n, transition_bits;
  uint32_t split_count;
  uint32_t algorithm;  // 1 = ordered M16, 2 = direct M64 row-reuse.
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

#ifdef __cplusplus
}
#endif
