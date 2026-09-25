// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QVQ_HADAMARD_INPUT_RAW_ABI_VERSION 2u
#define QVQ_HADAMARD_INPUT_SWIGLU_RAW_ABI_VERSION 1u

// Framework-neutral N2048/N8192 FP16 input preconditioner. All pointers are
// caller-owned device buffers, including the MxN FP16 workspace. The
// workspace may alias output exactly: the high kernel loads all 8/32 inputs
// owned by each thread before writing its 8/32 outputs. No host allocation,
// synchronization, autotuning, or hidden stream is permitted.
typedef struct {
  uint32_t abi_version;
  uint32_t struct_bytes;
  uint32_t rows;
  uint32_t width;
} QvqHadamardInputRawConfig;

uint32_t qvq_hadamard_input_raw_abi_version(void);
uint64_t qvq_hadamard_input_raw_workspace_bytes(
    const QvqHadamardInputRawConfig* config);
int qvq_hadamard_input_raw_launch(
    const void* input_f16,
    const void* pre_scale_f16,
    void* output_f16,
    void* workspace_f16,
    uint64_t workspace_bytes,
    const QvqHadamardInputRawConfig* config,
    void* cuda_stream,
    char* error,
    uint64_t error_capacity);

// Fuse FP16 SwiGLU and the input preconditioner for the M960 N8192 down
// projection. The gate/up tensors and all output/workspace buffers are owned
// by the caller and remain valid through graph replay.
uint32_t qvq_hadamard_input_swiglu_raw_abi_version(void);
int qvq_hadamard_input_swiglu_raw_launch(
    const void* gate_f16,
    const void* up_f16,
    const void* pre_scale_f16,
    void* output_f16,
    const QvqHadamardInputRawConfig* config,
    void* cuda_stream,
    char* error,
    uint64_t error_capacity);

#ifdef __cplusplus
}
#endif
