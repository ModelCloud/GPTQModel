// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#ifndef QVQ_WINDOW_ABI_H
#define QVQ_WINDOW_ABI_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// All tensors are contiguous CUDA buffers. x/SU/SV/bias/A/B/levels/y are
// FP16; window is int32; banks is one unpacked uint8 selector per 16x16 tile.
// The caller validates the unified artifact's cryptographic binding at load.
typedef struct { void* data; uint64_t bytes; } QvqWindowBuffer;
typedef struct {
  uint32_t abi_version, struct_bytes;
  uint32_t m, k, n, transition_bits, bank_alt_id;
  uint32_t algorithm; // 1 = existing Hopper M16, 2 = existing tuned Hopper
  uint32_t block_m, block_n, block_k, warp_groups, pipeline_stages, split_k;
  uint32_t min_m, max_m, input_hadamard, output_hadamard, rank8_enabled;
} QvqP32WindowConfig;
// Returns zero on successful asynchronous submission. Errors are copied into
// caller-owned storage, never thrown across the C boundary. No Python runtime
// is used. Required Torch operator libraries must be loaded before calling.
// This initial reference ABI allocates temporaries and rejects CUDA capture.
int qvq_p32_window_linear(
    QvqWindowBuffer x, QvqWindowBuffer window, QvqWindowBuffer banks,
    QvqWindowBuffer levels, QvqWindowBuffer su, QvqWindowBuffer sv,
    QvqWindowBuffer bias, QvqWindowBuffer rank8_a, QvqWindowBuffer rank8_b,
    QvqWindowBuffer y, const QvqP32WindowConfig* config, void* cuda_stream,
    char* error, uint64_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
