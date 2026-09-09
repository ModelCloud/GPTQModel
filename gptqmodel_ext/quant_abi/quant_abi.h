// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Versioned, contiguous, device-resident tensors. Null data denotes an absent
// optional tensor; all other fields must then be zero. No ownership transfer.
enum QvqQuantDtype { QVQ_F16 = 1, QVQ_BF16 = 2, QVQ_F32 = 3, QVQ_I32 = 4, QVQ_U8 = 5 };
enum QvqQuantOp {
  QVQ_MACHETE_MM = 1, QVQ_MACHETE_PREPACK = 2,
  QVQ_SWORDFISH_DECODE = 3, QVQ_SWORDFISH_PREFILL = 4,
  QVQ_SWORDFISH_PREPACK = 5, QVQ_SWORDFISH_DEQUANT = 6
};
typedef struct {
  void* data;
  uint64_t bytes;
  int64_t shape[4];
  uint32_t rank, dtype;
} QvqQuantTensor;
typedef struct {
  uint32_t struct_bytes, abi_version, operation;
  int32_t device;
  int64_t weight_type, group_size, k, n;
  uint32_t activation_dtype, output_dtype, num_bits, transpose;
  // Swordfish decode: mode 0 deterministic, 1 atomic, 2 Stream-K.
  // threads=128; stages is a checked compiled constraint, NOT a hint.
  int32_t mode, m_tiles, split_k, ctas, cta_quad, threads, stages;
  // Swordfish prefill: exact compiled tile N128/256 and M chunk (multiple128).
  int32_t tile_n, chunk_m;
  uint32_t reserved;
  // Required exact compiled Machete MM schedule. Copied during prepare.
  const char* schedule;
} QvqQuantConfig;
typedef struct QvqQuantPlan QvqQuantPlan;
uint32_t qvq_quant_abi_version(void);
// Inputs by operation:
// Machete MM: A, packed B, group scales?, group zeros?, channel scales?, token scales?
// Machete prepack: B, group scales? (only its dtype is consulted)
// Swordfish decode/prefill: A, packed B, scales, prescaled zeros?
// Swordfish prepack: B, permutation?
// Swordfish dequant: packed B, scales, prescaled zeros?
// Output is supplied separately; exact dtype/shape checked against native result.
// Prepare warms kernels and captures a retained allocation pool; outside capture
// only, on a nondefault caller stream. Kernel op libraries must already be loaded.
int qvq_quant_prepare(const QvqQuantConfig*, const QvqQuantTensor* inputs,
                     uint32_t input_count, QvqQuantTensor output, void* stream,
                     QvqQuantPlan**, char* error, uint64_t error_capacity);
// Immutable pointer-bound plan. Same device and stream as prepare. Launch may
// insert a child graph into caller capture. Keep buffers/library/stream/plan alive
// until all launches AND all enclosing graphs have been destroyed. No concurrency
// with destroy. ZML command-buffer compatibility remains unvalidated.
int qvq_quant_launch(QvqQuantPlan*, void* stream, char* error, uint64_t error_capacity);
int qvq_quant_destroy(QvqQuantPlan*, char* error, uint64_t error_capacity);
// Query Machete's compiled schedule set (not shape feasibility). Optional dtype
// entries use zero. Newline-delimited, NUL-terminated; required includes NUL.
// Passing buffer=NULL/capacity=0 is a size query. Insufficient capacity errors.
int qvq_machete_schedules(uint32_t activation_dtype, int64_t weight_type,
                         const uint32_t optional_dtypes[5], char* buffer,
                         uint64_t capacity, uint64_t* required,
                         char* error, uint64_t error_capacity);
#ifdef __cplusplus
}
#endif
