// ---------------------------------------------------------------------------
// NF12 / NF12-AD kernel computation.
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original implementation: https://github.com/dphnAI/nf12
// Copyright (c) 2026 AlpinDale and the nf12 contributors.
// Licensed under the Apache License, Version 2.0.
//
// Vendored into GPT-QModel (https://github.com/ModelCloud/GPTQModel) and
// modified for the NF12-AD (AlpinDale) method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
/* SPDX-License-Identifier: Apache-2.0
 * NF12 public C API. See SPEC.md section 8.1.
 *
 * Tensor and workspace pointers refer to device memory. Weight descriptors and
 * analysis outputs are host-resident. No function allocates device memory.
 * nf12_encode_analyze and nf12_encode synchronize the supplied stream.
 */
#ifndef NF12_NF12_H_
#define NF12_NF12_H_

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NF12_VERSION_MAJOR 1
#define NF12_VERSION_MINOR 0

typedef enum nf12_status {
  NF12_OK = 0,
  NF12_ERROR_INVALID_ARGUMENT = 1, /* shape/alignment/pointer violates SPEC.md */
  NF12_ERROR_UNSUPPORTED = 2,      /* unsupported device or kernel configuration */
  NF12_ERROR_INSUFFICIENT_WORKSPACE = 3,
  NF12_ERROR_CUDA = 4              /* see nf12_last_cuda_error() */
} nf12_status_t;

/* Non-owning encoded weight view. */
typedef struct nf12_weight {
  const void* packed;      /* N*K*3/2 bytes, 16-byte aligned, SPEC.md 4.3 */
  const void* escapes;     /* num_escapes * 16 bytes, 16-byte aligned (may be NULL if num_escapes == 0) */
  uint32_t num_escapes;
  int64_t n;               /* rows (out_features), multiple of 16 */
  int64_t k;               /* cols (in_features), multiple of 64 */
  uint8_t exp_base;        /* SPEC.md section 2, 0..120 */
} nf12_weight_t;

const char* nf12_status_string(nf12_status_t status);
cudaError_t nf12_last_cuda_error(void); /* thread-local, set when NF12_ERROR_CUDA is returned */

size_t nf12_packed_bytes(int64_t n, int64_t k);
size_t nf12_bf16t_bytes(int64_t n, int64_t k);

/* Analyze selects exp_base and counts escapes before allocation.
 * exp_base_out and num_escapes_out are host pointers.
 * Analyze and encode synchronize stream and cannot be graph-captured.
 * Both require nf12_encode_workspace_bytes(n, k) bytes of workspace. */
size_t nf12_encode_workspace_bytes(int64_t n, int64_t k);
nf12_status_t nf12_encode_analyze(const void* w_bf16, int64_t n, int64_t k, int64_t ldw, void* workspace,
                                  size_t workspace_bytes, uint8_t* exp_base_out, uint32_t* num_escapes_out,
                                  cudaStream_t stream);
/* Packing is deterministic for a fixed exp_base. Capacity is checked before writing outputs. */
nf12_status_t nf12_encode(const void* w_bf16, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base, void* packed_out,
                          void* escapes_out, uint32_t escapes_capacity, void* workspace, size_t workspace_bytes,
                          cudaStream_t stream);

/* Row-major BF16 output [n, k], with row stride ldo in elements. */
nf12_status_t nf12_decode(const nf12_weight_t* w, void* out_bf16, int64_t ldo, cudaStream_t stream);

/* y = x @ W.T + bias, with BF16 input/output and FP32 accumulation.
 * x is [m, k], y is [m, n], and ldx/ldy are row strides in elements. bias may be NULL.
 * Workspace must hold nf12_linear_workspace_bytes(...) bytes, which may be zero.
 * Contents need no initialization and are overwritten. Reuse across shapes and
 * graph replays is safe only when calls cannot overlap. */
size_t nf12_linear_workspace_bytes(int64_t m, int64_t n, int64_t k, int device);
nf12_status_t nf12_linear(const void* x_bf16, int64_t m, int64_t ldx, const nf12_weight_t* w, const void* bias_bf16,
                          void* y_bf16, int64_t ldy, void* workspace, size_t workspace_bytes, cudaStream_t stream);

/* Heuristic N-slice height for decode + external BF16 GEMM (SPEC.md 6, 8.1).
 * Returns a multiple of 16, or 0 to retain nf12_linear or reject invalid arguments.
 * For n0 % 16 == 0, slice [n0, n0 + rows) has the view
 * {packed + n0 * k * 3 / 2, escapes, num_escapes, rows, k, exp_base}.
 * Escape indices remain absolute within the original table. */
int64_t nf12_linear_decode_gemm_rows(int64_t m, int64_t n, int64_t k, int device);

/* Uncompressed bf16t layout for exactness comparisons (SPEC.md 4.4). */
nf12_status_t nf12_pack_bf16t(const void* w_bf16, int64_t n, int64_t k, int64_t ldw, void* out, cudaStream_t stream);
nf12_status_t nf12_linear_bf16t(const void* x_bf16, int64_t m, int64_t ldx, const void* w_bf16t, int64_t n, int64_t k,
                                const void* bias_bf16, void* y_bf16, int64_t ldy, void* workspace,
                                size_t workspace_bytes, cudaStream_t stream);

/* MoE operations support graph capture without host synchronization (SPEC.md 9). */
typedef struct nf12_moe_weight {
  const void* packed;            /* [E, n, k*3/2] bytes, expert-major, 16-byte aligned */
  const void* escapes;           /* [sum, 8] int16, expert-major, NULL allowed if empty */
  const int64_t* escape_offsets; /* device [E + 1] */
  const uint8_t* exp_base;       /* device [E] */
  int64_t num_experts;
  int64_t n;                     /* rows per expert, multiple of 16 */
  int64_t k;                     /* cols per expert, multiple of 64 */
} nf12_moe_weight_t;

/* Stable sort of assignment ids (t * topk + j) by expert. Ids outside [0, num_experts) are dropped. */
size_t nf12_moe_align_workspace_bytes(int64_t num_tokens, int64_t topk, int64_t num_experts);
nf12_status_t nf12_moe_align(const int32_t* topk_ids, int64_t num_tokens, int64_t topk, int64_t num_experts,
                             int32_t* sorted_out, int64_t* expert_offsets_out, void* workspace,
                             size_t workspace_bytes, cudaStream_t stream);

/* y[out(p)] = x[in(p)] . W_e^T (* row_scale[out(p)]) for each valid sorted position p of expert e.
 * in(p) = gather ? gather[p] / gather_div : p. out(p) = scatter ? scatter[p] : p.
 * gather, scatter and row_scale may be NULL. max_assignments bounds sorted positions and the grid. */
size_t nf12_moe_linear_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                       int device);
nf12_status_t nf12_moe_linear(const void* x_bf16, int64_t ldx, const int32_t* gather, int64_t gather_div,
                              const int32_t* scatter, const int64_t* expert_offsets, int64_t max_assignments,
                              const nf12_moe_weight_t* w, const float* row_scale, void* y_bf16, int64_t ldy,
                              void* workspace, size_t workspace_bytes, cudaStream_t stream);
/* bf16t twin: w_bf16t is [E, n, 2k] bytes (per-expert SPEC.md 4.4 layout). */
nf12_status_t nf12_moe_linear_bf16t(const void* x_bf16, int64_t ldx, const int32_t* gather, int64_t gather_div,
                                    const int32_t* scatter, const int64_t* expert_offsets, int64_t max_assignments,
                                    const void* w_bf16t, int64_t num_experts, int64_t n, int64_t k,
                                    const float* row_scale, void* y_bf16, int64_t ldy, void* workspace,
                                    size_t workspace_bytes, cudaStream_t stream);

/* out[t] = sum_j y[t * topk + j] (FP32, increasing j), BF16 out. y rows have stride ldy, out rows ldo. */
nf12_status_t nf12_moe_combine(const void* y_bf16, int64_t ldy, int64_t num_tokens, int64_t topk, int64_t n,
                               void* out_bf16, int64_t ldo, cudaStream_t stream);

/* For BF16 h = [gate | up], out[r, i] = bf16(s * u), where g = h[r, i],
 * u = h[r, inter + i] and s = g / (1 + expf(-g)). Arithmetic is FP32 (SPEC.md 9.2).
 * h and out have row strides ldh and ldo in elements. */
nf12_status_t nf12_moe_silu_mul(const void* h_bf16, int64_t ldh, int64_t rows, int64_t inter, void* out_bf16,
                                int64_t ldo, cudaStream_t stream);

/* Decode expert rows [row0, row0 + rows), with row0/rows multiples of 16 (SPEC.md 9.4).
 * Expert e's decoded row r starts at out + (e * rows + r) * ldo elements.
 * Non-null expert_offsets skip inactive experts and leave their output untouched.
 * Activity follows nf12_moe_linear's offsets, bounded by max_assignments.
 * Supports graph capture without host synchronization. */
nf12_status_t nf12_moe_decode(const nf12_moe_weight_t* w, const int64_t* expert_offsets, int64_t max_assignments,
                              int64_t row0, int64_t rows, void* out_bf16, int64_t ldo, cudaStream_t stream);
/* Heuristic N-slice height for decode + external grouped GEMM without scatter or row_scale.
 * Returns 0 to retain nf12_moe_linear. */
int64_t nf12_moe_linear_decode_gemm_rows(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                         int device);
/* group_ends (device int32 [E]) contains expert ends using nf12_moe_linear's clamped,
 * monotone offsets. Expert e owns [group_ends[e-1], group_ends[e]), with group_ends[-1] = 0.
 * With gather, x_out[p] copies x[gather[p] / gather_div] for p < group_ends[E-1].
 * x_out has row stride ldo in elements. Other output rows remain untouched. */
nf12_status_t nf12_moe_grouped_operands(const void* x_bf16, int64_t ldx, const int32_t* gather, int64_t gather_div,
                                        const int64_t* expert_offsets, int64_t max_assignments, int64_t num_experts,
                                        int64_t k, void* x_out_bf16, int64_t ldo, int32_t* group_ends,
                                        cudaStream_t stream);

/* Returns out[i, :] = W[ids[i], :] with row stride ldo in elements (SPEC.md 10).
 * IDs outside [0, w->n) produce +0.0 rows. Supports graph capture without host synchronization. */
nf12_status_t nf12_embedding(const nf12_weight_t* w, const int32_t* ids, int64_t num_ids, void* out_bf16, int64_t ldo,
                             cudaStream_t stream);
nf12_status_t nf12_embedding_i64(const nf12_weight_t* w, const int64_t* ids, int64_t num_ids, void* out_bf16,
                                 int64_t ldo, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
