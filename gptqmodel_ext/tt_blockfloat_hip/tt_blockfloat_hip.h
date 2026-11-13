// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Host-visible contract for the Tenstorrent block-float HIP emulation kernels.
//
// The storage layout is the one GPT-QModel already writes for
// ``bfloat8_b`` / ``bfloat4_b``:
//
//   codes      packed uint8, two nibbles per byte for bf4 (low nibble first),
//              one byte per value for bf8.  Code range is [0, 2*(2*levels-1)].
//   exponents  int8, one per 16 consecutive values along K.
//   value      (code - (2*levels-1)) / levels * 2**exponent
//
// ``levels`` is 4 for bf4 (2 stored mantissa bits) and 64 for bf8 (6 stored
// mantissa bits), matching ``gptqmodel.utils.tt_blockfloat_quant``.
//
// This header is deliberately torch-free: the same device translation unit is
// linked by the torch JIT extension and by the standalone hipcc self-test.

#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

// Output/activation element types.
enum ttbf_dtype {
    TTBF_DTYPE_F32 = 0,
    TTBF_DTYPE_F16 = 1,
    TTBF_DTYPE_BF16 = 2,
};

extern "C" {

// Expand packed codes plus per-block exponents into a dense [rows, k] tensor.
// ``out`` is ``rows * k`` elements of ``dtype``.
hipError_t ttbf_dequant(const uint8_t* codes, const int8_t* exponents, int bits,
                        int64_t rows, int64_t k, void* out, int dtype,
                        hipStream_t stream);

// Quantise a float32 [rows, k] matrix onto the block-float grid.  This mirrors
// ``quantize_blockfloat`` including the three-candidate exponent search.
// ``codes`` is the packed output (rows*k/2 bytes for bf4, rows*k for bf8).
hipError_t ttbf_quantize(const float* x, int bits, int64_t rows, int64_t k,
                         uint8_t* codes, int8_t* exponents, hipStream_t stream);

// Gather ``tokens`` rows selected by ``ids`` and dequantise only those rows.
// ``out`` is ``tokens * k`` elements of ``dtype``; out-of-range ids yield zero.
hipError_t ttbf_embedding(const int64_t* ids, const uint8_t* codes,
                          const int8_t* exponents, int bits, int64_t tokens,
                          int64_t rows, int64_t k, void* out, int dtype,
                          hipStream_t stream);

// Split an already-on-grid float32 matrix into codes and exponents.  Bit-exact
// inverse of the dequantiser.  ``failed`` is set to 1 when any block is not
// exactly representable, mirroring ``decompose_blockfloat``'s refusal.
hipError_t ttbf_decompose(const float* x, int bits, int64_t rows, int64_t k,
                          uint8_t* codes, int8_t* exponents, int32_t* failed,
                          hipStream_t stream);

// Fused dequant + GEMM: ``out[m, n] = x[m, k] @ dequant(codes)[n, k]^T + bias``.
// ``x``, ``bias`` and ``out`` share ``dtype``.  No dense weight is materialised
// outside shared memory.  ``bias`` may be null.
hipError_t ttbf_linear(const void* x, int dtype,
                       const uint8_t* codes, const int8_t* exponents, int bits,
                       int64_t m, int64_t n, int64_t k,
                       const void* bias, void* out, hipStream_t stream);

// Writes a sentinel so callers can prove device code actually ran.
hipError_t ttbf_selftest(int32_t* out, hipStream_t stream);

}  // extern "C"
