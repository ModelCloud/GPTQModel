// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Host-visible contract for the Tenstorrent block-float CUDA emulation kernels.
//
// The dtypes are ``bf4_b`` and ``bf8_b``; ``hw/dtypes.md`` section 8 is the
// single definition of their bit layout, and ``tt_blockfloat_quant`` implements
// the quantiser.  In short: one shared power-of-two exponent per 16 consecutive
// values along K (``exponents``, int8), plus packed sign/mantissa codes
// (``codes``, uint8, two per byte for ``bf4_b`` with the low nibble first and
// one per value for ``bf8_b``).
//
// Same layout and semantics as ``tt_blockfloat_hip.h``; only the runtime types
// differ.  Keeping the two headers identical is deliberate: a caller selects
// the device extension and everything above this line is shared.
//
// This header is deliberately torch-free: the same device translation unit is
// linked by the torch JIT extension and by the standalone nvcc self-test.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

// Output/activation element types.
enum blockfloat_dtype {
    BLOCKFLOAT_DTYPE_F32 = 0,
    BLOCKFLOAT_DTYPE_F16 = 1,
    BLOCKFLOAT_DTYPE_BF16 = 2,
};

extern "C" {

// Expand packed codes plus per-block exponents into a dense [rows, k] tensor.
// ``out`` is ``rows * k`` elements of ``dtype``.
cudaError_t blockfloat_dequant(const uint8_t* codes, const int8_t* exponents, int bits,
                               int64_t rows, int64_t k, void* out, int dtype,
                               cudaStream_t stream);

// Quantise a float32 [rows, k] matrix onto the block-float grid.  This mirrors
// ``quantize_blockfloat`` including the three-candidate exponent search.
// ``codes`` is the packed output (rows*k/2 bytes for bf4_b, rows*k for bf8_b).
cudaError_t blockfloat_quantize(const float* x, int bits, int64_t rows, int64_t k,
                                uint8_t* codes, int8_t* exponents, cudaStream_t stream);

// Gather ``tokens`` rows selected by ``ids`` and dequantise only those rows.
// ``out`` is ``tokens * k`` elements of ``dtype``; out-of-range ids yield zero.
cudaError_t blockfloat_embedding(const int64_t* ids, const uint8_t* codes,
                                 const int8_t* exponents, int bits, int64_t tokens,
                                 int64_t rows, int64_t k, void* out, int dtype,
                                 cudaStream_t stream);

// Split an already-on-grid float32 matrix into codes and exponents.  Bit-exact
// inverse of the dequantiser.  ``failed`` is set to 1 when any block is not
// exactly representable, mirroring ``decompose_blockfloat``'s refusal.
cudaError_t blockfloat_decompose(const float* x, int bits, int64_t rows, int64_t k,
                                 uint8_t* codes, int8_t* exponents, int32_t* failed,
                                 cudaStream_t stream);

// GEMM over the packed weights: ``out[m, n] = x[m, k] @ dequant(codes)[n, k]^T
// + bias``.  ``x`` and ``out`` share ``dtype``; ``bias`` is float32 (the
// checkpoint's dense bias precision, added to the float32 accumulator before
// the single output cast) and may be null.
cudaError_t blockfloat_linear(const void* x, int dtype,
                              const uint8_t* codes, const int8_t* exponents, int bits,
                              int64_t m, int64_t n, int64_t k,
                              const void* bias, void* out, cudaStream_t stream);

// Writes a sentinel so callers can prove device code actually ran.
cudaError_t blockfloat_selftest(int32_t* out, cudaStream_t stream);

}  // extern "C"
