// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// Tenstorrent bf8_b / bf4_b execution on AMD ROCm GPUs.
//
// Tenstorrent stores these formats as blocks of 16 values that share one
// exponent; the Tensix unpack stage expands the packed codes in the data path,
// so a weight never exists as a dense matrix in DRAM.  This file reproduces
// that execution model on a GPU:
//
//   * the packed code stream and int8 exponents are consumed as-is, so a
//     checkpoint written for a Tenstorrent device is read without conversion;
//   * bf4 nibbles and bf8 bytes are unpacked in registers;
//   * dequantisation is fused into the GEMM, with the expanded weights living
//     only in shared memory for the duration of one K tile;
//   * the dequantised values are exact float32 grid points, so the emulation
//     is numerically identical to ``recompose_blockfloat`` followed by an
//     fp32 GEMM.
//
// Two GEMM shapes are provided because the two regimes have different
// bottlenecks.  Decode (M == 1) is bandwidth-bound: one CTA owns 32 output
// features, stages the activation row in shared memory once, and streams each
// packed weight row exactly once.  Prefill (M > 1) is compute-bound: a
// 32x64x32 register-tiled GEMM amortises the unpack of one weight tile over
// 32 token rows.
//
// No torch types cross this translation unit; it is compiled both by the JIT
// extension and by the standalone hipcc self-test.
// ---------------------------------------------------------------------------

#include "tt_blockfloat_hip.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>

namespace {

// ---------------------------------------------------------------------------
// Element helpers
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float ttbf_load(const T* p);

template <>
__device__ __forceinline__ float ttbf_load<float>(const float* p) {
    return *p;
}

template <>
__device__ __forceinline__ float ttbf_load<__half>(const __half* p) {
    return __half2float(*p);
}

template <>
__device__ __forceinline__ float ttbf_load<__hip_bfloat16>(const __hip_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename T>
__device__ __forceinline__ T ttbf_cast(float v);

template <>
__device__ __forceinline__ float ttbf_cast<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ __half ttbf_cast<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__
__hip_bfloat16 ttbf_cast<__hip_bfloat16>(float v) {
    // AMD's bf16 constructor lowers to cvt_pk_bf16, i.e. round-to-nearest-even.
    return __float2bfloat16(v);
}

// Pack two floats into one 32-bit word of the destination type.
template <typename T>
__device__ __forceinline__
uint32_t ttbf_pack2(float a, float b);

template <>
__device__ __forceinline__
uint32_t ttbf_pack2<__half>(float a, float b) {
    union {
        __half2 h;
        uint32_t u;
    } cv;
    cv.h = __floats2half2_rn(a, b);
    return cv.u;
}

template <>
__device__
__forceinline__ uint32_t ttbf_pack2<__hip_bfloat16>(float a, float b) {
    union {
        __hip_bfloat162 h;
        uint32_t u;
    } cv;
    cv.h = __float22bfloat162_rn(make_float2(a, b));
    return cv.u;
}

// Store eight consecutive values, vectorised when the destination allows it.
template <typename T>
__device__ __forceinline__
void ttbf_store8(T* p, const float* v) {
    if constexpr (sizeof(T) == 4) {
        const float4 a = make_float4(v[0], v[1], v[2], v[3]);
        const float4 b = make_float4(v[4], v[5], v[6], v[7]);
        *reinterpret_cast<float4*>(p) = a;
        *reinterpret_cast<float4*>(p + 4) = b;
    } else {
        uint4 u;
        u.x = ttbf_pack2<T>(v[0], v[1]);
        u.y = ttbf_pack2<T>(v[2], v[3]);
        u.z = ttbf_pack2<T>(v[4], v[5]);
        u.w = ttbf_pack2<T>(v[6], v[7]);
        *reinterpret_cast<uint4*>(p) = u;
    }
}

// ---------------------------------------------------------------------------
// Grid constants (keep in sync with tt_blockfloat_quant.py)
// ---------------------------------------------------------------------------

template <int BITS>
struct Grid {
    static_assert(BITS == 4 || BITS == 8, "only bf4_b / bf8_b exist");
    // Stored mantissa bits: 2 for bf4, 6 for bf8.  One implicit leading bit.
    static constexpr int kMantBits = BITS - 2;
    static constexpr int kLevels = 1 << kMantBits;
    static constexpr int kOffset = 2 * kLevels - 1;
    static constexpr float kInvLevels = 1.0f / static_cast<float>(kLevels);
};

// Decode one packed code, given a pre-computed ``2**exponent``.
__device__ __forceinline__
float ttbf_value(int code, int offset, float inv_levels, float scale) {
    return static_cast<float>(code - offset) * inv_levels * scale;
}

// Exact ``2**e`` for the integer block exponent.  ``exp2f`` lowers to a
// software ``ldexp`` on this target; the bit pattern is one add and one shift.
// Exponents below the normal range (a block whose maximum is FLT_MIN) fall
// back to ``ldexpf`` so subnormals stay exact.
__device__ __forceinline__
float ttbf_exp2i(int e) {
    if (e >= -126) {
        return __int_as_float(static_cast<int>((e + 127) << 23));
    }
    return ldexpf(1.0f, e);
}

// ---------------------------------------------------------------------------
// Dequantise packed codes -> dense [rows, k]
// ---------------------------------------------------------------------------

template <typename T, int BITS>
__global__ void dequant_kernel(const uint8_t* __restrict__ codes,
                               const int8_t* __restrict__ exponents,
                               int64_t rows, int64_t k,
                               T* __restrict__ out) {
    using G = Grid<BITS>;
    const int64_t groups_per_row = k >> 3;
    const int64_t total = rows * groups_per_row;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         g < total; g += stride) {
        const int64_t row = g / groups_per_row;
        const int64_t col = (g - row * groups_per_row) << 3;
        const int64_t block = col >> 4;
        const float scale = ttbf_exp2i(exponents[row * (k >> 4) + block]);

        int code[8];
        if (BITS == 4) {
            // Eight nibbles = four bytes.  Low nibble is the lower column.
            const uint32_t word =
                *reinterpret_cast<const uint32_t*>(codes + ((row * k + col) >> 1));
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                code[i] = static_cast<int>((word >> (4 * i)) & 0xFu);
            }
        } else {
            const uint2 word =
                *reinterpret_cast<const uint2*>(codes + row * k + col);
            const uint32_t lo = word.x;
            const uint32_t hi = word.y;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                code[i] = static_cast<int>((lo >> (8 * i)) & 0xFFu);
                code[i + 4] = static_cast<int>((hi >> (8 * i)) & 0xFFu);
            }
        }

        float value[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            value[i] = ttbf_value(code[i], G::kOffset, G::kInvLevels, scale);
        }
        ttbf_store8<T>(out + row * k + col, value);
    }
}

// ---------------------------------------------------------------------------
// Embedding gather: dequantise only the rows the token ids select
// ---------------------------------------------------------------------------
//
// A block-float embedding table is a lookup table, so expanding it to a dense
// matrix and then indexing wastes both bandwidth and memory.  This mirrors the
// Tenstorrent reader: the packed rows are fetched by id and expanded directly
// into the output.

template <typename T, int BITS>
__global__ void embedding_kernel(const int64_t* __restrict__ ids,
                                 const uint8_t* __restrict__ codes,
                                 const int8_t* __restrict__ exponents,
                                 int64_t tokens, int64_t rows, int64_t k,
                                 T* __restrict__ out) {
    using G = Grid<BITS>;
    const int64_t groups_per_row = k >> 3;
    const int64_t total = tokens * groups_per_row;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         g < total; g += stride) {
        const int64_t token = g / groups_per_row;
        const int64_t col = (g - token * groups_per_row) << 3;
        const int64_t row = ids[token];
        if (row < 0 || row >= rows) {
            float zero[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f};
            ttbf_store8<T>(out + token * k + col, zero);
            continue;
        }
        const float scale =
            ttbf_exp2i(exponents[row * (k >> 4) + (col >> 4)]);
        int code[8];
        if (BITS == 4) {
            const uint32_t word =
                *reinterpret_cast<const uint32_t*>(codes + ((row * k + col) >> 1));
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                code[i] = static_cast<int>((word >> (4 * i)) & 0xFu);
            }
        } else {
            const uint2 word =
                *reinterpret_cast<const uint2*>(codes + row * k + col);
            const uint32_t lo = word.x;
            const uint32_t hi = word.y;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                code[i] = static_cast<int>((lo >> (8 * i)) & 0xFFu);
                code[i + 4] = static_cast<int>((hi >> (8 * i)) & 0xFFu);
            }
        }
        float value[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            value[i] = ttbf_value(code[i], G::kOffset, G::kInvLevels, scale);
        }
        ttbf_store8<T>(out + token * k + col, value);
    }
}

// ---------------------------------------------------------------------------
// Quantise float -> packed codes (bit-exact with quantize_blockfloat)
// ---------------------------------------------------------------------------
//
// One thread owns one 16-value block.  ``floor(log2(block_max))`` is not always
// the exponent with the lowest error, so the reference evaluates the base
// exponent and its two neighbours and keeps the smallest squared error.  Ties
// keep the earlier candidate (base, then base-1, then base+1), matching the
// strict ``<`` in the torch implementation.

template <int BITS>
__device__ __forceinline__
void ttbf_quantise_block(const float* __restrict__ xp, int* code_out,
                         int* exponent_out) {
    using G = Grid<BITS>;
    float magnitude = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        magnitude = fmaxf(magnitude, fabsf(xp[i]));
    }
    const float safe = fmaxf(magnitude, 1.17549435e-38f);  // FLT_MIN
    const int base = static_cast<int>(floorf(log2f(safe)));

    float best_error = INFINITY;
    int best_exponent = base;
    int best_code[16];

#pragma unroll
    for (int candidate = 0; candidate < 3; ++candidate) {
        const int exponent = base + (candidate == 0 ? 0 : (candidate == 1 ? -1 : 1));
        const float scale = ttbf_exp2i(exponent);
        float error = 0.0f;
        int code[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            // Divide first: the reference does ``x / scale * levels``, and a
            // reciprocal would overflow for the tiny exponent of a zero block.
            float q = rintf((xp[i] / scale) * static_cast<float>(G::kLevels));
            q = fminf(fmaxf(q, -static_cast<float>(G::kOffset)),
                      static_cast<float>(G::kOffset));
            code[i] = static_cast<int>(q);
            const float reconstructed = (q * G::kInvLevels) * scale;
            const float diff = reconstructed - xp[i];
            error = error + diff * diff;
        }
        // The base candidate is taken unconditionally, matching the reference
        // (``best = _round_with(base_exponent)`` then compare the neighbours).
        // A strict comparison against INFINITY would leave ``best_code``
        // uninitialised whenever every candidate's error overflows -- which
        // happens for inputs near FLT_MAX, where the squared error is inf.
        if (candidate == 0 || error < best_error) {
            best_error = error;
            best_exponent = exponent;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                best_code[i] = code[i];
            }
        }
    }

    *exponent_out = best_exponent;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        code_out[i] = best_code[i];
    }
}

template <int BITS>
__device__ __forceinline__
void ttbf_write_block(uint8_t* __restrict__ dst, const int* code) {
    using G = Grid<BITS>;
    if (BITS == 4) {
        uint32_t words[2] = {0u, 0u};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t lo = static_cast<uint32_t>(code[2 * j] + G::kOffset);
            const uint32_t hi = static_cast<uint32_t>(code[2 * j + 1] + G::kOffset);
            words[j >> 2] |= (lo | (hi << 4)) << (8 * (j & 3));
        }
        uint2 packed;
        packed.x = words[0];
        packed.y = words[1];
        *reinterpret_cast<uint2*>(dst) = packed;
    } else {
        uint32_t words[4] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint32_t byte = static_cast<uint32_t>(code[j] + G::kOffset);
            words[j >> 2] |= byte << (8 * (j & 3));
        }
        uint4 packed;
        packed.x = words[0];
        packed.y = words[1];
        packed.z = words[2];
        packed.w = words[3];
        *reinterpret_cast<uint4*>(dst) = packed;
    }
}

template <int BITS>
__global__ void quantize_kernel(const float* __restrict__ x,
                                int64_t rows, int64_t k,
                                uint8_t* __restrict__ codes,
                                int8_t* __restrict__ exponents) {
    const int64_t blocks_per_row = k >> 4;
    const int64_t total = rows * blocks_per_row;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         b < total; b += stride) {
        const int64_t row = b / blocks_per_row;
        const int64_t block = b - row * blocks_per_row;
        const int64_t col = block << 4;

        int code[16];
        int exponent = 0;
        ttbf_quantise_block<BITS>(x + row * k + col, code, &exponent);

        if (BITS == 4) {
            ttbf_write_block<BITS>(codes + ((row * k + col) >> 1), code);
        } else {
            ttbf_write_block<BITS>(codes + row * k + col, code);
        }
        exponents[row * blocks_per_row + block] = static_cast<int8_t>(exponent);
    }
}

// ---------------------------------------------------------------------------
// Decompose an already-on-grid float matrix into codes + exponents
// ---------------------------------------------------------------------------
//
// The reference tries shifts (0, +1, -1, +2) around floor(log2(max)) and
// accepts the first candidate that reproduces every value exactly.  No error
// search is needed because the input is already a grid point.

template <int BITS>
__global__ void decompose_kernel(const float* __restrict__ x,
                                 int64_t rows, int64_t k,
                                 uint8_t* __restrict__ codes,
                                 int8_t* __restrict__ exponents,
                                 int32_t* __restrict__ failed) {
    using G = Grid<BITS>;
    const int64_t blocks_per_row = k >> 4;
    const int64_t total = rows * blocks_per_row;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         b < total; b += stride) {
        const int64_t row = b / blocks_per_row;
        const int64_t block = b - row * blocks_per_row;
        const int64_t col = block << 4;
        const float* xp = x + row * k + col;

        float magnitude = 0.0f;
#pragma unroll
        for (int i = 0;i < 16; ++i) {  // NOLINT
            magnitude = fmaxf(magnitude, fabsf(xp[i]));
        }
        const float safe = fmaxf(magnitude, 1.17549435e-38f);
        const int base = static_cast<int>(floorf(log2f(safe)));

        int chosen_code[16] = {0};
        int chosen_exponent = base;
        bool found = false;

        for (int t = 0; t < 4 && !found; ++t) {
            const int shift = (t == 0) ? 0 : (t == 1 ? 1 : (t == 2 ? -1 : 2));
            const int exponent = base + shift;
            const float scale = ttbf_exp2i(exponent);
            int code[16];
            bool ok = true;
#pragma unroll
            for (int i = 0;i < 16; ++i) {  // NOLINT
                const float q =
                    rintf((xp[i] / scale) * static_cast<float>(G::kLevels));
                if (fabsf(q) > static_cast<float>(G::kOffset)) {
                    ok = false;
                    break;
                }
                code[i] = static_cast<int>(q);
            }
            if (!ok) {
                continue;
            }
#pragma unroll
            for (int i = 0;i < 16; ++i) {  // NOLINT
                const float reconstructed =
                    (static_cast<float>(code[i]) * G::kInvLevels) * scale;
                if (reconstructed != xp[i]) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }
            chosen_exponent = exponent;
#pragma unroll
            for (int i = 0;i < 16; ++i) {  // NOLINT
                chosen_code[i] = code[i];
            }
            found = true;
        }

        if (!found) {
            atomicExch(failed, 1);
        }

        if (BITS == 4) {
            ttbf_write_block<BITS>(codes + ((row * k + col) >> 1), chosen_code);
        } else {
            ttbf_write_block<BITS>(codes + row * k + col, chosen_code);
        }
        exponents[row * blocks_per_row + block] =
            static_cast<int8_t>(chosen_exponent);
    }
}

// ---------------------------------------------------------------------------
// Decode path: one CTA owns 32 output features, M == 1
// ---------------------------------------------------------------------------
//
// 256 threads = 8 warps.  Each warp owns four output features and each lane
// owns 16 consecutive K values, so one lane reads exactly one exponent and
// 8 (bf4) or 16 (bf8) code bytes.  The activation row is staged once per CTA
// and reused by all 32 output features.

constexpr int kGemvOutputs = 32;
// 512 threads = 16 warps x 2 rows.  The extra warps double the outstanding
// global loads per CTA, which is what keeps HBM busy on a bandwidth-bound
// decode; 4 CTAs per CU then fill all 2048 thread slots.
constexpr int kGemvThreads = 512;
constexpr int kGemvRowsPerWarp = 2;
constexpr int kGemvChunk = 4096;  // floats staged in shared memory
// Each lane owns 16 consecutive K values.  Laying those 16-value segments out
// with a 17-float stride makes the per-`i` read `s_act[segment*17 + i]` map to
// bank `(17*lane + i) % 32`, i.e. all 32 banks, instead of the two banks a
// contiguous 16-value stride produces (a 16-way conflict).
constexpr int kGemvLd = 17;

template <typename T, int BITS>
__global__ __launch_bounds__(kGemvThreads)
void gemv_kernel(const T* __restrict__ x,
                 const uint8_t* __restrict__ codes,
                 const int8_t* __restrict__ exponents,
                 const float* __restrict__ bias,
                 T* __restrict__ out,
                 int64_t n, int64_t k) {
    using G = Grid<BITS>;
    __shared__ float s_act[(kGemvChunk / 16) * kGemvLd];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t base_n = static_cast<int64_t>(blockIdx.x) * kGemvOutputs;

    int64_t row[kGemvRowsPerWarp];
    float acc[kGemvRowsPerWarp] = {};
#pragma unroll
    for (int r = 0; r < kGemvRowsPerWarp; ++r) {
        row[r] = base_n + warp * kGemvRowsPerWarp + r;
    }

    for (int64_t k0 = 0; k0 < k; k0 += kGemvChunk) {
        const int64_t chunk =
            (k - k0 < kGemvChunk) ? (k - k0) : kGemvChunk;
        for (int64_t i = threadIdx.x; i < chunk; i += blockDim.x) {
            s_act[(i >> 4) * kGemvLd + (i & 15)] = ttbf_load<T>(x + k0 + i);
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < kGemvRowsPerWarp; ++r) {
            if (row[r] >= n) {
                continue;
            }
            const uint8_t* row_codes =
                codes + (BITS == 4 ? (row[r] * k) >> 1 : row[r] * k);
            const int8_t* row_exp = exponents + row[r] * (k >> 4);

            for (int64_t kk = lane * 16; kk < chunk; kk += 32 * 16) {
                const int64_t kg = k0 + kk;
                const float scale = ttbf_exp2i(row_exp[kg >> 4]);
                int code[16];
                if (BITS == 4) {
                    const uint2 word =
                        *reinterpret_cast<const uint2*>(row_codes + (kg >> 1));
                    const uint32_t words[2] = {word.x, word.y};
#pragma unroll
                    for (int i = 0;i < 16; ++i) {  // NOLINT
                        code[i] = static_cast<int>(
                            (words[i >> 3] >> (4 * (i & 7))) & 0xFu);
                    }
                } else {
                    const uint4 word =
                        *reinterpret_cast<const uint4*>(row_codes + kg);
                    const uint32_t words[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
                    for (int i = 0;i < 16; ++i) {  // NOLINT
                        code[i] = static_cast<int>(
                            (words[i >> 2] >> (8 * (i & 3))) & 0xFFu);
                    }
                }
                const int64_t segment = kk >> 4;
                // Hoist the two constant multiplies out of the 16-value walk:
                // each term is (code - offset) * (inv_levels * scale) * act,
                // so folding the level scale into the block scale saves one
                // multiply per element on the decode path.
                const float factor = G::kInvLevels * scale;
                float partial0 = 0.0f;
                float partial1 = 0.0f;
#pragma unroll
                for (int i = 0;i < 16; ++i) {  // NOLINT
                    const float term = static_cast<float>(code[i] - G::kOffset) *
                                       factor * s_act[segment * kGemvLd + i];
                    if ((i & 1) == 0) {
                        partial0 += term;
                    } else {
                        partial1 += term;
                    }
                }
                acc[r] += partial0 + partial1;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < kGemvRowsPerWarp; ++r) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[r] += __shfl_down(acc[r], offset, 32);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kGemvRowsPerWarp; ++r) {
            if (row[r] < n) {
                float v = acc[r];
                if (bias != nullptr) {
                    v += bias[row[r]];
                }
                out[row[r]] = ttbf_cast<T>(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill path: 32x64x32 register-tiled GEMM
// ---------------------------------------------------------------------------
//
// 256 threads are arranged as a 16x16 grid.  Each thread owns two rows and four
// columns of the 32x64 output tile.  The weight tile is unpacked straight into
// shared memory as exact float32 grid values, so the inner loop is plain FMA
// and the packed bytes are read exactly once per K tile.

constexpr int kGemmM = 32;
constexpr int kGemmN = 64;
constexpr int kGemmK = 32;
constexpr int kGemmThreads = 256;
constexpr int kGemmTileM = 2;
constexpr int kGemmTileN = 4;

template <typename T, int BITS>
__global__ __launch_bounds__(kGemmThreads)
void gemm_kernel(const T* __restrict__ x,
                 const uint8_t* __restrict__ codes,
                 const int8_t* __restrict__ exponents,
                 const float* __restrict__ bias,
                 T* __restrict__ out,
                 int64_t m, int64_t n, int64_t k) {
    using G = Grid<BITS>;
    __shared__ float s_a[kGemmM * kGemmK];
    __shared__ float s_w[kGemmN * kGemmK];

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int64_t base_m = static_cast<int64_t>(blockIdx.y) * kGemmM;
    const int64_t base_n = static_cast<int64_t>(blockIdx.x) * kGemmN;

    float acc[kGemmTileM][kGemmTileN];
#pragma unroll
    for (int i = 0; i < kGemmTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kGemmTileN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < k; k0 += kGemmK) {
        // --- activation tile -------------------------------------------------
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int idx = tid + it * kGemmThreads;
            const int r = idx >> 5;
            const int c = idx & 31;
            const int64_t gm = base_m + r;
            const int64_t gk = k0 + c;
            float v = 0.0f;
            if (gm < m && gk < k) {
                v = ttbf_load<T>(x + gm * k + gk);
            }
            s_a[r * kGemmK + c] = v;
        }

        // --- weight tile: eight codes per thread, unpacked on the way in -----
        {
            const int wrow = tid >> 2;
            const int chunk = tid & 3;
            const int kk = chunk * 8;
            const int64_t gn = base_n + wrow;
            const int64_t gk = k0 + kk;
            float value[8];
            if (gn < n && gk < k) {
                const float scale =
                    ttbf_exp2i(exponents[gn * (k >> 4) + (gk >> 4)]);
                const uint8_t* src =
                    codes + (BITS == 4 ? (gn * k + gk) >> 1 : gn * k + gk);
                int code[8];
                if (BITS == 4) {
                    const uint32_t word =
                        *reinterpret_cast<const uint32_t*>(src);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        code[i] = static_cast<int>((word >> (4 * i)) & 0xFu);
                    }
                } else {
                    const uint2 word = *reinterpret_cast<const uint2*>(src);
                    const uint32_t words[2] = {word.x, word.y};
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        code[i] = static_cast<int>(
                            (words[i >> 2] >> (8 * (i & 3))) & 0xFFu);
                    }
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    value[i] = ttbf_value(code[i], G::kOffset, G::kInvLevels,
                                          scale);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    value[i] = 0.0f;
                }
            }
            float* dst = s_w + wrow * kGemmK + kk;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                dst[i] = value[i];
            }
        }
        __syncthreads();

        // --- 8 accumulators per thread ---------------------------------------
#pragma unroll
        for (int kk = 0; kk < kGemmK; ++kk) {
            const float a0 = s_a[(ty * kGemmTileM + 0) * kGemmK + kk];
            const float a1 = s_a[(ty * kGemmTileM + 1) * kGemmK + kk];
            const float w0 = s_w[(tx * kGemmTileN + 0) * kGemmK + kk];
            const float w1 = s_w[(tx * kGemmTileN + 1) * kGemmK + kk];
            const float w2 = s_w[(tx * kGemmTileN + 2) * kGemmK + kk];
            const float w3 = s_w[(tx * kGemmTileN + 3) * kGemmK + kk];
            acc[0][0] += a0 * w0;
            acc[0][1] += a0 * w1;
            acc[0][2] += a0 * w2;
            acc[0][3] += a0 * w3;
            acc[1][0] += a1 * w0;
            acc[1][1] += a1 * w1;
            acc[1][2] += a1 * w2;
            acc[1][3] += a1 * w3;
        }
        __syncthreads();
    }

    // --- epilogue ------------------------------------------------------------
#pragma unroll
    for (int i = 0; i < kGemmTileM; ++i) {
        const int64_t gm = base_m + ty * kGemmTileM + i;
        if (gm >= m) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < kGemmTileN; ++j) {
            const int64_t gn = base_n + tx * kGemmTileN + j;
            if (gn >= n) {
                continue;
            }
            float v = acc[i][j];
            if (bias != nullptr) {
                v += bias[gn];
            }
            out[gm * n + gn] = ttbf_cast<T>(v);
        }
    }
}

__global__ void selftest_kernel(int32_t* out) {
    *out = 0x54544642;  // "TTFB"
}

// ---------------------------------------------------------------------------
// Prefill on RDNA4 matrix cores
// ---------------------------------------------------------------------------
//
// bf8 and bf4 values have 7 and 3 significant bits, so every dequantised value
// is exactly representable in bf16.  That makes a bf16 WMMA multiply lossless
// with respect to the stored weight: the only difference from the fp32 path is
// the fp32 accumulation order.  The intrinsic exists on gfx1200/gfx1201 only;
// other targets keep the fp32 GEMM.

#if defined(__gfx1200__) || defined(__gfx1201__)
#define TTBF_HAS_WMMA 1
#else
#define TTBF_HAS_WMMA 0
#endif

using WmmaA = short __attribute__((ext_vector_type(8)));
using WmmaD = float __attribute__((ext_vector_type(8)));

// Element (row = lane % 16, col = kb + i + 8 * (lane / 16)) of ``base``.
__device__ __forceinline__
WmmaA ttbf_wmma_row_frag(const __hip_bfloat16* __restrict__ base, int64_t ld,
                         int lane) {
    const int64_t off = static_cast<int64_t>(lane & 15) * ld + 8 * (lane >> 4);
    return *reinterpret_cast<const WmmaA*>(base + off);
}

// WMMA tile, tunable at compile time for experiments.
//
// K controls how many bytes of a weight row one tile reads.  A bf4 row is only
// ``k/2`` bytes, so at K = 32 a row contributes 16 bytes per tile and a warp
// touches eight 128-byte lines to use 16 bytes of each.  Larger K lengthens the
// per-row run (K/2 bytes for bf4, K for bf8) at the cost of shared memory.
//
// N controls CTA width.  Every CTA stages the whole activation tile for its K
// slice, and the grid is ``n/N`` CTAs wide, so a wider CTA means fewer CTAs and
// less redundant activation traffic but a bigger LDS footprint.
//
// BUFS = 2 lets the next tile's shared-memory write overlap the matrix-core
// work; BUFS = 1 halves the footprint but puts a barrier between them.
// WMMA tile shape.
//
// MTILE x NTILE is the CTA's output block.  The grid is (n/NTILE, m/MTILE), so
// MTILE also sets how many times the packed weight is re-read: the whole
// n x k weight tile is streamed once per M block.  Doubling MTILE from 32 to
// 64 halves both that traffic and the total unpack work, which was worth 1.55x
// at m=512.
//
// WTM x WTN is the per-warp block.  A warp issues one WMMA per 16x16 quadrant,
// and each quadrant needs an A fragment and a W fragment (512 B each) from
// shared memory.  At 16x16 that is 1 KiB of LDS traffic per WMMA, which at
// 128 B/cycle is eight cycles -- longer than the WMMA itself, so the kernel is
// LDS-bandwidth bound.  A 32x32 warp block reuses each fragment across two
// quadrants and halves the traffic to 512 B per WMMA.
//
// K sets the bytes a weight row contributes per tile -- K/2 at bf4 density, K
// at bf8 -- and a warp only streams well once that is a full 128-byte line.
// Larger K costs shared memory, which decides how many CTAs share a CU.
template <int BITS, int MTILE, int NTILE, int K, int BUFS, int WTM, int WTN>
__global__ __launch_bounds__(32 * (MTILE / WTM) * (NTILE / WTN))
void gemm_wmma_kernel(const __hip_bfloat16* __restrict__ x,
                      const uint8_t* __restrict__ codes,
                      const int8_t* __restrict__ exponents,
                      const float* __restrict__ bias,
                      __hip_bfloat16* __restrict__ out,
                      int64_t m, int64_t n, int64_t k) {
    using G = Grid<BITS>;
    constexpr int kWarpsM = MTILE / WTM;
    constexpr int kWarpsN = NTILE / WTN;
    constexpr int kThreads = 32 * kWarpsM * kWarpsN;
    constexpr int kLd = K + 8;              // pad rows to rotate LDS banks
    // Codes per thread: the weight tile is N x K and every thread stages a
    // slice of it, so this grows as the CTA gets narrower.
    constexpr int kCodes = NTILE * K / kThreads;
    constexpr int kActs = K * MTILE / kThreads;
    constexpr int kActVecs = kActs / 4;     // uint2 per thread
    constexpr int kActThreads = kThreads / MTILE;   // threads per act row
    constexpr int kWThreads = kThreads / NTILE;     // threads per weight row
    static_assert(kThreads % MTILE == 0, "activation rows must divide evenly");
    static_assert(kThreads % NTILE == 0, "weight rows must divide evenly");
    static_assert(WTM % 16 == 0 && WTN % 16 == 0, "warp block is whole tiles");
    static_assert(MTILE % WTM == 0 && NTILE % WTN == 0, "CTA block divides");
    static_assert(kActs % 4 == 0, "activation slice must be a whole uint2");
    // Codes are little-endian by column: a word holds eight at bf4 density and
    // four at bf8, so a thread's span is a flat word array.
    constexpr int kGroups = kCodes / 16;
    constexpr int kCodeBytes = (BITS == 4) ? kCodes / 2 : kCodes;
    constexpr int kWordsTotal = kCodeBytes / 4;
    // Vector width: 16 bytes when the span divides evenly, else 8.  Every used
    // shape (K = 64, 128 and 256) lands on one of these; a 4-byte span would
    // need a separate staging path for a half 16-code group.
    constexpr bool kUseVec4 = (kCodeBytes % 16) == 0;
    constexpr int kVecBytes = kUseVec4 ? 16 : 8;
    constexpr int kVecs = kCodeBytes / kVecBytes;
    static_assert(kCodeBytes % 8 == 0, "code span must be a whole 8-byte load");
    static_assert(kCodes % 16 == 0, "staging works in whole 16-code groups");

    __shared__ __align__(16) __hip_bfloat16 s_a[BUFS][MTILE * kLd];
    __shared__ __align__(16) __hip_bfloat16 s_w[BUFS][NTILE * kLd];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;
    const int64_t base_m = static_cast<int64_t>(blockIdx.y) * MTILE;
    const int64_t base_n = static_cast<int64_t>(blockIdx.x) * NTILE;

    const int ar = tid / kActThreads;
    const int ac = (tid % kActThreads) * kActs;
    const int wrow = tid / kWThreads;
    const int wc = (tid % kWThreads) * kCodes;

    struct TileRegs {
        uint2 a_reg[kActVecs];
        uint32_t w[kWordsTotal];
        int w_exp[kGroups];
        bool a_ok;
        bool w_ok;
    };

    auto issue = [&](int64_t k0) {
        TileRegs r{};
        const int64_t agm = base_m + ar;
        const int64_t agk = k0 + ac;
        r.a_ok = (agm < m && agk + kActs <= k);
        if (r.a_ok) {
#pragma unroll
            for (int v = 0; v < kActVecs; ++v) {
                r.a_reg[v] = *reinterpret_cast<const uint2*>(
                    x + agm * k + agk + v * 4);
            }
        }
        const int64_t gn = base_n + wrow;
        const int64_t gk = k0 + wc;
        r.w_ok = (gn < n && gk + kCodes <= k);
        if (r.w_ok) {
#pragma unroll
            for (int g = 0; g < kGroups; ++g) {
                r.w_exp[g] = exponents[gn * (k >> 4) + (gk >> 4) + g];
            }
            const uint8_t* src =
                codes + (BITS == 4 ? (gn * k + gk) >> 1 : gn * k + gk);
#pragma unroll
            for (int v = 0; v < kVecs; ++v) {
                const uint8_t* p = src + v * kVecBytes;
                if constexpr (kUseVec4) {
                    const uint4 val = *reinterpret_cast<const uint4*>(p);
                    r.w[v * 4 + 0] = val.x;
                    r.w[v * 4 + 1] = val.y;
                    r.w[v * 4 + 2] = val.z;
                    r.w[v * 4 + 3] = val.w;
                } else {
                    const uint2 val = *reinterpret_cast<const uint2*>(p);
                    r.w[v * 2 + 0] = val.x;
                    r.w[v * 2 + 1] = val.y;
                }
            }
        }
        return r;
    };

    auto store = [&](const TileRegs& r, int buf, int64_t k0) {
        const int64_t agm = base_m + ar;
        const int64_t agk = k0 + ac;
        __hip_bfloat16* adst = s_a[buf] + ar * kLd + ac;
        if (r.a_ok) {
#pragma unroll
            for (int v = 0; v < kActVecs; ++v) {
                *reinterpret_cast<uint2*>(adst + v * 4) = r.a_reg[v];
            }
        } else {
            for (int j = 0; j < kActs; ++j) {
                adst[j] = (agm < m && agk + j < k)
                              ? x[agm * k + agk + j]
                              : __float2bfloat16(0.0f);
            }
        }

        const int64_t gn = base_n + wrow;
        const int64_t gk = k0 + wc;
        __hip_bfloat16* wdst = s_w[buf] + wrow * kLd + wc;
        if (r.w_ok) {
#pragma unroll
            for (int g = 0; g < kGroups; ++g) {
                const float factor = G::kInvLevels * ttbf_exp2i(r.w_exp[g]);
                uint32_t out_words[8];
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const int i = 2 * q;
                    const int c = g * 16 + i;
                    int c0;
                    int c1;
                    if (BITS == 4) {
                        // Two codes per byte: eight codes per word.
                        const uint32_t w = r.w[c >> 3];
                        c0 = static_cast<int>((w >> (4 * (c & 7))) & 0xFu);
                        c1 = static_cast<int>((w >> (4 * ((c + 1) & 7))) & 0xFu);
                    } else {
                        // One code per byte: four codes per word.
                        const uint32_t w = r.w[c >> 2];
                        c0 = static_cast<int>((w >> (8 * (c & 3))) & 0xFFu);
                        c1 = static_cast<int>((w >> (8 * ((c + 1) & 3))) & 0xFFu);
                    }
                    const float v0 =
                        static_cast<float>(c0 - G::kOffset) * factor;
                    const float v1 =
                        static_cast<float>(c1 - G::kOffset) * factor;
                    out_words[q] = ttbf_pack2<__hip_bfloat16>(v0, v1);
                }
                // 16 bf16 = 32 bytes = two 16-byte stores.
                uint4 lo;
                lo.x = out_words[0];
                lo.y = out_words[1];
                lo.z = out_words[2];
                lo.w = out_words[3];
                uint4 hi;
                hi.x = out_words[4];
                hi.y = out_words[5];
                hi.z = out_words[6];
                hi.w = out_words[7];
                *reinterpret_cast<uint4*>(wdst + g * 16) = lo;
                *reinterpret_cast<uint4*>(wdst + g * 16 + 8) = hi;
            }
        } else if (gn < n) {
            // Partial K span.  The API only promises ``k % 16 == 0``, which is
            // coarser than this thread's ``kCodes``-wide span (64 or 128 for
            // the shapes in use), so the last tile of a k that is not a
            // multiple of the tile width ends mid-span.  Zero-fill past the end
            // rather than dropping the span: dropping it would discard the
            // valid leading columns too and silently shrink the dot product.
#pragma unroll 1
            for (int c = 0; c < kCodes; ++c) {
                const int64_t col = gk + c;
                float value = 0.0f;
                if (col < k) {
                    const int64_t bit = gn * k + col;
                    const int code = (BITS == 4)
                                         ? static_cast<int>(
                                               (codes[bit >> 1] >>
                                                (4 * (bit & 1))) &
                                               0xF)
                                         : static_cast<int>(codes[bit]);
                    const float factor =
                        G::kInvLevels *
                        ttbf_exp2i(exponents[gn * (k >> 4) + (col >> 4)]);
                    value = static_cast<float>(code - G::kOffset) * factor;
                }
                wdst[c] = __float2bfloat16(value);
            }
        } else {
            // Row beyond the output width: the tile is padding, so zero it
            // with the same 16-byte stores the fast path uses.  ``kCodes`` is a
            // whole number of 16-code groups, so the 8-wide step divides.
#pragma unroll
            for (int c = 0; c < kCodes; c += 8) {
                const uint4 zero{};
                *reinterpret_cast<uint4*>(wdst + c) = zero;
            }
        }
    };

    constexpr int kQuadM = WTM / 16;
    constexpr int kQuadN = WTN / 16;
    constexpr int kQuads = kQuadM * kQuadN;
    // A 16x16 warp block has a single accumulator, so every WMMA in a K tile
    // would depend on the previous one and the matrix core would stall between
    // issues; two banks alternate to break the chain.  With four quadrants the
    // dependency distance is already four, so one bank is enough and the
    // registers are better spent on the fragments.
    constexpr int kBanks = (kQuads == 1) ? 2 : 1;
    WmmaD acc[kQuads][kBanks];
#pragma unroll
    for (int q = 0; q < kQuads; ++q) {
#pragma unroll
        for (int b = 0; b < kBanks; ++b) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[q][b][i] = 0.0f;
            }
        }
    }

    store(issue(0), 0, 0);
    __syncthreads();

    int buf = 0;
    for (int64_t k0 = 0; k0 < k; k0 += K) {
        const int64_t next = k0 + K;
        const bool has_next = (next < k);
        // Issue the next tile's loads before the matrix-core block.  They are
        // consumed by the store after it, so a full tile of WMMA covers the
        // memory latency.  A memory-clobber fence here was measured and made
        // no difference on this target, so none is used.
        TileRegs next_regs{};
        if (has_next) {
            next_regs = issue(next);
        }

#if TTBF_HAS_WMMA
        const __hip_bfloat16* a_base = s_a[buf] + (warp_m * WTM) * kLd;
        const __hip_bfloat16* w_base = s_w[buf] + (warp_n * WTN) * kLd;
#pragma unroll
        for (int kb = 0; kb < K; kb += 16) {
            // Each fragment is loaded once and reused across the other axis of
            // the warp block, which is what halves LDS traffic per WMMA.
            WmmaA af[kQuadM];
            WmmaA bf[kQuadN];
#pragma unroll
            for (int i = 0; i < kQuadM; ++i) {
                af[i] =
                    ttbf_wmma_row_frag(a_base + i * 16 * kLd + kb, kLd, lane);
            }
#pragma unroll
            for (int j = 0; j < kQuadN; ++j) {
                bf[j] =
                    ttbf_wmma_row_frag(w_base + j * 16 * kLd + kb, kLd, lane);
            }
#pragma unroll
            for (int i = 0; i < kQuadM; ++i) {
#pragma unroll
                for (int j = 0; j < kQuadN; ++j) {
                    WmmaD& dst =
                        acc[i * kQuadN + j][(kBanks == 2) ? ((kb / 16) & 1) : 0];
                    dst = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                        af[i], bf[j], dst);
                }
            }
        }
#endif

        if (has_next) {
            if constexpr (BUFS == 2) {
                store(next_regs, buf ^ 1, next);
            } else {
                // Single buffer: drain the matrix-core reads before refilling.
                __syncthreads();
                store(next_regs, 0, next);
            }
        }
        __syncthreads();
        buf = (BUFS == 2) ? (buf ^ 1) : 0;
    }

#if TTBF_HAS_WMMA
    // D reg (l, i) = D[i + 8*(l/16)][colbase + l%16].
#pragma unroll
    for (int qi = 0; qi < kQuadM; ++qi) {
#pragma unroll
        for (int qj = 0; qj < kQuadN; ++qj) {
            const int q = qi * kQuadN + qj;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int64_t gm =
                    base_m + warp_m * WTM + qi * 16 + i + 8 * (lane >> 4);
                const int64_t gn = base_n + warp_n * WTN + qj * 16 + (lane & 15);
                if (gm >= m || gn >= n) {
                    continue;
                }
                float v = acc[q][0][i];
                if constexpr (kBanks == 2) {
                    v += acc[q][1][i];
                }
                if (bias != nullptr) {
                    v += bias[gn];
                }
                out[gm * n + gn] = __float2bfloat16(v);
            }
        }
    }
#endif
}

#undef TTBF_HAS_WMMA

// ---------------------------------------------------------------------------
// Host-side dispatch (still torch-free)
// ---------------------------------------------------------------------------

// True when the current device exposes the gfx12 WMMA instruction.
//
// The architecture probe is cached per device because hipGetDeviceProperties is
// a driver call, but the cache is process-wide while the BF16 linear path can be
// entered from several Python threads at once on a GIL-disabled build.  A plain
// `bool initialised[]` guard is a data race there: two threads can both observe
// the flag as unset and write `cache[device]` concurrently, which is undefined
// behaviour and leaves the dispatch decision without a publication order.
// `std::call_once` gives each device one synchronized initialization, and its
// happens-before edge publishes `cache[device]` to every later caller.
bool ttbf_wmma_device() {
    static int cache[64] = {0};
    static std::once_flag initialised[64];
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= 64) {
        return false;
    }
    std::call_once(initialised[device], [device]() {
        hipDeviceProp_t props{};
        const bool ok = hipGetDeviceProperties(&props, device) == hipSuccess &&
                        (std::strncmp(props.gcnArchName, "gfx1200", 7) == 0 ||
                         std::strncmp(props.gcnArchName, "gfx1201", 7) == 0);
        cache[device] = ok ? 1 : 0;
    });
    return cache[device] != 0;
}

template <typename T>
hipError_t dequant_impl(const uint8_t* codes, const int8_t* exponents, int bits,
                        int64_t rows, int64_t k, void* out, hipStream_t stream) {
    const int64_t total = rows * (k >> 3);
    int blocks = static_cast<int>((total + 255) / 256);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 65535) {
        blocks = 65535;
    }
    if (bits == 4) {
        dequant_kernel<T, 4><<<blocks, 256, 0, stream>>>(
            codes, exponents, rows, k, static_cast<T*>(out));
    } else {
        dequant_kernel<T, 8><<<blocks, 256, 0, stream>>>(
            codes, exponents, rows, k, static_cast<T*>(out));
    }
    return hipGetLastError();
}

template <typename T>
hipError_t quantize_impl(const float* x, int bits, int64_t rows, int64_t k,
                         uint8_t* codes, int8_t* exponents,
                         hipStream_t stream) {
    const int64_t total = rows * (k >> 4);
    int blocks = static_cast<int>((total + 255) / 256);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 65535) {
        blocks = 65535;
    }
    if (bits == 4) {
        quantize_kernel<4><<<blocks, 256, 0, stream>>>(
            x, rows, k, codes, exponents);
    } else {
        quantize_kernel<8><<<blocks, 256, 0, stream>>>(
            x, rows, k, codes, exponents);
    }
    return hipGetLastError();
}

template <typename T>
hipError_t embedding_impl(const int64_t* ids, const uint8_t* codes,
                          const int8_t* exponents, int bits, int64_t tokens,
                          int64_t rows, int64_t k, void* out,
                          hipStream_t stream) {
    const int64_t total = tokens * (k >> 3);
    int blocks = static_cast<int>((total + 255) / 256);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 65535) {
        blocks = 65535;
    }
    if (bits == 4) {
        embedding_kernel<T, 4><<<blocks, 256, 0, stream>>>(
            ids, codes, exponents, tokens, rows, k, static_cast<T*>(out));
    } else {
        embedding_kernel<T, 8><<<blocks, 256, 0, stream>>>(
            ids, codes, exponents, tokens, rows, k, static_cast<T*>(out));
    }
    return hipGetLastError();
}

template <typename T>
hipError_t decompose_impl(const float* x, int bits, int64_t rows, int64_t k,
                          uint8_t* codes, int8_t* exponents, int32_t* failed,
                          hipStream_t stream) {
    const int64_t total = rows * (k >> 4);
    int blocks = static_cast<int>((total + 255) / 256);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 65535) {
        blocks = 65535;
    }
    if (bits == 4) {
        decompose_kernel<4><<<blocks, 256, 0, stream>>>(
            x, rows, k, codes, exponents, failed);
    } else {
        decompose_kernel<8><<<blocks, 256, 0, stream>>>(
            x, rows, k, codes, exponents, failed);
    }
    return hipGetLastError();
}

template <typename T>
hipError_t linear_impl(const void* x, const uint8_t* codes,
                       const int8_t* exponents, int bits, int64_t m, int64_t n,
                       int64_t k, const void* bias, void* out,
                       hipStream_t stream) {
    const T* xp = static_cast<const T*>(x);
    const float* bp = static_cast<const float*>(bias);
    T* op = static_cast<T*>(out);
    if (m == 1) {
        const int blocks =
            static_cast<int>((n + kGemvOutputs - 1) / kGemvOutputs);
        if (bits == 4) {
            gemv_kernel<T, 4><<<blocks, kGemvThreads, 0, stream>>>(
                xp, codes, exponents, bp, op, n, k);
        } else {
            gemv_kernel<T, 8><<<blocks, kGemvThreads, 0, stream>>>(
                xp, codes, exponents, bp, op, n, k);
        }
    } else {
        const dim3 grid(
            static_cast<unsigned>((n + kGemmN - 1) / kGemmN),
            static_cast<unsigned>((m + kGemmM - 1) / kGemmM));
        if constexpr (std::is_same<T, __hip_bfloat16>::value) {
            // bf8/bf4 grid values are exact in bf16, so the matrix-core path
            // changes only the fp32 accumulation order.
            if (ttbf_wmma_device()) {
                const auto* xb = static_cast<const __hip_bfloat16*>(x);
                const auto* bb = static_cast<const float*>(bias);
                auto* ob = static_cast<__hip_bfloat16*>(out);
                // Tile choice, all measured rather than assumed.  Two things
                // move the large-M shapes: the M block sets how many times the
                // packed weight is re-read (once per block), and the warp block
                // sets LDS traffic per WMMA (one A and one W fragment, 1 KiB at
                // 16x16, 512 B at 32x32).
                //
                //  * m <= 16 / m <= 32: a 16- or 32-row block is the only way
                //    to avoid computing rows nobody asked for, and at these
                //    sizes the card is latency-bound rather than bandwidth- or
                //    arithmetic-bound, so the wide K = 256 tile wins because it
                //    makes each weight row a whole 128-byte line.  This shape
                //    tops out at 512 output tiles, i.e. 16 warps per CU, which
                //    is the real ceiling here; splitting K across CTAs would
                //    raise it but needs a partial-sum workspace.
                //  * m <= 64: a 64-row block with the same 16x16 warp tile.
                //    32x32 warp tiles need 32 rows per warp, so they would
                //    leave this block with only two warps.
                //  * m > 64: 128x128 with 32x32 warp tiles.  Both blocks double
                //    against 64x64, which halves the weight re-reads and halves
                //    LDS traffic per WMMA, and it still fits three CTAs per CU
                //    at K = 64.
                auto launch = [&](auto tile_m, auto tile_n, auto kk, auto bufs,
                                  auto wtm, auto wtn) {
                    constexpr int kM = decltype(tile_m)::value;
                    constexpr int kN = decltype(tile_n)::value;
                    constexpr int kK = decltype(kk)::value;
                    constexpr int kB = decltype(bufs)::value;
                    constexpr int kWM = decltype(wtm)::value;
                    constexpr int kWN = decltype(wtn)::value;
                    constexpr int kThreads = 32 * (kM / kWM) * (kN / kWN);
                    const dim3 g(static_cast<unsigned>((n + kN - 1) / kN),
                                 static_cast<unsigned>((m + kM - 1) / kM));
                    if (bits == 4) {
                        gemm_wmma_kernel<4, kM, kN, kK, kB, kWM, kWN>
                            <<<g, kThreads, 0, stream>>>(xb, codes, exponents,
                                                         bb, ob, m, n, k);
                    } else {
                        gemm_wmma_kernel<8, kM, kN, kK, kB, kWM, kWN>
                            <<<g, kThreads, 0, stream>>>(xb, codes, exponents,
                                                         bb, ob, m, n, k);
                    }
                };
                using I1 = std::integral_constant<unsigned, 1>;
                using I16 = std::integral_constant<unsigned, 16>;
                using I32 = std::integral_constant<unsigned, 32>;
                using I64 = std::integral_constant<unsigned, 64>;
                using I128 = std::integral_constant<unsigned, 128>;
                using I256 = std::integral_constant<unsigned, 256>;
                if (m <= 16) {
                    launch(I16{}, I64{}, I256{}, I1{}, I16{}, I16{});
                } else if (m <= 32) {
                    launch(I32{}, I64{}, I256{}, I1{}, I16{}, I16{});
                } else if (m <= 64) {
                    launch(I64{}, I64{}, I128{}, I1{}, I16{}, I16{});
                } else {
                    launch(I128{}, I128{}, I64{}, I1{}, I32{}, I32{});
                }
                return hipGetLastError();
            }
        }
        if (bits == 4) {
            gemm_kernel<T, 4><<<grid, kGemmThreads, 0, stream>>>(
                xp, codes, exponents, bp, op, m, n, k);
        } else {
            gemm_kernel<T, 8><<<grid, kGemmThreads, 0, stream>>>(
                xp, codes, exponents, bp, op, m, n, k);
        }
    }
    return hipGetLastError();
}

}  // namespace

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

hipError_t ttbf_dequant(const uint8_t* codes, const int8_t* exponents, int bits,
                        int64_t rows, int64_t k, void* out, int dtype,
                        hipStream_t stream) {
    if ((bits != 4 && bits != 8) || rows <= 0 || k <= 0 || (k & 15) != 0 ||
        codes == nullptr || exponents == nullptr || out == nullptr) {
        return hipErrorInvalidValue;
    }
    switch (dtype) {
        case TTBF_DTYPE_F32:
            return dequant_impl<float>(codes, exponents, bits, rows, k, out,
                                       stream);
        case TTBF_DTYPE_F16:
            return dequant_impl<__half>(codes, exponents, bits, rows, k, out,
                                        stream);
        case TTBF_DTYPE_BF16:
            return dequant_impl<__hip_bfloat16>(codes, exponents, bits, rows, k,
                                                out, stream);
        default:
            return hipErrorInvalidValue;
    }
}

hipError_t ttbf_quantize(const float* x, int bits, int64_t rows, int64_t k,
                         uint8_t* codes, int8_t* exponents,
                         hipStream_t stream) {
    if ((bits != 4 && bits != 8) || rows <= 0 || k <= 0 || (k & 15) != 0 ||
        x == nullptr || codes == nullptr || exponents == nullptr) {
        return hipErrorInvalidValue;
    }
    return quantize_impl<float>(x, bits, rows, k, codes, exponents, stream);
}

hipError_t ttbf_embedding(const int64_t* ids, const uint8_t* codes,
                          const int8_t* exponents, int bits, int64_t tokens,
                          int64_t rows, int64_t k, void* out, int dtype,
                          hipStream_t stream) {
    if ((bits != 4 && bits != 8) || tokens < 0 || rows <= 0 || k <= 0 ||
        (k & 15) != 0 || codes == nullptr || exponents == nullptr) {
        return hipErrorInvalidValue;
    }
    if (tokens == 0) {
        // An empty batch is a success, and its tensors have null data
        // pointers, so this has to come before the pointer checks.
        return hipSuccess;
    }
    if (ids == nullptr || out == nullptr) {
        return hipErrorInvalidValue;
    }
    switch (dtype) {
        case TTBF_DTYPE_F32:
            return embedding_impl<float>(ids, codes, exponents, bits, tokens,
                                         rows, k, out, stream);
        case TTBF_DTYPE_F16:
            return embedding_impl<__half>(ids, codes, exponents, bits, tokens,
                                          rows, k, out, stream);
        case TTBF_DTYPE_BF16:
            return embedding_impl<__hip_bfloat16>(ids, codes, exponents, bits,
                                                  tokens, rows, k, out, stream);
        default:
            return hipErrorInvalidValue;
    }
}

hipError_t ttbf_decompose(const float* x, int bits, int64_t rows, int64_t k,
                          uint8_t* codes, int8_t* exponents, int32_t* failed,
                          hipStream_t stream) {
    if ((bits != 4 && bits != 8) || rows <= 0 || k <= 0 || (k & 15) != 0 ||
        x == nullptr || codes == nullptr || exponents == nullptr ||
        failed == nullptr) {
        return hipErrorInvalidValue;
    }
    return decompose_impl<float>(x, bits, rows, k, codes, exponents, failed,
                                 stream);
}

hipError_t ttbf_linear(const void* x, int dtype, const uint8_t* codes,
                       const int8_t* exponents, int bits, int64_t m, int64_t n,
                       int64_t k, const void* bias, void* out,
                       hipStream_t stream) {
    if (m < 0 || n <= 0 || k <= 0 || (bits != 4 && bits != 8) ||
        (k & 15) != 0 || codes == nullptr || exponents == nullptr) {
        return hipErrorInvalidValue;
    }
    if (m == 0) {
        // As above: an empty batch must not fail on its null data pointers.
        return hipSuccess;
    }
    if (x == nullptr || out == nullptr) {
        return hipErrorInvalidValue;
    }
    switch (dtype) {
        case TTBF_DTYPE_F32:
            return linear_impl<float>(x, codes, exponents, bits, m, n, k, bias,
                                      out, stream);
        case TTBF_DTYPE_F16:
            return linear_impl<__half>(x, codes, exponents, bits, m, n, k, bias,
                                       out, stream);
        case TTBF_DTYPE_BF16:
            return linear_impl<__hip_bfloat16>(x, codes, exponents, bits, m, n,
                                               k, bias, out, stream);
        default:
            return hipErrorInvalidValue;
    }
}

hipError_t ttbf_selftest(int32_t* out, hipStream_t stream) {
    if (out == nullptr) {
        return hipErrorInvalidValue;
    }
    selftest_kernel<<<1, 1, 0, stream>>>(out);
    return hipGetLastError();
}

}  // extern "C"
