// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
/*
 * ttbfp - CPU kernels for Tenstorrent block floats (bf8_b / bf4_b / bf2_b).
 *
 * Bit-exact port of the packing/unpacking semantics used by tt-metal's
 * tt-llk reference helpers:
 *
 *   tt_metal/tt-llk/tests/python_tests/helpers/{pack,unpack,bfp_format_utils}.py
 *
 * Format summary
 * --------------
 *   bf8_b (Tenstorrent Bfp8_b / BLOCKFP8):
 *           16-element blocks, one shared 8-bit exponent per block,
 *           each datum is 8 bits: 1 sign + 7 magnitude bits.
 *   bf4_b  (Tenstorrent Bfp4_b / BLOCKFP4):
 *           16-element blocks, one shared 8-bit exponent per block,
 *           each datum is 4 bits: 1 sign + 3 magnitude bits,
 *           two datums per byte (even datum in the low nibble).
 *
 *   decode(datum) = sign * magnitude * 2^(exponent - 126 - magnitude_bits)
 *
 *   The packer first truncates fp32 -> bf16 (top 16 bits), normalizes each
 *   block to the maximum element exponent, rounds the 7-bit BFP8 magnitude
 *   (round-to-nearest, ties away from zero, one guard bit), and for BFP4
 *   truncates the 7-bit magnitude to 3 bits.  See README.md for details.
 *
 * Storage layouts
 * ---------------
 *   Flat (default API): a row-major [rows, cols] tensor is split into
 *   contiguous 16-element blocks along the last dimension.  Blocks are
 *   stored as:
 *
 *       exps[rows * nblk]                  (nblk = ceil(cols / 16))
 *       data[rows * nblk * 16]             (bf8_b)
 *       data[rows * nblk * 8]              (bf4_b)
 *
 *   `exps` and `data` are separate buffers (struct-of-arrays).  Padding in
 *   a partial final block is encoded as zero.
 *
 *   Tile: ttbfp_tile_pack()/ttbfp_tile_unpack() use the exact 32x32 L1 tile
 *   byte layout: 64 exponent bytes followed by 1024 (bf8_b) or 512
 *   (bf4_b) datum bytes.  Faces are 16x16 row-major in the order
 *   (top-left, top-right, bottom-left, bottom-right), i.e. the layout
 *   produced by tt-metal's `tilize_block`.
 *
 * Threading
 * ---------
 * Parallel loops cover independent output tiles only, so every entry point
 * returns identical bits for any thread count.  The library never calls
 * omp_set_*; the caller owns the pool.
 *
 * When running or measuring, set:
 *
 *   OMP_NUM_THREADS=<cores>    no more threads than cores you own
 *   OMP_WAIT_POLICY=passive    do not let idle workers spin
 *
 * GNU OpenMP spins by default after a parallel region ends (GOMP_SPINCOUNT
 * is large), so back-to-back kernels burn a core per idle worker and report
 * inflated CPU time; the spinners also steal from the next kernel.  The
 * passive wait policy removes that.  It changes performance only, never
 * numerics.  Wall-clock timing is unreliable on a busy host because of this
 * and because samples quantize to the scheduler slice; prefer process CPU
 * time for benchmarks.
 */
#ifndef TTBFP_H
#define TTBFP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TTBFP_BLOCK 16

/*
 * Naming: the Tenstorrent block-float formats are vendor-specific and are
 * NOT the OCP MX (microscaling) formats.  We therefore expose them with a
 * "tt" prefix:
 *
 *   TTBFP_BF8_B  == TT Bfp8_b == BLOCKFP8  (E8M6: 8-bit shared exponent)
 *   TTBFP_BF4_B  == TT Bfp4_b == BLOCKFP4  (E8M2: 8-bit shared exponent)
 *   TTBFP_BF2_B  == TT Bfp2_b == BLOCKFP2  (E8M0: 8-bit shared exponent)
 *
 * The canonical names match hw/dtypes.md; the tt* spellings are aliases.
 *
 * The TT-internal spellings are kept as deprecated aliases.
 * The 5-bit-exponent FP16-family variants (TT Bfp8/Bfp4/Bfp2, a.k.a.
 * BFP8a/BFP4a/BFP2a) are not part of this library yet.
 */
typedef enum {
    TTBFP_BF8_B = 8, /* canonical: bf8_b (hw/dtypes.md) */
    TTBFP_BF4_B = 4, /* canonical: bf4_b */
    TTBFP_BF2_B = 2, /* canonical: bf2_b */
    TTBFP_TTBF8 = 8, /* deprecated alias */
    TTBFP_TTBF4 = 4, /* deprecated alias */
    TTBFP_TTBF2 = 2, /* deprecated alias */
    TTBFP_BFP8_B = 8, /* deprecated alias */
    TTBFP_BFP4_B = 4, /* deprecated alias */
    TTBFP_BFP2_B = 2  /* deprecated alias */
} ttbfp_fmt;

/* ---- size helpers ---------------------------------------------------- */

/* Number of 16-element blocks per row (cols is rounded up). */
size_t ttbfp_blocks_per_row(int64_t cols);
size_t ttbfp_exps_bytes(ttbfp_fmt fmt, int64_t rows, int64_t cols);
size_t ttbfp_data_bytes(ttbfp_fmt fmt, int64_t rows, int64_t cols);
size_t ttbfp_packed_bytes(ttbfp_fmt fmt, int64_t rows, int64_t cols);

/* Bytes per 32x32 tile: 1088 (bf8_b), 576 (bf4_b) or 320 (bf2_b). */
size_t ttbfp_tile_bytes(ttbfp_fmt fmt);

/* ---- pack / unpack --------------------------------------------------- */

/*
 * Quantize a row-major [rows, cols] fp32 tensor into `exps` + `data`.
 * `exps` must hold ttbfp_exps_bytes(); `data` ttbfp_data_bytes().
 * Elements of a partial final block are encoded as zero.
 */
void ttbfp_quantize(ttbfp_fmt fmt,
                    const float *src, int64_t rows, int64_t cols,
                    uint8_t *exps, uint8_t *data);

/*
 * Dequantize `exps` + `data` into a row-major [rows, cols] fp32 tensor.
 * If `flush_denormals` is non-zero, results with |v| < 2^-126 are flushed
 * to zero (matching Tensix FTZ behaviour).  If it is zero, subnormal weights
 * are preserved even when the calling thread has ambient FTZ/DAZ enabled:
 * the kernel clears MXCSR FTZ/DAZ for the call and restores it on return.
 */
void ttbfp_dequantize(ttbfp_fmt fmt,
                      const uint8_t *exps, const uint8_t *data,
                      int64_t rows, int64_t cols,
                      float *dst, int flush_denormals);

/* ---- fused matmul ---------------------------------------------------- */

/*
 * C[M,N] = A[M,K] * dequant(B)[K,N]
 *
 * A is row-major fp32 with leading dimension `lda` (>= K).
 * B is the packed [K, N] weight matrix (exps: K * nblk bytes, data:
 * K * nblk * datum_bytes).  C is row-major fp32 with leading dimension
 * `ldc` (>= N).
 *
 * B is never materialized in fp32; each 16-wide B block is decoded and
 * scaled on the fly and reused across every row of the current A row tile
 * (4, 8, or 16 rows, chosen from M and the packed layout).
 *
 * This is the row-major packed path.  For repeated use, prepack B with
 * ttbfp_transpose_packed() and call ttbfp_matmul_prepacked(): the k loop
 * then streams sequentially and runs ~1.4-1.5x faster.  Benchmark with
 * OMP_WAIT_POLICY=passive (see the Threading note above).
 */
void ttbfp_matmul(ttbfp_fmt fmt,
                  const float *A, int64_t lda,
                  const uint8_t *exps, const uint8_t *data,
                  int64_t M, int64_t K, int64_t N,
                  float *C, int64_t ldc, int flush_denormals);

/* Same, with A stored as bf16 (uint16_t bit patterns, row-major). */
void ttbfp_matmul_bf16(ttbfp_fmt fmt,
                       const uint16_t *A, int64_t lda,
                       const uint8_t *exps, const uint8_t *data,
                       int64_t M, int64_t K, int64_t N,
                       float *C, int64_t ldc, int flush_denormals);

/*
 * Block-major ("transposed") packed layout.
 *
 * The flat layout stores a [K, N] matrix as K rows of nblk blocks, so a
 * column block is strided by nblk*datum_bytes for consecutive k.  For a
 * matmul that is one cache line (and often one page) per k.  The
 * block-major layout stores, for each column block j, the K blocks that
 * belong to it contiguously:
 *
 *     exps_t[j * K + k]        data_t[(j * K + k) * datum_bytes]
 *
 * which makes the k loop a sequential stream.  Pack once, multiply many
 * times.  Same byte counts as the flat layout.
 */
void ttbfp_transpose_packed(ttbfp_fmt fmt,
                            const uint8_t *exps, const uint8_t *data,
                            int64_t K, int64_t N,
                            uint8_t *exps_t, uint8_t *data_t);

void ttbfp_matmul_prepacked(ttbfp_fmt fmt,
                            const float *A, int64_t lda,
                            const uint8_t *exps_t, const uint8_t *data_t,
                            int64_t M, int64_t K, int64_t N,
                            float *C, int64_t ldc, int flush_denormals);

void ttbfp_matmul_prepacked_bf16(ttbfp_fmt fmt,
                                 const uint16_t *A, int64_t lda,
                                 const uint8_t *exps_t, const uint8_t *data_t,
                                 int64_t M, int64_t K, int64_t N,
                                 float *C, int64_t ldc, int flush_denormals);

/*
 * Kernel-native layout: same values, datums stored as signed int8.
 *
 * The TT on-device datum is sign-magnitude (1 sign bit + magnitude bits),
 * which costs 8 uops to decode for bf8_b and 12 for bf4_b.  Storing
 * sign*magnitude as a two's-complement int8 instead makes the decode
 * vpmovsxbd + vcvtdq2ps + vmulps, i.e. 3 uops, and the decoded values are
 * identical, so every path stays bitwise equal.
 *
 * This is the same move ATen makes for _weight_int4pack_mm /
 * _weight_int8pack_mm: pack the weights into whatever shape makes the inner
 * loop cheapest.  It matters most when the kernel is decode-bound, i.e. for
 * small M (autoregressive decode): ~1.5x for bf8_b and ~2x for bf4_b at M=1.
 *
 * Layout (block-major, one int8 per datum):
 *     exps_t[j * K + k]        s8_t[(j * K + k) * 16]
 * bf8_b needs 16 bytes per block either way; bf4_b doubles from 8 to 16.
 */
void ttbfp_convert_s8(ttbfp_fmt fmt,
                      const uint8_t *exps, const uint8_t *data,
                      int64_t K, int64_t N,
                      uint8_t *exps_t, uint8_t *s8_t);

void ttbfp_matmul_prepacked_s8(ttbfp_fmt fmt,
                               const float *A, int64_t lda,
                               const uint8_t *exps_t, const uint8_t *s8_t,
                               int64_t M, int64_t K, int64_t N,
                               float *C, int64_t ldc, int flush_denormals);

void ttbfp_matmul_prepacked_s8_bf16(ttbfp_fmt fmt,
                                    const uint16_t *A, int64_t lda,
                                    const uint8_t *exps_t,
                                    const uint8_t *s8_t,
                                    int64_t M, int64_t K, int64_t N,
                                    float *C, int64_t ldc, int flush_denormals);

/*
 * Block-major ("transposed") packed layout.
 *
 * The flat layout stores a [K, N] matrix as K rows of nblk blocks, so a
 * column block is strided by nblk*datum_bytes for consecutive k.  For a
 * matmul that is one cache line (and often one page) per k.  The
 * block-major layout stores, for each column block j, the K blocks that
 * belong to it contiguously:
 *
 *     exps_t[j * K + k]        data_t[(j * K + k) * datum_bytes]
 *
 * which makes the k loop a sequential stream.  Pack once, multiply many
 * times.  Same byte counts as the flat layout.
 */
void ttbfp_transpose_packed(ttbfp_fmt fmt,
                            const uint8_t *exps, const uint8_t *data,
                            int64_t K, int64_t N,
                            uint8_t *exps_t, uint8_t *data_t);

void ttbfp_matmul_prepacked(ttbfp_fmt fmt,
                            const float *A, int64_t lda,
                            const uint8_t *exps_t, const uint8_t *data_t,
                            int64_t M, int64_t K, int64_t N,
                            float *C, int64_t ldc, int flush_denormals);

void ttbfp_matmul_prepacked_bf16(ttbfp_fmt fmt,
                                 const uint16_t *A, int64_t lda,
                                 const uint8_t *exps_t, const uint8_t *data_t,
                                 int64_t M, int64_t K, int64_t N,
                                 float *C, int64_t ldc, int flush_denormals);

/* ---- exact 32x32 tile helpers --------------------------------------- */

/* `tile` is a logical 32x32 row-major fp32 tile. */
void ttbfp_tile_pack(ttbfp_fmt fmt, const float *tile, uint8_t *out);
void ttbfp_tile_unpack(ttbfp_fmt fmt, const uint8_t *in, float *tile,
                       int flush_denormals);

/* ---- introspection --------------------------------------------------- */

const char *ttbfp_version(void);
/* "scalar", "avx2+fma", or "avx512" (best available on this CPU). */
const char *ttbfp_cpu_features(void);

#ifdef __cplusplus
}
#endif

#endif /* TTBFP_H */
