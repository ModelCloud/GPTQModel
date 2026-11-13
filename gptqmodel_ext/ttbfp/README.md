# ttbfp — Tenstorrent block floats executed on CPU

`ttbfp` is a small C++/AVX2 + NumPy library that implements Tenstorrent's
**block-float** tensor formats directly on ordinary CPUs, so models and kernels
that use them can run on non-Tenstorrent hardware.

This directory is vendored in GPT-QModel as `gptqmodel_ext/ttbfp/`.  It is a
standalone library: `setup.py` does not build it and it is not imported by the
`gptqmodel` package.  Build and test it from this directory with `make`.

It provides:

* bit-exact **quantize / dequantize** for `bf8_b`, `bf4_b` and `bf2_b`;
* a fused **dequantize + GEMM** (`C = A @ dequant(B)`) that never materializes
  the weights in fp32;
* exact **32×32 TT L1 tile** pack/unpack (1088 B for `bf8_b`, 576 B for
  `bf4_b`, 320 B for `bf2_b`);
* portable scalar, x86 AVX2+FMA, and x86 AVX-512 paths selected at runtime;
* OpenMP parallelism over independent output tiles / rows.
* optional PyTorch CPU helpers (`ttbfp.torch_ops`);
* an optional CUDA implementation of the same kernels (`src/ttbfp_cuda.cu`).

The quantizer is validated bit-for-bit against the actual `tt-llk` reference
helpers (`pack.py`, `unpack.py`, `bfp_format_utils.py`) over 200 random seeds ×
4 value distributions × all three formats, including NaNs/Infs.  See
`tools/verify_against_tt.py`.

---

## 1. Naming: `bf8_b` / `bf4_b` / `bf2_b`

Tenstorrent's block floats are **not** an industry-standard format.  Block
floating point itself is an old, generic idea, but the specific TT encoding
differs from the standardized block-scaled formats:

| | Tenstorrent `bf8_b` | OCP MXFP8 (industry standard) |
|---|---|---|
| block size | **16** datums | 32 datums |
| scale | shared **8-bit exponent**, bias 127 (or 5-bit/bias 15 for the `a` variants) | **E8M0** scale byte per 32 |
| element | sign + magnitude, 7 magnitude bits, **no exponent field** | FP8 E4M3 or E5M2 (sign + exponent + mantissa) |
| packing | exponents stored separately, then mantissas | scale section + element bytes |
| rounding | bf16 truncation, then block-normalize + guard-bit round-to-nearest/ties-away; `bf4_b` truncates the 7-bit magnitude to 3 bits | RNE to the element format |

Consequences:

* `bf8_b`/`bf4_b`/`bf2_b` are **vendor-specific** and are this repository's
  canonical names, defined in `hw/dtypes.md`.  The `_b` suffix is
  load-bearing: the un-suffixed `Bfp8`/`Bfp4`/`Bfp2` are the 5-bit-exponent
  FP16 family.  The older `ttbf8`/`ttbf4`/`ttbf2` spellings are deprecated
  aliases.
* The OCP **MX** formats (`mxfp8_e4m3`, `mxfp8_e5m2`, `mxfp4_e2m1`, `mxint8`,
  …) are the industry-standard block-scaled family — and they are **not native
  to P300C** (they are Quasar-specific in `tt-llk`).
* `bf16`, `fp16`, `fp8_e4m3`, `fp8_e5m2`, `tf32`, `int8`, `uint8`, `fp32` are
  standard or de-facto standard names and keep their normal spelling.

Recommended names:

| canonical | TT software name | TT marketing | meaning |
|---|---|---|---|
| `bf8_b` | `Bfp8_b` | BLOCKFP8 | 8-bit shared exp, 1+7 bits/datum, block 16 |
| `bf4_b` | `Bfp4_b` | BLOCKFP4 | 8-bit shared exp, 1+3 bits/datum, block 16 |
| `bf2_b` | `Bfp2_b` | BLOCKFP2 | 8-bit shared exp, 1+1 bits/datum, block 16 |
| `ttbf8a` | `Bfp8` | BFP8a | 5-bit shared exp (FP16 family), 1+7 bits/datum |
| `ttbf4a` | `Bfp4` | BFP4a | 5-bit shared exp, 1+3 bits/datum |
| `ttbf2a` | `Bfp2` | BFP2a | 5-bit shared exp, 1+1 bits/datum |

The `_b` suffix in TT's names means the **bfloat16-exponent family** (8-bit
exponent, bias 127); the un-suffixed `Bfp8`/`Bfp4`/`Bfp2` are the
FP16-exponent family (5-bit exponent, bias 15), which TT's ISA docs call
`BFP8a`/`BFP4a`/`BFP2a`.

---

## 2. P300C (Blackhole) native formats

P300C is a Blackhole card: two ASICs, 120 Tensix cores each, 64 GB GDDR6.
The vendor precision table for the Blackhole Tensix core lists:

```
Floating point        FP8, FP16, BFLOAT16
                      FP32 (output only)
Block floating point  BLOCKFP2, BLOCKFP4, BLOCKFP8
Integer               INT8, INT32 (output only)
Unsigned integer      UINT8
TensorFloat           TF32
Vector                VTF19, VFP32
```

Expanded, with the standard/non-standard classification:

> This table is about the **hardware**.  The library in this repository
> implements `bf8_b`, `bf4_b` and `bf2_b`; the other rows are context for naming and
> for deciding what is safe to call "standard".

| portable name | TT name(s) | native P300C Tensix? | industry standard? | notes |
|---|---|---|---|---|
| `bf8_b` | `Bfp8_b`, BLOCKFP8 | **yes** | **no — TT-specific** | E8M6, 16-datum block |
| `bf4_b` | `Bfp4_b`, BLOCKFP4 | **yes** | **no — TT-specific** | E8M2 |
| `bf2_b` | `Bfp2_b`, BLOCKFP2 | **yes** | **no — TT-specific** | E8M0 |
| `ttbf8a` | `Bfp8` (BFP8a) | yes (LLK enum) | **no — TT-specific** | E5M6, FP16-family |
| `ttbf4a` | `Bfp4` (BFP4a) | yes (LLK enum) | **no — TT-specific** | E5M2 |
| `ttbf2a` | `Bfp2` (BFP2a) | yes (LLK enum) | **no — TT-specific** | E5M0 |
| `ttvtf19` | VTF19 | yes (vector/SFPU) | **no — TT-specific** | 19-bit vector float |
| `fp8_e4m3` | `Fp8_e4m3` | **yes** | **yes — OCP OFP8** | Lf8 path with 4-bit exponent mode |
| `fp8_e5m2` | `Lf8` | **yes** | **yes — OCP OFP8** | default Lf8 mode |
| `fp16` | `Float16` | **yes** | **yes — IEEE 754 binary16** | |
| `bf16` | `Float16_b` | **yes** | **yes — OCP BF16 / de facto** | not IEEE 754 |
| `tf32` | `Tf32` | **yes** | **de facto (NVIDIA-defined)** | no formal standard |
| `int8` | `Int8` | **yes** | **yes** | |
| `uint8` | `UInt8` | **yes** | **yes** | |
| `int32` | `Int32` | yes (output only) | **yes** | |
| `uint32` | `UInt32` | enum/storage | **yes** | not in the vendor Tensix arithmetic table |
| `uint16` | `UInt16` | enum/storage/typecast | **yes** | not in the vendor Tensix arithmetic table |
| `fp32` | `Float32` | yes (output only) | **yes — IEEE 754 binary32** | |
| `vfp32` | VFP32 | yes (vector/SFPU) | yes (FP32 vector) | |

Not native to P300C (present in `tt-llk`/`tt-metal` for other architectures):

| portable name | TT name | where | industry standard? |
|---|---|---|---|
| `mxfp8_e4m3` | `MxFp8P` | Quasar (QSR) | yes — OCP MX |
| `mxfp8_e5m2` | `MxFp8R` | Quasar | yes — OCP MX |
| `mxfp4_e2m1` | `MxFp4` | Quasar | yes — OCP MX |
| `mxint8` / `mxint4` / `mxint2` | `MxInt8/4/2` | Quasar | yes — OCP MX |
| `int4` / `uint4` | `Int4` / `UInt4` | Quasar | common, not formally standardized |
| `int16` | `Int16` | Quasar Tensix / big RISC-V | yes |
| `ttint8_2x` / `ttuint8_2x` | `Int8_2x` / `UInt8_2x` | 4-row arch, src-reg only | **no — TT-specific packing** |
| `ttmxfp4_2x_a/b` | `MxFp4_2x_A/B` | Quasar, src-reg only | **no — TT-specific packing** |

The `testMan7`/`testMan2` enum values are internal test-only intermediate
formats and should not be exposed.

**Rule of thumb:** if the name contains `Bfp`/`BLOCKFP`, `VTF`, `_2x`, or
`Lf8`, it is TT-specific or a TT-internal spelling.  `Lf8` is just TT's name
for the standard E5M2 FP8 path, so expose it as `fp8_e5m2`.

---

## 3. Format definition

All TT block floats use **16-element blocks** of contiguous elements along the
last dimension.  Each block has one shared 8-bit exponent (bias 127), and each
datum is a sign bit followed by magnitude bits:

```
decode:  value = sign * magnitude * 2^(E - 126 - magnitude_bits)
```

| format | datum bits | magnitude bits | exponent offset | bytes / 16 datums |
|---|---|---|---|---|
| `bf8_b` | 8 | 7 | 133 | 16 |
| `bf4_b` | 4 | 3 | 129 | 8 |
| `bf2_b` | 2 | 1 | 127 | 4 |

`bf4_b` packs two datums per byte, **even index in the low nibble**.
`bf2_b` packs four datums per byte, lowest index in the low bits.

Packing (`bf8_b`; `bf4_b`/`bf2_b` truncate the 7-bit magnitude afterwards):

1. truncate fp32 to bf16 (drop the low 16 bits);
2. extract sign, 8-bit exponent, and a 7-bit mantissa `((bf16 & 0x7F) >> 1) | 0x40`;
3. shared exponent = max exponent in the block;
4. `delta = shared - elem_exp`; `shift = min(delta, 8)`;
   `guard = (mant >> min(delta-1, 8)) & 1` when `delta > 0`;
5. `q8 = ((mant >> shift) + guard) & 0x7F`;
6. `bf4_b`: `q4 = q8 >> 4`; `bf2_b`: `q2 = q8 >> 6`;
7. datum = `(sign << magnitude_bits) | q`, but **a zero magnitude drops the
   sign bit** (a sign-only datum decodes to −0 / −inf in hardware).

The shared exponent is stored separately, one byte per block.  A block whose
exponent byte is zero is decoded as all zeros (matching TT's FTZ behaviour).

---

## 4. Layouts

### Flat layout (default API)

For a row-major `[rows, cols]` tensor, `nblk = ceil(cols / 16)`:

```
exps[rows * nblk]                 # one byte per 16-element block
data[rows * nblk * 16]            # bf8_b
data[rows * nblk * 8]             # bf4_b
```

`exps` and `data` are separate buffers (struct-of-arrays).  Padding in a
partial final block is encoded as datum zero.

### Block-major layout (recommended for matmul)

The flat layout is row-major over K, so a fixed column block sits
`nblk * datum_bytes` away for each consecutive `k`.  Walking K then touches a
new cache line — and usually a new page — every step, which costs far more
than the arithmetic.

The block-major layout stores each column block's K blocks contiguously:

```
exps_t[j * K + k]                  # j = column block, k = row
data_t[(j * K + k) * datum_bytes]
```

Now the k loop is a sequential stream, and the kernel runs ~1.4-1.5x faster
for realistic batch sizes.  Pack once, multiply many times:

```c
ttbfp_transpose_packed(TTBFP_BF8_B, exps, data, K, N, exps_t, data_t);
ttbfp_matmul_prepacked(TTBFP_BF8_B, A, lda, exps_t, data_t, M, K, N, C, ldc, 1);
```

The byte counts are identical to the flat layout (it is a pure permutation).

### int8 kernel-native layout (fastest at every M)

The TT datum is sign-magnitude: one sign bit plus magnitude bits, which costs
8 instructions to decode for bf8_b and 12 for bf4_b.  Storing `sign*magnitude`
as a two's-complement int8 instead makes the decode `vpmovsxbd` + `vcvtdq2ps` +
`vmulps` — 3 instructions — and the decoded values are identical, so every path
stays bitwise equal.

This is the same trade ATen makes for `_weight_int4pack_mm` /
`_weight_int8pack_mm`: pack the weights into whatever shape makes the inner
loop cheapest.  It pays off twice: at small M (autoregressive decode) the
kernel is decode-bound, and at large M the 3-instruction decode leaves enough
registers and issue slots to use a 4x4 tile, which cuts A-panel traffic 4x.
Measured 1.2x over the sign-magnitude layout at M >= 16 and 1.7x at M = 1:

```c
ttbfp_convert_s8(TTBFP_BF8_B, exps, data, K, N, exps_t, s8);   /* once */
ttbfp_matmul_prepacked_s8(TTBFP_BF8_B, A, lda, exps_t, s8, M, K, N, C, ldc, 1);
```

```python
exps_t, s8 = ttbfp.prepack_s8(w, "bf4_b")
c = ttbfp.matmul_prepacked_s8(a, exps_t, s8, "bf4_b", cols=N)
```

bf8_b needs 16 bytes per block either way; bf4_b doubles from 8 to 16, which is
hidden at large M (compute-bound) and still wins at small M.

### TT tile layout

`ttbfp_tile_pack` / `ttbfp_tile_unpack` use the exact 32×32 L1 tile layout:
64 exponent bytes followed by 1024 (`bf8_b`) or 512 (`bf4_b`) datum bytes.
Faces are 16×16 row-major in the order top-left, top-right, bottom-left,
bottom-right — the layout produced by tt-metal's `tilize_block`.

---

## 5. Build and test

```bash
make                      # builds libttbfp.so (needs g++ + OpenMP)
make cuda                 # optional: libttbfp_cuda.so (needs nvcc + a GPU)
make test                 # pytest; requires numpy + pytest
make bench                # microbenchmarks
```

Set two OpenMP variables before running or benchmarking:

```bash
export OMP_NUM_THREADS=$(nproc)     # never more threads than cores you own
export OMP_WAIT_POLICY=passive      # stop idle workers spinning
```

`OMP_WAIT_POLICY=passive` matters more than it looks.  GNU OpenMP spins for a
long time after a parallel region ends, so back-to-back kernels burn a full
core per idle worker: an 8-thread run reports roughly 8x the CPU time, and the
spinners compete with the next kernel.  Passive waiting makes workers sleep;
it changes performance only, never results.  This is also documented at the
top of `src/ttbfp_cpu.cpp` and `include/ttbfp.h`, and in the repository skill
`.agents/skills/ttbfp-perf/SKILL.md`.

To compare two builds on the same buffers:

```bash
OMP_WAIT_POLICY=passive OMP_NUM_THREADS=8 \
    python bench/ab_compare.py --old /tmp/baseline.so --new ./libttbfp.so
```

The Python package is in `python/ttbfp`; either install it or use:

```bash
export TTBFP_LIB=$PWD/libttbfp.so
export PYTHONPATH=$PWD/python
```

`tools/verify_against_tt.py` needs `torch` and checks the quantizer against
the actual tt-llk reference implementation.

### C API

```c
#include "ttbfp.h"

size_t eb = ttbfp_exps_bytes(TTBFP_BF8_B, rows, cols);
size_t db = ttbfp_data_bytes(TTBFP_BF8_B, rows, cols);

ttbfp_quantize(TTBFP_BF8_B, src, rows, cols, exps, data);
ttbfp_dequantize(TTBFP_BF8_B, exps, data, rows, cols, dst, 1);

// C[M,N] = A[M,K] @ dequant(B)[K,N], B packed (row-major layout)
ttbfp_matmul(TTBFP_BF8_B, A, K, exps, data, M, K, N, C, N, 1);

// Same, with B in the block-major layout: ~1.4-1.5x faster.
ttbfp_transpose_packed(TTBFP_BF8_B, exps, data, K, N, exps_t, data_t);
ttbfp_matmul_prepacked(TTBFP_BF8_B, A, K, exps_t, data_t, M, K, N, C, N, 1);
```

`ttbfp_matmul_bf16` takes A as bf16 bit patterns (`uint16_t`).

### Python

```python
import numpy as np, ttbfp

w = np.random.randn(4096, 4096).astype(np.float32)
exps, data = ttbfp.pack(w, ttbfp.BF8_B)       # or "bf8_b"
w_hat = ttbfp.unpack(exps, data, ttbfp.BF8_B, cols=4096)

a = np.random.randn(1, 4096).astype(np.float32)
c = ttbfp.matmul(a, exps, data, ttbfp.BF8_B, cols=4096)

# Faster: prepack B once into the block-major layout.
exps_t, data_t = ttbfp.pack_transposed(w, ttbfp.BF8_B)
c = ttbfp.matmul_prepacked(a, exps_t, data_t, ttbfp.BF8_B, cols=4096)
```

`ttbfp.pack` returns `exps (rows, nblk)` and `data (rows, nblk*16 or nblk*8)`.
`ttbfp.tile_pack(tile, fmt)` / `ttbfp.tile_unpack(blob, fmt)` handle the exact
TT L1 tile layout.

### PyTorch (CPU)

```python
import torch
from ttbfp import torch_ops

w = torch.randn(4096, 4096)
exps, data = torch_ops.pack(w, "bf8_b")
y = torch_ops.unpack(exps, data, "bf8_b", cols=4096)
c = torch_ops.matmul(torch.randn(1, 4096), exps, data, "bf8_b", cols=4096)
```

`torch_ops.pack_transposed` / `torch_ops.matmul_prepacked` are the
block-major equivalents.  All of these are zero-copy wrappers around the
NumPy API (CPU tensors only).

### CUDA

`include/ttbfp_cuda.h` + `src/ttbfp_cuda.cu` implement the same quantize,
dequantize, and fused-matmul kernels for NVIDIA GPUs.  Device pointers and a
`cudaStream_t` are passed explicitly; there is no allocation inside the
library:

```c
ttbfp_cuda_quantize(TTBFP_BF8_B, d_src, rows, cols, d_exps, d_data, stream);
ttbfp_cuda_dequantize(TTBFP_BF8_B, d_exps, d_data, rows, cols, d_dst, 1, stream);
ttbfp_cuda_matmul(TTBFP_BF8_B, d_A, K, d_exps, d_data, M, K, N, d_C, N, 1, stream);
```

The CUDA source is compile-checked (PTX) in this repository, but it has not
been run on a GPU in this environment; the CPU implementation is the tested
reference.

---

## 6. Numerics and performance notes

### Kernel structure

Each microkernel computes `MR` rows of A against `J` consecutive 16-wide B
blocks for the whole K range:

* A is staged through a small float panel, so its loads are sequential and
  each A element is broadcast once and reused `J` times.
* `J` B blocks are decoded per k, so a 64-byte cache line of packed B is
  consumed completely.
* Accumulators stay in registers: `MR * J <= 16` zmm, so 16x1, 8x2 and 4x4
  are the three shapes used.
* The k order of accumulation never changes, so scalar, AVX2, AVX-512, every
  tile shape and every thread count are **bitwise identical**.  That is
  asserted by the test suite (including against the float64 reference).

Tile shapes chosen at runtime.  An `MR x J` tile spends two resources that
trade off against each other, with `MR * J <= 16` set by the accumulator
registers:

* decode work `~ 1/MR` - one B block is decoded per `MR` FMAs;
* A traffic `~ 1/J` - A is re-staged once per column panel.

Which one binds depends on the decoder, so the shape is chosen per decoder:

| layout | decoder | M | row tile | col blocks |
|---|---|---|---|---|
| row-major | sign-magnitude | any | 4 | 4 |
| block-major | sign-magnitude | M >= 16 | 16 | 1 |
| block-major | sign-magnitude | 8 <= M < 16 | 8 | 1 |
| block-major | sign-magnitude | M < 8 | 4 | 4 |
| block-major | int8 | M >= 4 | 4 | 4 |
| block-major | int8 | M < 4 | M | 4 |

Sign-magnitude decode costs ~7 instructions, so it wants the tallest row
tile and lets `J = 1` carry the A traffic.  The int8 decode is 3
instructions, so `4 x 4` wins at every M.  `TTBFP_MRT` / `TTBFP_JM` override
the shape (for tuning on a particular CPU); `MR * J` is clamped to 16.

### Measured speedup over the previous release

Single thread, `K = N = 4096`, process CPU time, same process, same data:

| format | M | before | row-major | block-major | int8 | speedup |
|---|---|---|---|---|---|---|
| bf8_b | 1 | 4.60 ms | 2.22 ms | 1.34 ms | **0.79 ms** | **5.8x** |
| bf8_b | 4 | 3.87 ms | 2.74 ms | 2.21 ms | **1.67 ms** | **2.3x** |
| bf8_b | 16 | 15.55 ms | 10.76 ms | 7.58 ms | **6.52 ms** | **2.4x** |
| bf8_b | 64 | 62.12 ms | 43.48 ms | 30.27 ms | **26.10 ms** | **2.4x** |
| bf4_b | 1 | 2.84 ms | 1.92 ms | 1.81 ms | **0.80 ms** | **3.6x** |
| bf4_b | 4 | 3.41 ms | 2.75 ms | 2.59 ms | **1.67 ms** | **2.0x** |
| bf4_b | 16 | 13.70 ms | 10.78 ms | 7.35 ms | **6.47 ms** | **2.1x** |
| bf4_b | 64 | 54.73 ms | 43.25 ms | 29.31 ms | **25.90 ms** | **2.1x** |

With 8 threads and `bench/ab_compare.py`, the int8 layout against the original
row-major release gives 4.5x / 2.2x / 2.2x / 2.4x for bf8_b at M = 1 / 4 /
16 / 64, and 3.5x / 2.1x / 2.2x / 2.2x for bf4_b.  All variants are bitwise
equal.

`bf2_b` (`Bfp2_b`) was added after the original release, so it has no "before"
column.  Same host, same shape:

| format | M | block-major | int8 |
|---|---|---|---|
| bf2_b | 1 | 2.33 ms | **0.80 ms** |
| bf2_b | 4 | 3.70 ms | **1.60 ms** |
| bf2_b | 16 | 8.08 ms | **6.50 ms** |
| bf2_b | 64 | 30.89 ms | **25.74 ms** |

The int8 layout stores 16 bytes per block for every format, so its cost is the
same for all three; the sign-magnitude layout shrinks with the datum and its
decode grows.

### Where the time goes

At M >= 16 the kernel is FMA-bound.  The pure FMA path (no decode, no scale,
no broadcast) runs a 16x4096x16 panel in ~0.0178 ms, i.e. ~118 GFLOP/s, or
~18.2 ms for the whole M=64, K=N=4096 problem.  The int8 kernel takes
~26.1 ms, so it is at ~70% of that floor.

The 30% gap is fully accounted for, per k step of a 4x4 tile:

* 16 FMA (8 cycles at two 512-bit FMA pipes);
* 4 `vpmovsxbd` + 4 `vcvtdq2ps` + 4 `vmulps` - 8 vector uops that share the
  same two FP pipes, which is exactly the observed 1.26x over the FMA-only
  panel (0.0225 vs 0.0178 ms);
* 4 broadcast-load FMAs (folded into the FMA, so they cost load slots, not
  issue slots).

Removing the scale multiply entirely (`vmulps` deleted) gives 0.0192 ms, so
the remaining decode+scale overhead is ~8%.  That is the headroom, and it
cannot be recovered without changing the arithmetic:

* **fp32 B prepack** (`ttbfp_dequantize` then a plain fp32 GEMM) removes the
  decode and scale, but B grows 4x and the kernel becomes DRAM-bound at ~95
  GFLOP/s - no better than the int8 path, and much worse at small M.
* **`vpsignb` sign-magnitude decode** is 2 uops shorter but slower in
  practice (higher latency on the shuffle port).
* **Exponent-add scaling** (`vpaddd` on the float exponent instead of
  `vmulps`) helps only the 16x1 shape (4%), because Zen 5 vector integer
  adds share the FP pipes.
* **J = 8** halves A traffic but adds 4 more decodes per 16 FMAs; measured
  28% slower.
* **`kAPanel` 64/128, prefetch, register-forced broadcasts, k unrolling** -
  all measured neutral or worse on this host.

Going below the FMA floor would require integer dot products (`vpdpbusd`),
which change the fp32 accumulation order and would break bitwise parity with
tt-llk.  That trade is deliberately not taken.

### Measuring

`time.perf_counter` is unreliable on a busy host (this development box runs at
load average ~28, and wall-clock samples quantize to ~1 ms).  The benchmark
uses `time.process_time`, which is stable.  Two OpenMP settings matter:

* `OMP_WAIT_POLICY=passive` - otherwise idle worker threads spin and inflate
  CPU time (and compete with the next kernel).
* `OMP_NUM_THREADS` - the kernels parallelize over independent
  `(row tile x column panel)` output tiles, with no cross-thread reduction.
  More threads than available cores is slower, sometimes much slower.

### Other notes

* `TTBFP_FORCE_SCALAR=1` disables both SIMD paths; `TTBFP_FORCE_AVX2=1`
  disables AVX-512 (both useful for debugging and for parity tests).
* Packing is the slow path (a few GB/s of input); prepacking is a single
  streaming pass.
* The fused matmul never materializes fp32 weights, which is why it beats
  "dequantize then BLAS" by 2-20x for M <= 64.

## 7. Limitations / roadmap

* All three 8-bit-exponent `_b` variants are implemented (`bf8_b`, `bf4_b`,
  `bf2_b`).  The 5-bit-exponent `ttbf8a`/`ttbf4a`/`ttbf2a` variants are
  straightforward extensions of the same block quantizer.
* No sub-tile geometries (TT pads sub-tiles to 16 exponent bytes; this
  library assumes 16-element blocks and a full 32×32 tile for tile helpers).
* Special values (Inf/NaN) follow the reference packer's bit handling but are
  not a supported arithmetic domain; keep masks in bf16/fp32.
* The CPU library is the tested implementation.  The optional CUDA kernels in
  `src/ttbfp_cuda.cu` are compile-checked but have not been run on a GPU here.
  (There is no "CUDA kernel that runs on CPU" — this library is the CPU
  implementation of the TT kernels.)

---

## 8. References

* Tenstorrent ISA, floating-point bit patterns (BFP8/BFP4/BFP2, sign-magnitude,
  shared 8-bit exponent):
  https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/FloatBitPatterns.md
* Tenstorrent tt-metal data-format tech report:
  https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/data_formats/data_formats.md
* tt-metal `DataFormat` enum (Blackhole):
  https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix_types.h
* tt-llk pack/unpack reference helpers:
  https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tests/python_tests/helpers
* Blackhole P300C card:
  https://docs.tenstorrent.com/aibs/blackhole/p300.html
* Blackhole supported precision formats:
  https://docs.tenstorrent.com/aibs/blackhole/
* OCP Microscaling Formats (MX) v1.0:
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
* OCP 8-bit Floating Point Specification (OFP8):
  https://www.opencompute.org/documents/ocp-fp8-specification-1-0-pdf
