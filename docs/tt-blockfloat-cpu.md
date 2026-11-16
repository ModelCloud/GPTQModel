# Tenstorrent block floats on CPU (`ttbfp`)

`gptqmodel_ext/ttbfp/` provides C++ kernels and NumPy bindings that reproduce
Tenstorrent's block-float tensor formats on ordinary CPUs, so models and
kernels that use them can be tested on non-Tenstorrent hardware. See
[`gptqmodel_ext/ttbfp/README.md`](../gptqmodel_ext/ttbfp/README.md).

## Dtype naming

Tenstorrent block floats are **not** an industry-standard format.  Block
floating point is an old, generic idea, but the TT encoding differs from the
standardized block-scaled formats (OCP MX and friends):

| Property | TT `bf8_b` / `bf4_b` / `bf2_b` | OCP MXFP8 / MXFP4 |
|---|---|---|
| Block size | 16 elements | 32 elements |
| Shared exponent | 8 bits, bias 127 | 8 bits, bias 127 |
| Datum encoding | sign-magnitude | sign-magnitude |
| Element bits | 8 / 4 / 2 | 8 / 4 (E4M3/E2M1) |
| Exponent in datum | none (block shared) | none (block shared) |

The canonical names are `bf8_b`, `bf4_b` and `bf2_b`, matching
[`hw/dtypes.md`](../hw/dtypes.md).  The library's older
`ttbf8`/`ttbf4`/`ttbf2` and `BFP8`/`BFP4`/`BFP2` spellings remain
deprecated aliases.

The P300C (Blackhole) native formats covered here are `Bfp8_b` (BLOCKFP8),
`Bfp4_b` (BLOCKFP4) and `Bfp2_b` (BLOCKFP2); `README.md` section 2 in the extension documents the full
native-dtype table.

## Bit-exactness invariants

Two invariants drive every implementation decision:

1. **The quantizer matches tt-llk bit-for-bit.**  fp32→bf16 truncation, block
   max exponent, guard-bit round-to-nearest/ties-away, magnitude truncation for
   `bf4_b`, sign dropped when magnitude is zero, and FTZ below `FLT_MIN` when
   `e < 7`.  `tools/verify_against_tt.py` checks 200 seeds x 4 distributions x
   3 formats against the verbatim upstream helpers.
2. **Every execution path is bitwise identical.**  Scalar, AVX2, AVX-512,
   sign-magnitude vs int8 datums, every tile shape and every thread count
   accumulate over `k` in the same order.  The test suite asserts this, and
   `bench/ab_compare.py` refuses to report a speedup if two builds disagree.

`ftz=False` is honoured even when the calling thread has ambient FTZ/DAZ
enabled: the CPU kernels clear MXCSR FTZ/DAZ for the duration of the call
(per OpenMP worker) and restore it afterwards, so genuinely subnormal weights
survive both the scaling multiply and the FMA's DAZ input check.  `ftz=True`
keeps the explicit flush and does not change the ambient mode.

## Dispatch flow

```mermaid
flowchart TD
    A["ttbfp_matmul_prepacked<br/>sign-magnitude B"] --> D["matmul_driver"]
    B["ttbfp_matmul_prepacked_s8<br/>int8 kernel-native B (bf8_b)"] --> D
    D --> E["prepacked_tile(M, s8)"]
    E --> F{"decoder"}
    F -->|"SmDec (7 uops/block)"| G["MR=16 J=1 (M>=16)<br/>MR=8 J=1 (M=8)<br/>MR=4 J=4 (M&lt;8)"]
    F -->|"S8Dec (bf8_b only; 3 uops/block)"| H["MR=4 J=4 (all M)"]
    G --> I["mm_run &rarr; mm_panel"]
    H --> I
    I --> J{"have_avx512()"}
    J -->|yes| K["mm_panel_avx512"]
    J -->|no| L["mm_panel_avx2 / mm_panel_scalar"]
```

The tile shape only decides which B blocks share an A broadcast; it never
changes the `k` order, so every branch above is bitwise equal.

## Layouts

| Layout | Exponent index | Datum index | Use |
|---|---|---|---|
| flat (row-major) | `k*nblk + j` | `(k*nblk + j)*db` | default API, no prepack |
| block-major | `j*K + k` | `(j*K + k)*db` | recommended for matmul |
| int8 block-major | `j*K + k` | `(j*K + k)*16` | bf8_b only; fastest at every M |

`db` is 16 bytes for `bf8_b`, 8 for `bf4_b` and 4 for `bf2_b`.  The int8 layout stores
`sign*magnitude` as a signed int8, which turns the decode into
`vpmovsxbd` + `vcvtdq2ps` + `vmulps`; the decoded values are unchanged.
It is bf8_b-only: bf4_b/bf2_b would expand 2x/4x, so they stay in the packed
block-major layout.

`cols` may trim only the final partial block.  The flat layout's row stride is
`ceil(cols/16) * db`, so a `cols` that drops a whole block would change the
stride; the Python wrappers reject it and ask for a repack.  The block-major
layouts are indexed by block and do not have this constraint.

## Where the time goes

Single thread, `K = N = 4096`, process CPU time:

| format | layout | M=1 | M=4 | M=16 | M=64 |
|---|---|---|---|---|---|
| `bf8_b` | int8 | 0.80 ms | 1.60 ms | 6.48 ms | 25.64 ms |
| `bf4_b` | block-major | 1.81 ms | 2.59 ms | 7.35 ms | 29.31 ms |
| `bf2_b` | block-major | 2.33 ms | 3.70 ms | 8.08 ms | 30.89 ms |

The pure FMA path (no decode, no scale, no broadcast) runs a 16x4096x16 panel
in ~0.0178 ms, i.e. ~118 GFLOP/s or ~18.2 ms for the whole M=64 problem.  The
bf8_b int8 kernel is at ~70% of that floor.  The gap is accounted for per `k` step
of a 4x4 tile: 16 FMA (8 cycles at two 512-bit FMA pipes) plus 4
`vpmovsxbd` + 4 `vcvtdq2ps` + 4 `vmulps`, eight vector uops that share the
same FP pipes.  Deleting the scale multiply alone gives 0.0192 ms, so the
remaining headroom is ~8%.

Going below the FMA floor would require integer dot products (`vpdpbusd`),
which change the fp32 accumulation order and would break bitwise parity with
tt-llk.  That trade is deliberately not taken.

Rejected alternatives, all measured on the same host: fp32 B prepack (DRAM
bound, no better), `vpsignb` decode (slower), exponent-add scaling (helps only
16x1, by 4%), `J=8` (28% slower), `kAPanel` 64/128, prefetch, and removing the
FTZ branch (free).  Cache blocking over `k` would reach the floor but changes
the accumulation association.

## Build and test

```bash
cd gptqmodel_ext/ttbfp
make
make test
OMP_WAIT_POLICY=passive OMP_NUM_THREADS=8 \
    python bench/ab_compare.py --old baseline.so --new ./libttbfp.so
```

`OMP_WAIT_POLICY=passive` is required for meaningful back-to-back numbers:
GNU OpenMP spins after a parallel region, so idle workers inflate CPU time and
steal issue slots from the next kernel.  It changes performance only, never
results.
