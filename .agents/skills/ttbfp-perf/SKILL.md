---
name: ttbfp-perf
description: Measure, profile, or optimize the ttbfp CPU kernels (Tenstorrent ttbf8/ttbf4/ttbf2 block floats) under gptqmodel_ext/ttbfp/ - running the suite, reporting throughput, A/B comparing builds, or editing gptqmodel_ext/ttbfp/src/ttbfp_cpu.cpp. Use when a task concerns ttbfp performance, thread or tile tuning, or a reported slowdown. Not for questions about the TT block-float bit layout or dtype naming; those are answered in gptqmodel_ext/ttbfp/README.md.
---

# ttbfp performance work

The kernels are fused decode+GEMM microkernels over 16-element block-float
blocks. Performance work here means changing tile shape, blocking, or the
decoder - never the numerics.

## Measure this way

These four rules decide whether a number means anything. Skip one and the
result is noise.

1. **`OMP_WAIT_POLICY=passive`.** GNU OpenMP spins after a parallel region
   ends (large default `GOMP_SPINCOUNT`). Back-to-back kernels then burn a
   full core per idle worker: an 8-thread run reports roughly 8x the CPU
   time, and the spinners steal issue slots and cache from the next kernel.
   Passive wait makes workers sleep and CPU time track real work. It changes
   performance only, never results.
2. **Cap `OMP_NUM_THREADS` at cores you actually own.** Oversubscription is
   much slower than fewer threads, not slightly slower.
3. **Time with process CPU time, not wall clock.** On a busy shared host
   wall-clock samples quantize to the scheduler slice (~1 ms here), which is
   coarser than the kernel for small M.
4. **A/B in one process on the same buffers.** Host load and page state
   dominate cross-run comparisons.

```bash
OMP_WAIT_POLICY=passive OMP_NUM_THREADS=8 \
  python gptqmodel_ext/ttbfp/bench/ab_compare.py \
  --old /path/to/baseline.so --new gptqmodel_ext/ttbfp/libttbfp.so
```

`gptqmodel_ext/ttbfp/bench/ab_compare.py` loads both builds, runs them interleaved on identical
inputs, reports row-major / block-major / int8 timings, and fails if the two
builds disagree bitwise. For absolute numbers use `gptqmodel_ext/ttbfp/bench/bench_matmul.py`
with the same environment.

Report the thread count and wait policy alongside any number; a throughput
figure without them is not reproducible.

## Do not break these invariants

- **Bitwise parity across paths.** Scalar, AVX2, AVX-512, sign-magnitude vs
  int8 datums, every tile shape, and every thread count must produce
  identical bits. Each output accumulates over k in the same order; keep it
  that way. New kernels must be added to the parity tests in
  `gptqmodel_ext/ttbfp/tests/test_ttbfp.py`, not just to a tolerance check.
- **Quantizer stays bit-exact with tt-llk.** After touching encode/decode,
  run `gptqmodel_ext/ttbfp/tools/verify_against_tt.py` (needs torch); it compares against the
  real upstream packers.
- **Names are `ttbf8` / `ttbf4`.** They are Tenstorrent-specific, not OCP MX.
  Do not rename them to `bf8`/`bf4`.

## Where the time goes

Start from the FMA floor, not from the current number.  A pure FMA loop
(no decode, no scale, no broadcast) runs a 16x4096x16 panel in ~0.0178 ms on
this host, i.e. ~118 GFLOP/s or ~18.2 ms for the whole M=64, K=N=4096
problem.  The int8 kernel takes ~26.1 ms, so it is at ~70% of that floor,
not at it.  The gap is fully accounted for and worth re-deriving before
trying anything:

* 16 FMA per k at two 512-bit FMA pipes = 8 cycles;
* 4 `vpmovsxbd` + 4 `vcvtdq2ps` + 4 `vmulps` in a 4x4 tile - 8 vector uops
  that share the same two FP pipes (Zen 5 vector integer ops do too).  That
  predicts 1.5x over the floor; measured 1.26x (0.0225 vs 0.0178 ms);
* the 4 broadcast-load FMAs fold into the FMA, costing load slots not issue
  slots.

Deleting the scale multiply alone (`vmulps`) gives 0.0192 ms, so the whole
remaining overhead is ~8%.  Anything that claims more is measuring noise.

Two regimes:

- **Small M (decode-bound).** Per k there is one decode per column block and
  only `MR` FMAs, so decode dominates.  The fix is a cheaper datum, not
  blocking: `ttbfp_convert_s8` / `prepack_s8` store `sign*magnitude` as a
  signed int8, so the decode is `vpmovsxbd` + `vcvtdq2ps` + `vmulps` instead
  of 7-12 instructions.  Worth ~1.7x at M=1.  This mirrors ATen's
  `_weight_int4pack_mm` / `_weight_int8pack_mm`; values are unchanged, so
  parity still holds.
- **Large M (FMA-bound).** The decode is still on the critical path, so the
  shape that minimizes it (tallest `MR`) is not the fastest one: A-panel
  traffic scales as `1/J` and can cost more than the decode.  That is why
  the tile shape is chosen per decoder, not per layout:

  | decoder | M | shape | why |
  |---|---|---|---|
  | sign-magnitude | M >= 16 | 16x1 | ~7-uop decode; one decode per 16 FMA |
  | sign-magnitude | M == 8 | 8x1 | same, but only 8 rows exist |
  | sign-magnitude | M < 8 | 4x4 | too few rows to amortize a decode |
  | int8 | any | 4x4 | 3-uop decode; cutting A traffic 4x wins 1.2x |

  The shape only decides which B blocks share an A broadcast, so all shapes
  stay bitwise equal.

Software-pipeline the decode (decode k+1 during k's FMAs) only for `J <= 2`.
It gives 3-5% at 16x1 and 8x2, but at `J = 4` the extra live B vectors cost
more registers than the overlap buys (~15% slower at M=4).

### Measured dead ends (do not retry)

All bitwise-equal or neutral-to-worse on this host, in one process, with
`OMP_WAIT_POLICY=passive`:

* fp32 B prepack (dequantize then plain GEMM): removes decode and scale but
  grows B 4x and becomes DRAM-bound at ~95 GFLOP/s - no better.
* `vpsignb` sign-magnitude decode: 2 uops shorter, slower (shuffle port).
* Exponent-add scaling (`vpaddd` on the float exponent): helps only 16x1
  (4%); Zen 5 vector integer adds share the FP pipes.
* `J = 8` (2x8): halves A traffic, adds 4 decodes per 16 FMAs; 28% slower.
* `kAPanel` 64 or 128 (instead of 32): neutral or worse.
* Prefetching B, forcing a register broadcast, k unrolling, removing the
  FTZ branch: all neutral (FTZ branch is perfectly predicted and free).
* Cache blocking over k: changes the accumulation association, so it breaks
  bitwise parity. Not allowed, not just slow.

The A staging panel is load-bearing: reading `A[r*lda+k]` directly is ~1.9x
slower at 16x1 because each k touches `MR` different cache lines and the
compiler runs out of address registers.  It is free at 4x4, where only four
rows are staged.

Tile shape can be overridden for experiments without a rebuild:

```bash
TTBFP_MRT=8 TTBFP_JM=2 python gptqmodel_ext/ttbfp/bench/bench_matmul.py  # MR=8, J=2
TTBFP_FORCE_AVX2=1 ...                           # skip AVX-512
TTBFP_FORCE_SCALAR=1 ...                         # reference path
```

## Commands

```bash
make -C gptqmodel_ext/ttbfp            # libttbfp.so
make -C gptqmodel_ext/ttbfp test       # 174 tests, includes bitwise parity
make -C gptqmodel_ext/ttbfp bench      # microbenchmarks
make -C gptqmodel_ext/ttbfp cuda       # optional CUDA library (needs nvcc + GPU)
```

The library never calls `omp_set_*`; callers own the pool. If you add a
parallel region, note the wait-policy requirement in its comment, as the
existing loops do.
