---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: ttbfp-perf
description: Measure, profile, or optimize the ttbfp CPU kernels (Tenstorrent bf8_b/bf4_b/bf2_b block floats) under gptqmodel_ext/ttbfp/ - running the suite, reporting throughput, A/B comparing builds, or editing gptqmodel_ext/ttbfp/src/ttbfp_cpu.cpp. Use when a task concerns ttbfp performance, thread or tile tuning, or a reported slowdown. Not for questions about the TT block-float bit layout or dtype naming; those are answered in gptqmodel_ext/ttbfp/README.md.
---

# ttbfp performance work

## At a glance

Tune tile shape, blocking or decode in `gptqmodel_ext/ttbfp/src/ttbfp_cpu.cpp`,
never the numerics. These fused decode+GEMM kernels use 16-element blocks.
Compare builds in one process on the same buffers; check the measured dead
ends below before proposing an optimization.

## Measure this way

Use process CPU time (wall-clock scheduler slices can exceed the kernel time),
`OMP_WAIT_POLICY=passive` (spinning idle workers distort CPU time and contend
with the next kernel) and no more `OMP_NUM_THREADS` than cores you own. Report
both OpenMP settings with every result.

```bash
OMP_WAIT_POLICY=passive OMP_NUM_THREADS=8 \
  python gptqmodel_ext/ttbfp/bench/ab_compare.py \
  --old /path/to/baseline.so --new gptqmodel_ext/ttbfp/libttbfp.so
```

`gptqmodel_ext/ttbfp/bench/ab_compare.py` loads both builds, runs them interleaved on identical
inputs, reports row-major / block-major / int8 timings, and fails if the two
builds disagree bitwise. For absolute numbers use `gptqmodel_ext/ttbfp/bench/bench_matmul.py`
with the same environment.

## Do not break these invariants

- **Bitwise parity across paths.** Scalar, AVX2, AVX-512, sign-magnitude vs
  int8 datums, every tile shape, and every thread count must produce
  identical bits. Each output accumulates over k in the same order; keep it
  that way. New kernels must be added to the parity tests in
  `gptqmodel_ext/ttbfp/tests/test_ttbfp.py`, not just to a tolerance check.
- **Quantizer stays bit-exact with tt-llk.** After touching encode/decode,
  run `gptqmodel_ext/ttbfp/tools/verify_against_tt.py` (needs torch); it compares against the
  real upstream packers.
- **Names are `bf8_b` / `bf4_b` / `bf2_b`.** They are Tenstorrent-specific,
  not OCP MX. Do not rename them to `bf8`/`bf4`/`bf2`.

## Where the time goes

Read the [kernel structure and tile table](../../../gptqmodel_ext/ttbfp/README.md#kernel-structure)
and the [measured FMA floor and dead ends](../../../gptqmodel_ext/ttbfp/README.md#where-the-time-goes)
before tuning. Those are Zen 5 measurements, not portable targets. Re-derive
the pure FMA floor on your host; more speedup than the remaining measured
decode/scale cost calls the measurement into question.

- **Small M:** decode dominates. `ttbfp_convert_s8` / `prepack_s8` turns
  sign-magnitude into signed INT8 without changing values (~1.7× at M=1).
  Only bf8_b may use this layout; bf4_b/bf2_b would expand 2×/4×.
- **Large M:** balance decode per FMA against A-panel traffic, which scales
  as `1/J`; choose tiles per decoder, not merely per layout. Keep A staging:
  direct `A[r*lda+k]` reads were ~1.9× slower at 16x1 (cache lines/register
  pressure), neutral at 4x4.
- Software-pipeline decode only for `J <= 2` (3–5% at 16x1/8x2); at
  `J = 4` register pressure made M=4 ~15% slower.

### Measured dead ends (do not retry)

The linked floor measurements cover FP32 B prepack, `vpsignb`, exponent-add
scaling, `J = 8`, larger `kAPanel`, prefetch, register broadcasts and
k unrolling. Removing the predicted FTZ branch was neutral too. Do not block
the reduction over k: reassociating the accumulation breaks bitwise parity.

Tile shape can be overridden for experiments without a rebuild. These
`TTBFP_*` variables are benchmark and debugging knobs of the standalone
library; GPT-QModel code must never set or depend on them (see the
environment-variable rule in `AGENTS.md`):

```bash
TTBFP_MRT=8 TTBFP_JM=2 python gptqmodel_ext/ttbfp/bench/bench_matmul.py  # MR=8, J=2
TTBFP_FORCE_AVX2=1 ...                           # skip AVX-512
TTBFP_FORCE_SCALAR=1 ...                         # reference path
```

## Commands

```bash
make -C gptqmodel_ext/ttbfp            # libttbfp.so
make -C gptqmodel_ext/ttbfp test       # full suite, includes bitwise parity
make -C gptqmodel_ext/ttbfp bench      # microbenchmarks
make -C gptqmodel_ext/ttbfp cuda       # optional CUDA library (needs nvcc + GPU)
```

The library never calls `omp_set_*`; callers own the pool. If you add a
parallel region, note the wait-policy requirement in its comment, as the
existing loops do.
