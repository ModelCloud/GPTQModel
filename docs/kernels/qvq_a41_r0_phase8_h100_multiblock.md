# QVQ A41/R0 Phase 8: exact H100 multiblock MLP recovery

Phase 8 replaces the under-parallelized single-block gate/up output recovery
for the Llama 3.2 1B intermediate width with an exact two-stage transform on
the physical H100.  The complete W2--W3.5 MLP improves by **1.159x** geometric
mean, or **13.7% lower latency**, versus the last committed Phase-7 matrix.
All 20 rate/M rows improve.

This changes only execution.  Canonical P32 payloads, quantized weights,
scales, bias, activation math, and the FP16 rounding contract are unchanged.
H200 and unsupported widths retain the existing single-block kernel.

## Why the old launch was slow

For each MLP row the gate and up projections independently recover an
8192-wide FP32 inner result into FP16.  The old paired kernel launches one
1024-thread block per projection-row:

\[
B_{old}=2M.
\]

At M=1 this exposes only two blocks to an H100 with 132 streaming
multiprocessors.  Nsight Compute measured no spills and modest per-block
shared memory, so the limiting resource was useful block count rather than
memory bandwidth or occupancy inside an active multiprocessor.

## Exact factorization

The production transform applies the Walsh-Hadamard butterfly bits in the
ascending order

\[
1,2,4,\ldots,4096.
\]

For \(N=8192=32\times256\), bits 1 through 128 never cross a contiguous
256-value tile.  Bits 256 through 4096 operate only across the 32 tile indices
for a fixed within-tile column.  Therefore the same ordered transform can be
factorized as

\[
H_{8192}=H_{32}^{(tile)}H_{256}^{(column)},
\]

where `low` runs the first eight stages independently for each tile and
`high` runs the last five stages independently for each within-tile column.
This expression describes execution order; it does not reorder the original
butterflies.

The exact per-value sequence remains:

```text
FP32 inner result
  -> emulate the historical FP16 cast unless it overflows
  -> normalize and emulate the historical FP16 rounding
  -> butterfly bits 1..128, rounding after every add/sub
  -> store an FP32 intermediate (no new rounding)
  -> butterfly bits 256..4096, rounding after every add/sub
  -> child-local SV multiply and rounding
  -> optional child-local bias add and rounding
  -> final FP16 output store
```

The intermediate workspace is FP32 because the original single-block kernel
also holds these values in FP32 after applying
`round_fp16_unless_overflow`.  Storing and loading that FP32 value is exact and
does not add a numerical boundary.

## Launch geometry and storage

The low stage launches 32 blocks per projection-row, each with 256 threads.
The high stage launches four blocks per projection-row, each with 64 threads;
one thread owns one within-tile column and its 32 tile-index values.

\[
B_{new}=2M(32+4)=72M.
\]

Thus M1 grows from 2 to 72 useful blocks.  The temporary workspace is

\[
2M\times8192\times4\ \text{bytes},
\]

or 64 KiB at M1 and 1 MiB at M16.  It is an operation-local allocator/graph
pool allocation, not checkpoint state, a module cache, or persistent model
VRAM.

The specialization budget is two fixed device kernels.  Rate, M, bias
presence, and normalization mode remain runtime values.  Production dispatch
requires all of:

- physical device name `NVIDIA H100` with compute capability 9.0;
- paired gate/up recovery;
- output width exactly 8192.

No visible CUDA device index is used as an architecture test.

## Isolated recovery result

Times are warmed CUDA Graph replays measured by CUDA events: 30 warmups, 100
samples, and 50 replays per sample.  The candidate is bit-exact to the
single-block kernel.

| M | Single p50 us | Single mean us | Single p95 us | Multiblock p50 us | Multiblock mean us | Multiblock p95 us | Speedup | Better |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 19.495 | 19.582 | 19.510 | 6.039 | 6.040 | 6.047 | 3.228x | Yes |
| 2 | 19.539 | 19.538 | 19.554 | 6.212 | 6.214 | 6.218 | 3.145x | Yes |
| 4 | 19.467 | 19.468 | 19.494 | 6.279 | 6.281 | 6.289 | 3.100x | Yes |
| 8 | 19.556 | 19.557 | 19.578 | 6.813 | 6.813 | 6.830 | 2.871x | Yes |
| 16 | 19.678 | 19.679 | 19.685 | 7.743 | 7.743 | 7.757 | 2.541x | Yes |

## Complete Llama 3.2 1B MLP

Each row includes both gate/up projections, exact SiLU and FP16 product,
down-input transform, and the down projection.  `vs` is comparator latency
divided by QVQ latency, so values below one mean the W4 baseline remains
faster.  `Better` compares against the last committed Phase-7 H100 benchmark.

| Rate | M | MKN (gate/up; down) | QVQ us | vs Marlin W4 | vs Machete W4 | Better vs last |
|---:|---:|:---|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 83.040 | 0.359x | 0.618x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 83.162 | 0.386x | 0.608x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 84.207 | 0.380x | 0.608x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 85.112 | 0.352x | 0.595x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 80.478 | 0.411x | 0.630x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 83.477 | 0.357x | 0.614x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 84.276 | 0.381x | 0.600x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 84.670 | 0.378x | 0.604x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 86.046 | 0.348x | 0.589x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 81.295 | 0.406x | 0.623x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 83.698 | 0.356x | 0.613x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 84.558 | 0.380x | 0.598x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 85.085 | 0.376x | 0.601x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 86.074 | 0.348x | 0.588x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 81.768 | 0.404x | 0.620x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 83.576 | 0.357x | 0.614x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 84.433 | 0.380x | 0.599x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 85.058 | 0.377x | 0.601x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 85.777 | 0.349x | 0.590x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 81.215 | 0.407x | 0.624x | Yes |

Geometric means:

- **1.159x** versus the previous QVQ benchmark;
- **1.687x** versus ordinary per-module QVQ;
- **0.374x** versus Marlin W4;
- **0.607x** versus Machete W4.

The dense-equivalent logical work is

\[
2M(2048\cdot8192+2048\cdot8192+8192\cdot2048).
\]

The W4 comparisons are figurative kernel-efficiency baselines; they do not
claim equal quantization quality or equal decode work.

## Correctness and reproduction

- 30 H100 cases cover M=1/2/4/8/16, three random seeds, and both output
  normalization modes; every case is repeated ten times and is bit-exact.
- A real one-layer Llama 3.2 configuration exercises the production H100 gate,
  full grouped MLP lifecycle, cached generation, repeatability, and telemetry.
- CUDA Graph replay is exercised by the production full-MLP benchmark and the
  real-layer test.

Artifacts and drivers:

- `artifacts/a41_phase8_h100/multiblock_recovery_experiment.json`
- `artifacts/a41_phase8_h100/production_mlp_vs_baselines.json`
- `scripts/benchmark_qvq_phase8_multiblock_recovery.py`
- `scripts/benchmark_qvq_a41_phase5_mlp.py`
