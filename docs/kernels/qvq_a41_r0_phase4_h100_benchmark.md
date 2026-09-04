# QVQ A41/R0 Phase 4: production H100 benchmark

This benchmark measures the complete production projection path introduced in
Phase 4: shared `SU`/input Hadamard, grouped P32 decode and WGMMA, child-local
output Hadamard/`SV`/bias, output cast, and the sibling coordinator.  It is not
an inner-kernel-only measurement.

## Result

Across W2/W2.5/W3/W3.5, QKV and gate/up, and M1/M2/M4/M8/M16, every one of the
40 grouped cases is faster than the current independent full-operation QVQ
path.  The group-wise geometric speedup is **1.684x**, or approximately
**40.6% lower latency**.  Summing QKV and gate/up as two model sites gives a
**1.627x** geometric speedup, approximately **38.5% lower latency**, exceeding
the 25% Phase-4 target.

This run includes `origin/main` commit `b75ed8ad` and its Hopper low-address/PGC
fusion.  Against the immediately preceding Phase-4 H100 run, all 20 combined
M/rate rows improved.  Their geometric absolute-latency improvement is
**1.072x** (about **6.7% lower latency**): QKV improved **1.098x** and gate/up
improved **1.057x**.

| Group | Phase 4 vs plain QVQ | Latency reduction | Phase 4 vs Marlin W4 | Phase 4 vs Machete W4 | Better than prior run |
|---|---:|---:|---:|---:|---:|
| QKV | 2.218x | 54.9% | 1.245x | 0.796x | 20/20 |
| gate/up | 1.280x | 21.9% | 0.193x | 0.390x | 20/20 |

A comparator ratio above 1 means QVQ is faster; below 1 means the W4 GPTQ
baseline is faster.  W2–W3.5 QVQ and W4 GPTQ reconstruct different weights,
so these are execution-efficiency comparisons, not output-equality claims.

## Combined model-site matrix

Each row sums the three QKV child calls and the two gate/up child calls.
Aggregate N is shown as `3072 + 16384` so the table does not imply one
mathematical GEMM.  “Better” compares the grouped latency against the preceding
Phase-4 H100 benchmark recorded before the latest `origin/main` merge.  `No`
therefore identifies an absolute-latency regression, independently of whether
the current grouped path still beats plain QVQ.

| Rate | M | M x K x aggregate N | Plain QVQ us | Phase 4 us | Speedup | Marlin W4 us | vs Marlin | Machete W4 us | vs Machete | Better than last benchmark | Effective TFLOP/s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| W2 | 1 | 1 x 2048 x (3072 + 16384) | 196.922 | 119.664 | 1.646x | 63.718 | 0.532x | 64.494 | 0.539x | Yes | 0.666 |
| W2 | 2 | 2 x 2048 x (3072 + 16384) | 196.330 | 119.747 | 1.640x | 72.526 | 0.606x | 64.332 | 0.537x | Yes | 1.331 |
| W2 | 4 | 4 x 2048 x (3072 + 16384) | 194.595 | 120.137 | 1.620x | 73.741 | 0.614x | 64.226 | 0.535x | Yes | 2.653 |
| W2 | 8 | 8 x 2048 x (3072 + 16384) | 194.106 | 120.965 | 1.605x | 65.274 | 0.540x | 63.562 | 0.525x | Yes | 5.270 |
| W2 | 16 | 16 x 2048 x (3072 + 16384) | 182.002 | 116.298 | 1.565x | 70.634 | 0.607x | 64.041 | 0.551x | Yes | 10.964 |
| W2.5 | 1 | 1 x 2048 x (3072 + 16384) | 196.827 | 118.951 | 1.655x | 63.718 | 0.536x | 64.494 | 0.542x | Yes | 0.670 |
| W2.5 | 2 | 2 x 2048 x (3072 + 16384) | 196.199 | 119.366 | 1.644x | 72.526 | 0.608x | 64.332 | 0.539x | Yes | 1.335 |
| W2.5 | 4 | 4 x 2048 x (3072 + 16384) | 195.415 | 119.930 | 1.629x | 73.741 | 0.615x | 64.226 | 0.536x | Yes | 2.658 |
| W2.5 | 8 | 8 x 2048 x (3072 + 16384) | 194.910 | 120.174 | 1.622x | 65.274 | 0.543x | 63.562 | 0.529x | Yes | 5.305 |
| W2.5 | 16 | 16 x 2048 x (3072 + 16384) | 182.560 | 115.590 | 1.579x | 70.634 | 0.611x | 64.041 | 0.554x | Yes | 11.031 |
| W3 | 1 | 1 x 2048 x (3072 + 16384) | 195.976 | 118.753 | 1.650x | 63.718 | 0.537x | 64.494 | 0.543x | Yes | 0.671 |
| W3 | 2 | 2 x 2048 x (3072 + 16384) | 195.443 | 118.817 | 1.645x | 72.526 | 0.610x | 64.332 | 0.541x | Yes | 1.341 |
| W3 | 4 | 4 x 2048 x (3072 + 16384) | 195.675 | 119.210 | 1.641x | 73.741 | 0.619x | 64.226 | 0.539x | Yes | 2.674 |
| W3 | 8 | 8 x 2048 x (3072 + 16384) | 195.734 | 119.585 | 1.637x | 65.274 | 0.546x | 63.562 | 0.532x | Yes | 5.331 |
| W3 | 16 | 16 x 2048 x (3072 + 16384) | 182.836 | 115.440 | 1.584x | 70.634 | 0.612x | 64.041 | 0.555x | Yes | 11.045 |
| W3.5 | 1 | 1 x 2048 x (3072 + 16384) | 197.042 | 118.476 | 1.663x | 63.718 | 0.538x | 64.494 | 0.544x | Yes | 0.673 |
| W3.5 | 2 | 2 x 2048 x (3072 + 16384) | 196.146 | 118.759 | 1.652x | 72.526 | 0.611x | 64.332 | 0.542x | Yes | 1.342 |
| W3.5 | 4 | 4 x 2048 x (3072 + 16384) | 195.505 | 119.699 | 1.633x | 73.741 | 0.616x | 64.226 | 0.537x | Yes | 2.663 |
| W3.5 | 8 | 8 x 2048 x (3072 + 16384) | 195.914 | 119.650 | 1.637x | 65.274 | 0.546x | 63.562 | 0.531x | Yes | 5.328 |
| W3.5 | 16 | 16 x 2048 x (3072 + 16384) | 183.308 | 115.242 | 1.591x | 70.634 | 0.613x | 64.041 | 0.556x | Yes | 11.064 |

Effective TFLOP/s uses dense-equivalent logical work,

\[
\frac{2M K \sum_i N_i}{t},
\]

and therefore does not credit P32 decoder integer work or padded M16 rows.

## Why gate/up remains behind

The Phase-3 W3 inner grouped kernel measured about 31 microseconds for gate/up.
The Phase-4 complete gate/up operation measures about 73–76 microseconds.  The
difference is primarily the two independent 8192-wide output Hadamards,
`SV` application, output casting, and allocation traffic.  The input transform
is already shared, and the inner grouped kernel is no longer the dominant
gate/up cost.

QKV behaves differently.  Its narrow K/V launch tails were expensive, while
the output recovery tensors are much smaller.  The full QKV path is 2.218x
faster than independent QVQ and 1.245x faster than summed W4 Marlin.  It remains
about 1.26x slower than summed W4 Machete.

The next performance phase should therefore target output-side architecture
folding or fused output recovery for gate/up, not another sibling coordinator
rewrite.

## Method and controls

- Device: exclusive physical H100
  `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, SM 9.0, 132 SMs,
  102,077,169,664 bytes. `CUDA_VISIBLE_DEVICES=1` hid the H200.
- Shapes: Llama 3.2 1B QKV (`K=2048`, child `N=2048/512/512`) and
  gate/up (`K=2048`, child `N=8192/8192`).
- M: 1, 2, 4, 8, and 16. QVQ uses the production zero-padded M16 Hopper path;
  W4 baselines use their native logical M.
- Timing: one warmed CUDA Graph per complete sibling group. CUDA events bracket
  20 replays per sample; values are medians of 50 samples after 20 warmups.
  CPU and container scheduling are outside the measured interval.
- Plain QVQ and Phase 4 use identical children, activations, transforms,
  payloads, bank selectors, alternative-bank IDs, and split-1 policy. Every
  grouped output is bit-exact to its independent child output.
- Marlin and Machete use symmetric GPTQ W4, group size 128, FP16 activations,
  and independent child shapes. Both use the same synthetic packed W4 source.
- The recorded source fingerprint is stored with the raw JSON and was checked
  again after the matrix completed.

## Reproduction

- Driver: `scripts/benchmark_qvq_a41_phase4_production.py`
- Raw result: `artifacts/a41_phase4_h100/production_grouped_vs_baselines.json`
- Runtime design: `docs/kernels/qvq_a41_r0_phase4_runtime.md`
