# QVQ A41/R0 Phase 4: production H100 benchmark

This benchmark measures the complete production projection path introduced in
Phase 4: shared `SU`/input Hadamard, grouped P32 decode and WGMMA, child-local
output Hadamard/`SV`/bias, output cast, and the sibling coordinator.  It is not
an inner-kernel-only measurement.

## Result

Across W2/W2.5/W3/W3.5, QKV and gate/up, and M1/M2/M4/M8/M16, every one of the
40 grouped cases is faster than the last comparable full-operation QVQ path.
The group-wise geometric speedup is **1.672x**, or approximately **40.2% lower
latency**.  Summing QKV and gate/up as two model sites gives a **1.622x**
geometric speedup, approximately **38.4% lower latency**, exceeding the 25%
Phase-4 target.

| Group | Phase 4 vs plain QVQ | Latency reduction | Phase 4 vs Marlin W4 | Phase 4 vs Machete W4 | Better cases |
|---|---:|---:|---:|---:|---:|
| QKV | 2.188x | 54.3% | 1.135x | 0.735x | 20/20 |
| gate/up | 1.278x | 21.7% | 0.184x | 0.375x | 20/20 |

A comparator ratio above 1 means QVQ is faster; below 1 means the W4 GPTQ
baseline is faster.  W2–W3.5 QVQ and W4 GPTQ reconstruct different weights,
so these are execution-efficiency comparisons, not output-equality claims.

## Combined model-site matrix

Each row sums the three QKV child calls and the two gate/up child calls.
Aggregate N is shown as `3072 + 16384` so the table does not imply one
mathematical GEMM.  “Better” compares against independent production
`QVQLinear` calls on the same sources, which is the last comparable benchmark.

| Rate | M | M x K x aggregate N | Plain QVQ us | Phase 4 us | Speedup | Marlin W4 us | vs Marlin | Machete W4 us | vs Machete | Better than last benchmark | Effective TFLOP/s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| W2 | 1 | 1 x 2048 x (3072 + 16384) | 212.372 | 130.271 | 1.630x | 63.933 | 0.491x | 65.674 | 0.504x | Yes | 0.612 |
| W2 | 2 | 2 x 2048 x (3072 + 16384) | 209.255 | 130.352 | 1.605x | 72.781 | 0.558x | 65.390 | 0.502x | Yes | 1.223 |
| W2 | 4 | 4 x 2048 x (3072 + 16384) | 208.674 | 130.827 | 1.595x | 73.694 | 0.563x | 65.330 | 0.499x | Yes | 2.437 |
| W2 | 8 | 8 x 2048 x (3072 + 16384) | 208.479 | 131.102 | 1.590x | 65.402 | 0.499x | 64.430 | 0.491x | Yes | 4.863 |
| W2 | 16 | 16 x 2048 x (3072 + 16384) | 195.578 | 126.428 | 1.547x | 71.052 | 0.562x | 64.252 | 0.508x | Yes | 10.085 |
| W2.5 | 1 | 1 x 2048 x (3072 + 16384) | 210.338 | 131.726 | 1.597x | 63.933 | 0.485x | 65.674 | 0.499x | Yes | 0.605 |
| W2.5 | 2 | 2 x 2048 x (3072 + 16384) | 208.378 | 132.147 | 1.577x | 72.781 | 0.551x | 65.390 | 0.495x | Yes | 1.206 |
| W2.5 | 4 | 4 x 2048 x (3072 + 16384) | 208.195 | 132.601 | 1.570x | 73.694 | 0.556x | 65.330 | 0.493x | Yes | 2.404 |
| W2.5 | 8 | 8 x 2048 x (3072 + 16384) | 207.678 | 132.719 | 1.565x | 65.402 | 0.493x | 64.430 | 0.485x | Yes | 4.804 |
| W2.5 | 16 | 16 x 2048 x (3072 + 16384) | 195.603 | 128.195 | 1.526x | 71.052 | 0.554x | 64.252 | 0.501x | Yes | 9.946 |
| W3 | 1 | 1 x 2048 x (3072 + 16384) | 209.105 | 124.385 | 1.681x | 63.933 | 0.514x | 65.674 | 0.528x | Yes | 0.641 |
| W3 | 2 | 2 x 2048 x (3072 + 16384) | 207.050 | 124.587 | 1.662x | 72.781 | 0.584x | 65.390 | 0.525x | Yes | 1.279 |
| W3 | 4 | 4 x 2048 x (3072 + 16384) | 206.682 | 125.438 | 1.648x | 73.694 | 0.587x | 65.330 | 0.521x | Yes | 2.541 |
| W3 | 8 | 8 x 2048 x (3072 + 16384) | 207.093 | 125.501 | 1.650x | 65.402 | 0.521x | 64.430 | 0.513x | Yes | 5.080 |
| W3 | 16 | 16 x 2048 x (3072 + 16384) | 194.656 | 121.172 | 1.606x | 71.052 | 0.586x | 64.252 | 0.530x | Yes | 10.523 |
| W3.5 | 1 | 1 x 2048 x (3072 + 16384) | 212.277 | 123.964 | 1.712x | 63.933 | 0.516x | 65.674 | 0.530x | Yes | 0.643 |
| W3.5 | 2 | 2 x 2048 x (3072 + 16384) | 210.625 | 124.130 | 1.697x | 72.781 | 0.586x | 65.390 | 0.527x | Yes | 1.284 |
| W3.5 | 4 | 4 x 2048 x (3072 + 16384) | 209.944 | 124.762 | 1.683x | 73.694 | 0.591x | 65.330 | 0.524x | Yes | 2.555 |
| W3.5 | 8 | 8 x 2048 x (3072 + 16384) | 209.994 | 124.774 | 1.683x | 65.402 | 0.524x | 64.430 | 0.516x | Yes | 5.110 |
| W3.5 | 16 | 16 x 2048 x (3072 + 16384) | 197.658 | 120.642 | 1.638x | 71.052 | 0.589x | 64.252 | 0.533x | Yes | 10.569 |

Effective TFLOP/s uses dense-equivalent logical work,

\[
\frac{2M K \sum_i N_i}{t},
\]

and therefore does not credit P32 decoder integer work or padded M16 rows.

## Why gate/up remains behind

The Phase-3 W3 inner grouped kernel measured about 31 microseconds for gate/up.
The Phase-4 complete gate/up operation measures about 76–78 microseconds.  The
difference is primarily the two independent 8192-wide output Hadamards,
`SV` application, output casting, and allocation traffic.  The input transform
is already shared, and the inner grouped kernel is no longer the dominant
gate/up cost.

QKV behaves differently.  Its narrow K/V launch tails were expensive, while
the output recovery tensors are much smaller.  The full QKV path is 2.187x
faster than independent QVQ and 1.135x faster than summed W4 Marlin.  It remains
about 1.36x slower than summed W4 Machete.

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
