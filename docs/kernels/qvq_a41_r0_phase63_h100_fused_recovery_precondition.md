# Phase 63: H100 fused recovery-to-precondition pipeline

Phase 63 removes the materialized gate/up recovery boundary from the exact
Llama 3.2 1B A41 MLP on the physical NVIDIA H100. The promoted path combines
the final five paired-recovery Hadamard stages with the rounded FP16 SiLU,
gate/up product, down scale/divisor, and first eight down-input Hadamard
stages. It reduces the intermediate pipeline from four CUDA kernels to three
and removes both recovered `[M,8192]` FP16 tensors.

The complete MLP improves all 20 W2--W3.5 and M1--M16 cells. Geometric mean
speedup is **1.0310x versus Phase 56**, **1.0750x versus Machete W4**, and
**0.6601x versus Marlin W4**. Complete latency is **44.290--49.556 us**.

## Exact selected-output Hadamard math

The accepted recovery factorization first computes 32 independent low tiles
of 256 columns. For one within-tile column, recovery-high is a 32-point
Hadamard over tile index. The ordinary kernel evaluates all 32 outputs in one
thread. Phase 63 instead assigns one final output column to each thread and
evaluates only that output's dependency tree.

Let `R` be the existing operation that rounds to FP16 when finite but keeps an
overflowing intermediate in FP32. For requested output tile `t`, start with
the 32 low-stage values and repeatedly halve the active list:

```text
values = [x0, x1, ..., x31]

for stage = 0..4:
    sign = -1 if bit(stage, t) else +1
    values[j] = R(values[2*j] + sign * values[2*j + 1])

high_output(t) = values[0]
```

This executes exactly the same balanced, ascending-bit butterfly subtree as
the ordinary transform. One selected output requires 31 rounded operations.
The 32 destination-tile blocks expose all 8192 output columns concurrently,
which deliberately trades redundant cached reads and arithmetic for enough
parallelism to absorb the nonlinear operation without a global gate/up
materialization.

For each column `c`, the fused pointwise boundary remains:

```text
gate_h = fp16(R(recovery_high_gate(c) * SV_gate[c] + bias_gate[c]))
up_h   = fp16(R(recovery_high_up(c)   * SV_up[c]   + bias_up[c]))

activated_h = fp16(gate_h / (1 + exp(-gate_h)))
product_h   = fp16(activated_h * up_h)
seed_h      = fp16(R(product_h * SU_down[c]) / fp16(sqrt(8192)))
```

The existing exact packed-FP16 low and high down-input Hadamard stages then
consume `seed_h`. The final padded-M16 contract is unchanged.

## Launch and temporary-memory change

| Property | Phase 56 | Phase 63 |
|:--|--:|--:|
| recovery-low launches | 1 | 1 |
| recovery-high launches | 1 | folded into fused middle |
| precondition-low launches | 1 | folded into fused middle |
| precondition-high launches | 1 | 1 |
| total transform launches | 4 | **3** |
| recovered gate tensor | `M*8192*2` bytes | **removed** |
| recovered up tensor | `M*8192*2` bytes | **removed** |
| recovery FP32 workspace | `2*M*8192*4` bytes | unchanged |
| precondition FP16 workspace | `M*8192*2` bytes | unchanged |
| persistent VRAM added | 0 | 0 |

The physical checkpoint, grouped P32 payload, selectors, alternative-bank
state, split-K partials, and deterministic reduction order do not change.

## Isolated recovery plus precondition

Timing uses 30 warmups, 300 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample after three spaced 0% utilization / 0 MiB H100 admission
samples. The control and candidate run from one binary. Every output is bit
exact.

| M | M/K/N gate/up x2 | Four-kernel control | Three-kernel fused | vs control | Better |
|--:|:--|--:|--:|--:|:--:|
| 1 | 1/2048/8192 x2 | 9.834 us | 7.567 us | 1.2997x | Yes |
| 2 | 2/2048/8192 x2 | 9.887 us | 7.614 us | 1.2985x | Yes |
| 4 | 4/2048/8192 x2 | 10.293 us | 7.836 us | 1.3136x | Yes |
| 8 | 8/2048/8192 x2 | 10.653 us | 8.602 us | 1.2384x | Yes |
| 16 | 16/2048/8192 x2 | 11.609 us | 10.553 us | 1.1000x | Yes |

## Complete Llama 3.2 1B MLP

Marlin and Machete execute W4 and are figurative latency/efficiency baselines,
not equal-bit-rate comparisons. Effective TFLOP/s uses the dense-equivalent
gate/up/down operation count. `Better` compares with committed Phase 56.

| W | M | M/K/N: gate/up x2; down | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 56 | Better |
|--:|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 1 | 1/2048/8192 x2; 1/8192/2048 | 44.290 | 2.273 | 0.655x | 1.124x | 1.0376x | Yes |
| 2 | 2 | 2/2048/8192 x2; 2/8192/2048 | 44.558 | 4.518 | 0.699x | 1.113x | 1.0313x | Yes |
| 2 | 4 | 4/2048/8192 x2; 4/8192/2048 | 44.979 | 8.952 | 0.697x | 1.109x | 1.0397x | Yes |
| 2 | 8 | 8/2048/8192 x2; 8/8192/2048 | 45.900 | 17.545 | 0.638x | 1.092x | 1.0278x | Yes |
| 2 | 16 | 16/2048/8192 x2; 16/8192/2048 | 47.985 | 33.565 | 0.678x | 1.044x | 1.0110x | Yes |
| 2.5 | 1 | 1/2048/8192 x2; 1/8192/2048 | 45.466 | 2.214 | 0.638x | 1.095x | 1.0402x | Yes |
| 2.5 | 2 | 2/2048/8192 x2; 2/8192/2048 | 45.749 | 4.401 | 0.681x | 1.084x | 1.0370x | Yes |
| 2.5 | 4 | 4/2048/8192 x2; 4/8192/2048 | 46.318 | 8.693 | 0.677x | 1.077x | 1.0382x | Yes |
| 2.5 | 8 | 8/2048/8192 x2; 8/8192/2048 | 47.096 | 17.099 | 0.622x | 1.064x | 1.0308x | Yes |
| 2.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | 49.194 | 32.740 | 0.661x | 1.018x | 1.0112x | Yes |
| 3 | 1 | 1/2048/8192 x2; 1/8192/2048 | 45.027 | 2.236 | 0.644x | 1.105x | 1.0345x | Yes |
| 3 | 2 | 2/2048/8192 x2; 2/8192/2048 | 45.196 | 4.455 | 0.689x | 1.097x | 1.0383x | Yes |
| 3 | 4 | 4/2048/8192 x2; 4/8192/2048 | 45.726 | 8.806 | 0.685x | 1.091x | 1.0386x | Yes |
| 3 | 8 | 8/2048/8192 x2; 8/8192/2048 | 46.442 | 17.340 | 0.630x | 1.079x | 1.0362x | Yes |
| 3 | 16 | 16/2048/8192 x2; 16/8192/2048 | 48.505 | 33.205 | 0.671x | 1.032x | 1.0136x | Yes |
| 3.5 | 1 | 1/2048/8192 x2; 1/8192/2048 | 46.006 | 2.188 | 0.631x | 1.082x | 1.0388x | Yes |
| 3.5 | 2 | 2/2048/8192 x2; 2/8192/2048 | 46.478 | 4.332 | 0.670x | 1.067x | 1.0331x | Yes |
| 3.5 | 4 | 4/2048/8192 x2; 4/8192/2048 | 46.797 | 8.604 | 0.670x | 1.066x | 1.0371x | Yes |
| 3.5 | 8 | 8/2048/8192 x2; 8/8192/2048 | 47.355 | 17.006 | 0.618x | 1.058x | 1.0355x | Yes |
| 3.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | 49.556 | 32.501 | 0.657x | 1.010x | 1.0113x | Yes |

Geometric means:

- **1.0310x** versus Phase 56;
- **2.9167x** versus ordinary per-module QVQ;
- **1.0750x** versus Machete W4;
- **0.6601x** versus Marlin W4.

## Promotion gates

- Twelve focused fused-operator cases pass for M1/M8/M16, scale modes 3/4,
  padded/unpadded outputs, ordinary repeatability, and CUDA Graph replay.
- The real Llama 3.2 layer/logits/cached-generation test passes and observes
  the new lifecycle telemetry at the intended fused MLP point.
- The complete benchmark checks exact equality with the unchanged plain QVQ
  MLP output for every one of the 20 cells.
- Dispatch is restricted by the existing physical H100, SM90, gate/up
  N=8192, exact-SiLU, and legal grouped-runtime gates. Other devices, shapes,
  activations, and fallback paths retain Phase 56.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one CUDA
  split-compile partition.

Artifacts:

- `artifacts/a41_phase63_h100/fused_recovery_precondition.json`
- `artifacts/a41_phase63_h100/production_mlp_fused_recovery_precondition_vs_phase56.json`
