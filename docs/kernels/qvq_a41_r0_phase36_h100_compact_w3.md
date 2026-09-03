# Phase 36: compact H100 W3 gate/up specialization

Phase 36 promotes Phase 32's compact fixed Llama gate/up launch to W3. The
earlier full-descriptor fixed W3 experiment was mixed at complete-MLP level;
the compact representation changes that result. The new W3 kernel removes
168 static SASS instructions, six registers/thread, and 104 stack bytes, and
improves all five targeted H100 MLP cells.

## Exact design

The host dispatch gate is:

```text
physical NVIDIA H100 / SM90
transition bits = 4 (W2) or 6 (W3)
two children
K = 2048 per child
N = 8192 per child
split count = 1 per child
```

Both rates use the same fixed grid and compact two-bank descriptor introduced
in Phases 30 and 32. All rate-specific state extraction, level storage, PGC
math, and WGMMA schedule remain separate template instantiations. W2.5 and
W3.5 stay on the generic grouped kernel.

No canonical payload, selector, alternative-bank semantics, checkpoint state,
output recovery, or FP16 rounding boundary changes.

## Matched isolated result

Timing covers grouped gate/up P32 plus exact paired recovery using 30 warmups,
200 CUDA-event samples, and 50 warmed CUDA Graph replays per sample after the
strict 0% utilization / 0 MiB H100 gate.

| M/K/N per child | Generic W3 | Compact fixed W3 | Speedup | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 32.541 us | 31.800 us | 1.023x | 0.431x | 0.900x | Yes |
| 2/2048/8192 x2 | 32.599 us | 31.854 us | 1.023x | 0.441x | 0.880x | Yes |
| 4/2048/8192 x2 | 33.060 us | 32.359 us | 1.022x | 0.434x | 0.863x | Yes |
| 8/2048/8192 x2 | 33.556 us | 32.861 us | 1.021x | 0.413x | 0.854x | Yes |
| 16/2048/8192 x2 | 34.371 us | 33.688 us | 1.020x | 0.447x | 0.836x | Yes |

The isolated geometric mean is **1.0219x**, with five of five wins.

## Formal complete-MLP result

The complete path includes grouped gate/up, recovery, fused SiLU/product/down
preconditioning, split-16 down P32, down recovery, and runtime coordination.
Effective TFLOP/s is dense-equivalent; Marlin/Machete are figurative W4
baselines rather than equal-rate comparisons.

| W | M/K/N shapes | QVQ MLP | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 32 | Better than last benchmark |
|:--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 52.392 us | 1.921 | 0.558x | 0.964x | 1.000x | No |
| 2 | 2/2048/8192 x2; 2/8192/2048 | 52.517 us | 3.834 | 0.594x | 0.959x | 1.003x | Yes |
| 2 | 4/2048/8192 x2; 4/8192/2048 | 53.264 us | 7.560 | 0.589x | 0.953x | 1.001x | Yes |
| 2 | 8/2048/8192 x2; 8/8192/2048 | 53.741 us | 14.985 | 0.545x | 0.943x | 1.000x | Yes |
| 2 | 16/2048/8192 x2; 16/8192/2048 | 55.088 us | 29.237 | 0.590x | 0.921x | 1.000x | Yes |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 54.270 us | 1.855 | 0.539x | 0.931x | 0.999x | No |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 54.444 us | 3.698 | 0.573x | 0.925x | 0.998x | No |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 55.134 us | 7.303 | 0.569x | 0.921x | 0.999x | No |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 55.814 us | 14.428 | 0.524x | 0.908x | 0.998x | No |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 56.834 us | 28.339 | 0.572x | 0.893x | 0.998x | No |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 50.868 us | 1.979 | 0.575x | 0.993x | 1.012x | Yes |
| 3 | 2/2048/8192 x2; 2/8192/2048 | 50.890 us | 3.956 | 0.613x | 0.989x | 1.014x | Yes |
| 3 | 4/2048/8192 x2; 4/8192/2048 | 51.429 us | 7.829 | 0.610x | 0.987x | 1.013x | Yes |
| 3 | 8/2048/8192 x2; 8/8192/2048 | 52.093 us | 15.459 | 0.562x | 0.973x | 1.012x | Yes |
| 3 | 16/2048/8192 x2; 16/8192/2048 | 53.150 us | 30.303 | 0.612x | 0.955x | 1.016x | Yes |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 54.704 us | 1.840 | 0.535x | 0.924x | 0.999x | No |
| 3.5 | 2/2048/8192 x2; 2/8192/2048 | 54.836 us | 3.671 | 0.569x | 0.918x | 0.996x | No |
| 3.5 | 4/2048/8192 x2; 4/8192/2048 | 55.153 us | 7.301 | 0.569x | 0.920x | 0.999x | No |
| 3.5 | 8/2048/8192 x2; 8/8192/2048 | 55.673 us | 14.465 | 0.526x | 0.910x | 0.994x | No |
| 3.5 | 16/2048/8192 x2; 16/8192/2048 | 56.552 us | 28.480 | 0.575x | 0.897x | 0.999x | No |

The targeted W3 geometric mean improves **1.0136x**, with five of five wins.
The all-rate geomean improves 1.0025x. Untouched rates are reported as strict
cross-run telemetry but are not attributed to the W3 change. Current geomeans
are 0.5695x versus Marlin W4, 0.9387x versus Machete W4, and 2.5940x versus
ordinary per-module QVQ.

## Matched Nsight Compute and SASS

Both binaries were captured with Nsight Compute 2026.2.1 using 19 replay
passes on W3 M1. `cuobjdump` measured static SASS and resources.

| Metric | Generic W3 | Compact fixed W3 | Change |
|:--|--:|--:|--:|
| Static SASS instructions | 1,952 | 1,784 | **-8.61%** |
| Executed warp instructions | 10,792,216 | 10,639,754 | **-1.41%** |
| Registers/thread | 60 | 54 | **-10.0%** |
| Stack | 112 B | 8 B | **-92.9%** |
| Constant parameter space | 1,552 B | 1,448 B | -6.70% |
| Static shared memory | 45.824 KiB | 45.824 KiB | unchanged |
| NCU replay duration | 28.032 us | 28.000 us | -0.11% |
| Eligible warps/scheduler/cycle | 0.5827 | 0.5873 | +0.78% |
| DRAM throughput | 18.72% | 18.72% | unchanged |

Counter-replay duration is instrumentation data. CUDA-event/Graph latency is
the production promotion gate.

Reports remain outside Git:

```text
/root/qvq-profiler-artifacts/phase36-compact-w3-params/baseline_w3_m1.ncu-rep
/root/qvq-profiler-artifacts/phase36-compact-w3-params/candidate_w3_m1.ncu-rep
```

## Validation and next phase

- all 59 grouped Hopper kernel tests pass across W2-W3.5;
- exact child-local alternative banks, dense-oracle bounds, repeatability,
  deterministic split behavior, and CUDA Graph capture remain covered;
- no payload, checkpoint, VRAM allocation, output buffer, or graph node was
  added;
- compilation used Ninja `-j4`, one NVCC host thread, and one split-compile
  partition.

Artifacts:

- `artifacts/a41_phase36_h100/generic_w3_isolated_baseline.json`
- `artifacts/a41_phase36_h100/compact_w3_isolated_candidate.json`
- `artifacts/a41_phase36_h100/compact_w3_mlp_candidate.json`
- `artifacts/a41_phase36_h100/production_mlp_compact_w3_params_vs_phase32.json`
- `artifacts/a41_phase36_h100/compact_w3_ncu_summary.json`

Phase 37 should test W2.5 independently. Its Phase-30 full-descriptor fixed
candidate regressed, so it must not be enabled speculatively; however, Phase
32/36 prove the compact descriptor materially changes register/stack pressure.
The gate should add only transition width five, run the isolated matrix first,
and stop immediately if it is not 5/5 positive with smaller compiled work.
