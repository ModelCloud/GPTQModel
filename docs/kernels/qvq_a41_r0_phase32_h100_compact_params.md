# Phase 32: compact H100 W2 gate/up launch parameters

Phase 32 promotes a compact launch-parameter type for the fixed Llama 3.2 1B
W2 gate/up kernel. It removes 120 bytes of irrelevant per-launch metadata from
the specialized kernel ABI, cuts its `cuobjdump` stack allocation from 112 to
8 bytes, reduces both static and executed SASS, and improves every targeted
H100 MLP cell.

## Exact design

The generic grouped kernel needs descriptors for up to three variable-width,
variable-split children:

```text
segment count
N tile start[3]
N tile count[3]
alternative bank[3]
split count[3]
work start[3]
output offset[3]
partial-output offset[3]
```

That structure occupies 128 bytes. Phase 30's fixed specialization already
proves that W2 gate/up is exactly two K2048 x N8192 children with split one.
Inside that kernel the only descriptor state still used is:

```text
alternative_bank[0]
alternative_bank[1]
```

Phase 32 therefore templates the kernel on its launch-parameter type. Generic
and ordered grouped paths retain the full descriptor. The fixed W2 path receives
an eight-byte structure containing only the two child-local bank identifiers.

This is an ABI/control change, not a quantization-format change. The input,
trellis, selector stream, levels, child boundaries, WGMMA accumulation order,
SV/bias, output transforms, and FP16 rounding boundaries are unchanged.

## Isolated gate/up result

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB admission
gate. Results use 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample. Timing includes grouped inner P32 and exact paired recovery.

| M/K/N per child | Phase 30 | Phase 32 | Speedup | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 33.773 us | 33.021 us | 1.023x | 0.417x | 0.862x | Yes |
| 2/2048/8192 x2 | 33.791 us | 33.070 us | 1.022x | 0.419x | 0.845x | Yes |
| 4/2048/8192 x2 | 34.242 us | 33.535 us | 1.021x | 0.417x | 0.830x | Yes |
| 8/2048/8192 x2 | 34.670 us | 34.031 us | 1.019x | 0.396x | 0.819x | Yes |
| 16/2048/8192 x2 | 35.505 us | 34.780 us | 1.021x | 0.431x | 0.803x | Yes |

The isolated geometric mean is **1.0211x**, with five of five wins.

## Formal complete-MLP result

Timing includes grouped gate/up P32, recovery, fused SiLU/product/down
preconditioning, split-16 down P32, down recovery, and runtime coordination.
Effective TFLOP/s counts dense-equivalent gate, up, and down operations. Marlin
and Machete are figurative W4 references, not equal-rate comparisons.

| W | M/K/N shapes | QVQ MLP | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 30 | Better than last benchmark |
|:--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 52.379 us | 1.922 | 0.560x | 0.960x | 1.015x | Yes |
| 2 | 2/2048/8192 x2; 2/8192/2048 | 52.671 us | 3.822 | 0.593x | 0.962x | 1.011x | Yes |
| 2 | 4/2048/8192 x2; 4/8192/2048 | 53.318 us | 7.552 | 0.589x | 0.960x | 1.011x | Yes |
| 2 | 8/2048/8192 x2; 8/8192/2048 | 53.755 us | 14.981 | 0.545x | 0.953x | 1.014x | Yes |
| 2 | 16/2048/8192 x2; 16/8192/2048 | 55.089 us | 29.236 | 0.590x | 0.931x | 1.012x | Yes |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 54.192 us | 1.858 | 0.541x | 0.928x | 1.002x | Yes |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 54.351 us | 3.704 | 0.574x | 0.932x | 1.000x | No |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 55.080 us | 7.310 | 0.570x | 0.930x | 1.001x | Yes |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 55.691 us | 14.460 | 0.526x | 0.920x | 1.002x | Yes |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 56.728 us | 28.392 | 0.573x | 0.904x | 1.000x | Yes |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 51.490 us | 1.955 | 0.570x | 0.976x | 1.002x | Yes |
| 3 | 2/2048/8192 x2; 2/8192/2048 | 51.618 us | 3.900 | 0.605x | 0.981x | 1.004x | Yes |
| 3 | 4/2048/8192 x2; 4/8192/2048 | 52.110 us | 7.727 | 0.603x | 0.983x | 1.002x | Yes |
| 3 | 8/2048/8192 x2; 8/8192/2048 | 52.716 us | 15.276 | 0.556x | 0.972x | 1.003x | Yes |
| 3 | 16/2048/8192 x2; 16/8192/2048 | 54.003 us | 29.824 | 0.602x | 0.949x | 0.999x | No |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 54.645 us | 1.842 | 0.537x | 0.920x | 0.991x | No |
| 3.5 | 2/2048/8192 x2; 2/8192/2048 | 54.636 us | 3.685 | 0.571x | 0.927x | 1.005x | Yes |
| 3.5 | 4/2048/8192 x2; 4/8192/2048 | 55.077 us | 7.311 | 0.570x | 0.930x | 1.007x | Yes |
| 3.5 | 8/2048/8192 x2; 8/8192/2048 | 55.352 us | 14.549 | 0.529x | 0.926x | 1.009x | Yes |
| 3.5 | 16/2048/8192 x2; 16/8192/2048 | 56.493 us | 28.510 | 0.575x | 0.907x | 1.008x | Yes |

The targeted W2 geomean improves **1.0125x**, with five of five wins. The
all-rate geomean improves 1.0048x; untouched-rate cells are reported for
telemetry but are not attributed to this W2-only executable change. Current
geomeans are 0.5684x versus Marlin W4, 0.9422x versus Machete W4, and 2.5893x
versus ordinary per-module QVQ.

## Nsight Compute and compiled SASS

Nsight Compute 2026.2.1 ran 19 hardware-counter replay passes on the exact W2
M1 grouped kernel. CUDA `cuobjdump` independently measured static SASS and
resource use from both extension binaries.

| Metric | Phase 30 | Phase 32 | Change |
|:--|--:|--:|--:|
| Launch parameter bytes | 128 | 8 | **-93.75%** |
| `cuobjdump` stack | 112 B | 8 B | **-92.86%** |
| Constant parameter space | 1,552 B | 1,448 B | -6.70% |
| Static SASS instructions | 1,760 | 1,736 | **-1.36%** |
| Executed warp instructions | 10,290,194 | 10,256,914 | **-0.32%** |
| Registers/thread | 48 | 48 | unchanged |
| Static shared memory | 26.240 KiB | 26.240 KiB | unchanged |
| NCU replay duration | 28.960 us | 28.960 us | unchanged |
| Eligible warps/scheduler/cycle | 0.5243 | 0.5246 | +0.05% |
| Wait stall / issue-active | 0.8050 | 0.7977 | -0.91% |
| DRAM throughput | 12.17% | 12.18% | unchanged |

The counter-replay duration is instrumentation data, not the production
latency. The CUDA-event and CUDA-Graph results above are the promotion gate.

Profiler report remains outside Git:

```text
/root/qvq-profiler-artifacts/phase32-compact-w2-params/candidate_w2_m1.ncu-rep
```

## Validation and storage

- all 59 grouped Hopper kernel tests pass;
- all 50 grouped production-runtime/model lifecycle tests pass;
- tests cover W2-W3.5, M1/2/4/8/16, child-local alternative banks,
  deterministic ordered splits, dense-oracle bounds, repeatability, and CUDA
  Graph capture;
- no checkpoint state, packed payload, persistent/transient VRAM, output
  buffer, or graph node changed;
- compilation used Ninja `-j4`, one NVCC host thread, and one split-compile
  partition.

Artifacts:

- `artifacts/a41_phase32_h100/compact_params_isolated.json`
- `artifacts/a41_phase32_h100/production_mlp_compact_w2_params_vs_phase30.json`
- `artifacts/a41_phase32_h100/compact_params_ncu_summary.json`

## Next phase

Phase 32 shows that parameter/stack traffic is worth reducing when the ABI
actually becomes smaller. Phase 33 should inspect the fixed W2 kernel's
remaining scalar ABI. It still receives runtime K, total N, split count, and a
fallback bank argument even though the template fixes all four. A dedicated
fixed-kernel signature can remove those sixteen bytes entirely without putting
the constants back into CUTE tensor-shape expressions, avoiding the failed
Phase-31 transformation. Promotion again requires exactness, smaller SASS or
executed work, and all five M cells improving.
