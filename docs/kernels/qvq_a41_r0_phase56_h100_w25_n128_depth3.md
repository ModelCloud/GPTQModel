# Phase 56: H100 W2.5 N128 depth-three decode

Phase 56 reduces the W2.5 dual-consumer gate/up decoder from four live input
fragments to three. A matched H100 depth sweep selected depth three at every
target M. The complete W2.5 MLP improves in all five formal cells, with a
second independent run confirming the geometric improvement.

## Dependency math

Each consumer warpgroup issues one WGMMA operation per K16 tile. A decoded A
fragment cannot be overwritten until the WGMMA operation that references it
is no longer live. If the circular fragment depth is \(D\), fragment
\(k\bmod D\) is reused after \(D\) committed groups.

The prefetch path first loads and packs the next eight FP16 levels into
temporary registers. Immediately before those packed values overwrite the
old A fragment, it applies:

\[
\operatorname{wait\_group}(D-1).
\]

The tested schedules are therefore:

| Depth | Live A fragments | Reuse wait |
|--:|--:|--:|
| 2 | 2 | wait group 1 |
| 3 | 3 | wait group 2 |
| 4 | 4 | wait group 3 |

The wait remains after decoded-level loads, preserving overlap between shared
level access and the prior WGMMA. Only the overwrite distance changes. State
extraction, PGC, decoded values, WGMMA order, and FP32 accumulation are
identical.

The specialization is compile-time restricted to the Phase-55 W2.5 N128
kernel. N64 kernels, other transition widths, other devices, and other shapes
retain their existing fragment depth and source expressions.

## Matched depth sweep

The isolated workload contains grouped gate/up P32 plus exact paired output
recovery. Timing uses 30 warmups, 200 samples, and 50 warmed CUDA Graph
replays per sample on the idle physical H100.

| M/K/N per child | Depth 2 | Depth 3 | Depth 4 | Depth 3 vs 4 |
|:--|--:|--:|--:|--:|
| 1/2048/8192 | 27.220 us | **27.080 us** | 27.135 us | **1.0020x** |
| 2/2048/8192 | 26.844 us | **26.520 us** | 26.604 us | **1.0032x** |
| 4/2048/8192 | 27.266 us | **26.913 us** | 27.011 us | **1.0037x** |
| 8/2048/8192 | 27.515 us | **27.183 us** | 27.253 us | **1.0026x** |
| 16/2048/8192 | 28.420 us | **28.151 us** | 28.163 us | **1.0004x** |

Depth three wins 5/5 cells and improves **1.00236x** geometrically over depth
four. Depth two loses throughout, showing that two in-flight fragments do not
cover the decode/WGMMA dependency distance.

## Nsight Compute comparison

Nsight Compute 2026.2.1 profiled W2.5/M1 at the exact production source. The
Phase-55 depth-four report is a matched device, shape, launch, and metric
control.

| Metric | Depth 4 | Depth 3 | Change |
|:--|--:|--:|--:|
| Duration under NCU | 24.800 us | 24.800 us | unchanged |
| Executed instructions | 10,373,376 | 10,381,568 | +0.08% |
| Shared-load bank conflicts | 1,048,557 | 1,048,556 | unchanged |
| Shared-load wavefronts | 3,391,684 | 3,395,336 | +0.11% |
| Eligible warps/cycle | 0.679 | **0.741** | +9.1% |
| Long-scoreboard ratio | 0.854 | **0.799** | -6.4% |
| Registers/thread | 65 | **64** | -1 |
| Dynamic shared memory | 54.016 KiB | 54.016 KiB | unchanged |

Depth three does not reduce instructions or shared-memory conflicts. It
reduces the live register set by one register/thread and improves scheduler
eligibility and long-scoreboard behavior. This is a dependency-latency win,
not an arithmetic win.

## Complete Llama 3.2 1B MLP

The formal exact-SHA run acquired three spaced 0% utilization / 0 MiB idle
samples and uses CUDA-event timing of warmed CUDA Graph replay. Marlin and
Machete are figurative W4 baselines; ratios above one mean QVQ is faster.
`Better` compares with Phase 55.

| W | MKN: gate/up x2; down | QVQ | vs Marlin W4 | vs Machete W4 | Better |
|--:|:--|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 45.955 us | 0.636x | 1.097x | Yes |
| 2 | 2/2048/8192 x2; 2/8192/2048 | 45.951 us | 0.679x | 1.076x | Yes |
| 2 | 4/2048/8192 x2; 4/8192/2048 | 46.764 us | 0.671x | 1.074x | Yes |
| 2 | 8/2048/8192 x2; 8/8192/2048 | 47.175 us | 0.620x | 1.071x | Yes |
| 2 | 16/2048/8192 x2; 16/8192/2048 | 48.514 us | 0.669x | 1.048x | Yes |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 47.292 us | 0.618x | 1.066x | Yes |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 47.442 us | 0.658x | 1.043x | Yes |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 48.088 us | 0.653x | 1.044x | Yes |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 48.547 us | 0.603x | 1.041x | Yes |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 49.747 us | 0.652x | 1.022x | Yes |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 46.582 us | 0.627x | 1.082x | Yes |
| 3 | 2/2048/8192 x2; 2/8192/2048 | 46.925 us | 0.665x | 1.054x | Yes |
| 3 | 4/2048/8192 x2; 4/8192/2048 | 47.492 us | 0.661x | 1.057x | No |
| 3 | 8/2048/8192 x2; 8/8192/2048 | 48.121 us | 0.608x | 1.050x | Yes |
| 3 | 16/2048/8192 x2; 16/8192/2048 | 49.167 us | 0.660x | 1.034x | Yes |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 47.793 us | 0.611x | 1.054x | No |
| 3.5 | 2/2048/8192 x2; 2/8192/2048 | 48.015 us | 0.650x | 1.030x | No |
| 3.5 | 4/2048/8192 x2; 4/8192/2048 | 48.532 us | 0.647x | 1.035x | No |
| 3.5 | 8/2048/8192 x2; 8/8192/2048 | 49.036 us | 0.597x | 1.030x | No |
| 3.5 | 16/2048/8192 x2; 16/8192/2048 | 50.118 us | 0.647x | 1.015x | No |

The changed W2.5 path improves **1.00136x** geometrically with **5/5** strict
wins. An independent confirmation measured **1.00234x** with 4/5 strict wins;
the sole M8 loss was 0.015 us. The formal all-rate comparison is **1.00058x**
with 14/20 wins. Movements in W2/W3/W3.5 are run-to-run noise because their
kernel instantiations and runtime paths did not change.

The formal all-rate geometric ratios are **1.0510x versus Machete W4** and
**0.6411x versus Marlin W4**. This benchmark's Machete baseline ran faster
than the Phase-55 baseline, which explains the smaller cross-kernel ratio even
though QVQ itself improved.

## Correctness and resource scope

- 20 real Llama-shape W2.5/W3 grouped Hopper cases pass after the scheduling
  change.
- Every depth candidate retained exact output, repeatability, dense-oracle
  tolerance, and CUDA Graph stability.
- Quantization payload bytes, persistent VRAM, dynamic shared memory, launch
  count, and mathematical operation order are unchanged.
- Compilation used no more than four Ninja jobs, one NVCC host thread, and
  one CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase56_h100/production_mlp_w25_n128_depth3_vs_phase55.json`
- Nsight Compute report outside Git under
  `/root/qvq-profiler-artifacts/phase56-w25-n128-depth3/`

## Next experiment

The remaining N128 kernel still has roughly one million shared-load bank
conflicts and is long-scoreboard dominated. A next experiment should change
the shared level-table mapping or replace repeated shared lookups with a
conflict-free lane-local representation. It must be evaluated as a physical
layout/scheduler change, because merely reducing arithmetic has repeatedly
failed to improve the complete MLP.
