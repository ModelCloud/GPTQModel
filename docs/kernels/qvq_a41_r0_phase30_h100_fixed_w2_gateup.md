# Phase 30: fixed H100 W2 Llama gate/up launch

Phase 30 promotes a compile-time grouped-launch specialization for the exact
Llama 3.2 1B W2 gate/up geometry. It removes generic segment discovery and
address setup while preserving the Phase-28 decoder, N64 work tile, two
independent canonical child payloads, accumulation order, recovery, and all
FP16 rounding boundaries.

## Exact geometry and launch math

The production gate is deliberately narrow:

```text
device                 = physical NVIDIA H100, SM90
transition bits        = 4 (W2)
children               = 2
K per child            = 2048
N per child            = 8192
split-K per child      = 1
ordered split reducer  = disabled
```

The generic grouped kernel finds a segment from runtime descriptor arrays and
then derives its child-local N coordinate, output offset, and split coordinate.
For the fixed shape the launch is rectangular and those values are known:

```text
grid.x = 8192 / 64 = 128 N64 tiles per child
grid.y = 2 children

segment       = blockIdx.y
local_n64     = blockIdx.x
global_n64    = segment * 128 + local_n64
split_id      = 0
output_offset = segment * M16 * 8192
```

The packed payload still has 1024 N16 tiles across the two children. Each
child retains its own trellis, selector stream, alternative-bank identifier,
SU/SV, bias, and output transform. The specialization changes only launch
geometry computation; it does not concatenate quantization solutions or share
child-local arithmetic.

## Candidate narrowing

The first experiment enabled the fixed prologue at every rate. W2 and W3
improved in the isolated projection test, but W2.5 and W3.5 regressed. A
W2/W3 gate then passed the isolated test; however, W3 improved only two of five
complete-MLP cells. Production therefore enables the path for W2 only.

Isolated grouped P32 plus paired recovery results for the promoted W2 path:

| M/K/N per child | Generic launch | Fixed launch | Speedup | Better than last benchmark |
|:--|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 32.292 us | 31.705 us | 1.019x | Yes |
| 2/2048/8192 x2 | 31.753 us | 31.341 us | 1.013x | Yes |
| 4/2048/8192 x2 | 32.046 us | 31.595 us | 1.014x | Yes |
| 8/2048/8192 x2 | 32.388 us | 31.951 us | 1.014x | Yes |
| 16/2048/8192 x2 | 33.196 us | 32.766 us | 1.013x | Yes |

## Formal H100 complete-MLP result

Timing includes grouped gate/up P32, recovery, SiLU/product/down
preconditioning, split-16 down P32, down recovery, and runtime coordination.
All measurements use 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA
Graph replays per sample after the strict 0% utilization / 0 MiB H100 admission
gate. Effective TFLOP/s counts the dense-equivalent gate, up, and down matrix
operations. Marlin and Machete are W4 baselines.

| W | M/K/N shapes | QVQ MLP | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 28 | Better than last benchmark |
|:--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 53.185 us | 1.893 | 0.546x | 0.934x | 1.006x | Yes |
| 2 | 2/2048/8192 x2; 2/8192/2048 | 53.236 us | 3.782 | 0.586x | 0.943x | 1.009x | Yes |
| 2 | 4/2048/8192 x2; 4/8192/2048 | 53.888 us | 7.472 | 0.582x | 0.937x | 1.007x | Yes |
| 2 | 8/2048/8192 x2; 8/8192/2048 | 54.502 us | 14.776 | 0.537x | 0.932x | 1.004x | Yes |
| 2 | 16/2048/8192 x2; 16/8192/2048 | 55.743 us | 28.894 | 0.583x | 0.911x | 1.005x | Yes |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 54.303 us | 1.854 | 0.534x | 0.915x | 0.997x | No |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 54.346 us | 3.705 | 0.574x | 0.924x | 0.999x | No |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 55.140 us | 7.302 | 0.569x | 0.915x | 0.998x | No |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 55.779 us | 14.438 | 0.525x | 0.910x | 0.995x | No |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 56.729 us | 28.391 | 0.572x | 0.895x | 0.999x | No |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 51.600 us | 1.951 | 0.562x | 0.963x | 0.999x | No |
| 3 | 2/2048/8192 x2; 2/8192/2048 | 51.826 us | 3.885 | 0.602x | 0.969x | 0.996x | No |
| 3 | 4/2048/8192 x2; 4/8192/2048 | 52.198 us | 7.714 | 0.601x | 0.967x | 1.001x | Yes |
| 3 | 8/2048/8192 x2; 8/8192/2048 | 52.876 us | 15.230 | 0.554x | 0.960x | 0.996x | No |
| 3 | 16/2048/8192 x2; 16/8192/2048 | 53.934 us | 29.863 | 0.602x | 0.941x | 0.998x | No |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 54.180 us | 1.858 | 0.536x | 0.917x | 1.002x | Yes |
| 3.5 | 2/2048/8192 x2; 2/8192/2048 | 54.885 us | 3.668 | 0.568x | 0.915x | 0.990x | No |
| 3.5 | 4/2048/8192 x2; 4/8192/2048 | 55.460 us | 7.260 | 0.566x | 0.910x | 0.992x | No |
| 3.5 | 8/2048/8192 x2; 8/8192/2048 | 55.822 us | 14.426 | 0.525x | 0.910x | 0.995x | No |
| 3.5 | 16/2048/8192 x2; 16/8192/2048 | 56.926 us | 28.293 | 0.571x | 0.892x | 0.991x | No |

The W2 promotion population improves **1.0064x geometric mean**, with all five
cells improving. The all-rate same-run geometric comparison is 0.9990x because
the three untouched rates contain ordinary cross-run variance; no W2.5/W3/W3.5
executable path changed. Promotion is based on the five targeted W2 cells, not
on attributing unrelated timing drift to this specialization.

## Nsight Compute and SASS

Matched Nsight Compute 2026.2.1 reports use source-correlated SASS on W2 M1.

| Metric | Generic launch | Fixed launch | Change |
|:--|--:|--:|--:|
| Replay duration | 29.376 us | 28.960 us | **-1.42%** |
| Executed warp instructions | 10,408,466 | 10,290,194 | **-1.14%** |
| Registers/thread | 48 | 48 | unchanged |
| Static shared memory | 26.240 KiB | 26.240 KiB | unchanged |
| Eligible warps/scheduler/cycle | 0.499 | 0.524 | **+5.03%** |
| Long-scoreboard / issue-active | 1.424 | 1.384 | -2.77% |
| Wait stall / issue-active | 0.952 | 0.805 | **-15.5%** |
| DRAM throughput | 12.01% | 12.17% | +1.31% |

The fixed prologue removes 118,272 dynamic warp instructions without changing
register or shared-memory use. The largest count changes are fewer `IMAD`,
`R2UR`, branches, comparisons, and address-generation instructions. Decoder
loads, permutations, and WGMMA counts stay effectively fixed, which matches
the intended control/address-only transformation.

Profiler reports remain outside Git:

```text
/root/qvq-profiler-artifacts/phase29-w2-lane-plan/baseline_w2_m1.ncu-rep
/root/qvq-profiler-artifacts/phase30-fixed-w2-gateup/candidate_w2_m1.ncu-rep
```

## Correctness and resource gates

- all 59 grouped Hopper tests pass across W2-W3.5;
- all measured outputs are repeatable, CUDA-Graph stable, and satisfy the
  existing dense-P32 accuracy gate;
- W2.5/W3/W3.5 use the generic kernel exactly as before;
- no checkpoint state, persistent/transient VRAM, graph node, or launch was
  added;
- compilation used Ninja `-j4`, one NVCC host thread, and one split-compile
  partition.

Artifacts:

- `artifacts/a41_phase30_h100/fixed_gateup_baseline.json`
- `artifacts/a41_phase30_h100/fixed_gateup_candidate.json`
- `artifacts/a41_phase30_h100/fixed_gateup_rate_gated_candidate.json`
- `artifacts/a41_phase30_h100/fixed_gateup_full_mlp_candidate.json`
- `artifacts/a41_phase30_h100/fixed_w2_gateup_final_candidate.json`
- `artifacts/a41_phase30_h100/production_mlp_fixed_w2_gateup_vs_phase28.json`
- `artifacts/a41_phase30_h100/fixed_w2_gateup_ncu_summary.json`

## Next phase

Phase 31 should preserve this narrow W2 gate and make `K=2048` a device-side
compile-time constant throughout the fixed kernel. The Phase-30 prologue makes
the segment geometry static, but the TMA input shape and K-tile loop still
derive from the runtime `size_k` argument. Replacing those uses only inside the
already-validated fixed specialization may remove additional integer/control
work without perturbing other rates or shapes.
