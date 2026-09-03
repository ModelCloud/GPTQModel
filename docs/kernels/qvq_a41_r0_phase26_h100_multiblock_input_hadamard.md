# QVQ A41/R0 Phase 26: H100 multiblock shared input Hadamard

Phase 26 is promoted for the physical NVIDIA H100 Llama 3.2 1B A41/R0
path. It replaces the one-block-per-row `SU -> Hadamard -> M16 padding`
operation with an exact two-stage N=2048 factorization. Quantization,
canonical P32 storage, grouped payloads, and the Hopper P32 kernel are
unchanged.

Production executable commit: `acc5467c595f73661e4017dd83ca920c92fc5166`.

## Motivation

The Phase-25 W3/M1 Nsight Systems graph attributed 8.544 us to the shared
N=2048 input transform, versus 3.615 us for the complete down recovery tail.
The generic transform used one 1,024-thread block per logical row. At M=1 it
therefore exposed one block to a 132-SM H100 while executing all eleven
Hadamard stages and zero-padding the remaining M16 rows.

## Exact factorization

For input row `x`, shared scale `SU`, and `D = half(sqrt(2048))`, the existing
range-safe mode-2 initialization is retained exactly:

```text
p_i = float(x_i) * float(SU_i)
r_i = half(p_i), unless that narrowing overflows
v_i = half(float(r_i) / float(D))
```

The reference then executes ascending butterfly bits

```text
1, 2, 4, ..., 1024
```

with one FP16 rounding after every add/sub. Since bits 1 through 128 never
cross a contiguous 256-column tile, N=2048 decomposes into eight independent
low tiles:

```text
low CTA(tile t): bits 1..128 for columns [256t, 256(t+1))
global FP16 boundary
high thread(local c): bits 256, 512, 1024 across all eight tiles
```

The workspace is FP16 because the original single-block implementation also
stores every butterfly result in FP16 shared memory. The low-stage global
store therefore occurs at an already-existing numerical boundary, not a new
rounding point.

Adjacent scalar columns become one `half2` only after bit 1 has produced
`[half(a+b), half(a-b)]`. Bits 2 through 1024 apply the identical butterfly
to both lanes with `HADD2`/`HSUB2`; no lane is mixed and no expression is
reassociated.

## Launch and storage design

| Property | Phase 25 one-block | Phase 26 low | Phase 26 high |
|:--|--:|--:|--:|
| Grid at M1 | 1 block | 8 blocks | 4 x 16 = 64 blocks |
| Threads/block | 1,024 | 256 | 32 |
| Columns/thread | 2 over loop iterations | 1 initially, then one packed pair on even lanes | 2 |
| Butterfly bits | 1..1024 | 1..128 | 256..1024 |
| Registers/thread | 24 | 16 | 24 |
| User shared memory/block | 4.224 KiB dynamic | 512 B static | 0 |
| Register/shared spilling | 0 | 0 | 0 |

The output remains the existing `16 x 2048` FP16 padded activation. The low
stage writes valid logical rows into this tensor and the high stage transforms
them in place after first loading all eight tile values into registers. High
blocks for rows `M..15` write exact zero without reading uninitialized data.
There is no extra transient allocation, persistent allocation, checkpoint
storage, or VRAM cache.

The runtime gate requires all of:

```text
physical device name = NVIDIA H100
compute capability   = 9.0
input Hadamard       = enabled
K                    = 2048
M                    = 1, 2, 4, 8, or 16
```

H200 and unsupported shapes retain the established generic path.

## Isolated H100 result

The committed production-SHA artifact uses 30 warmups, 200 samples, and 50
CUDA Graph replays per sample. CUDA events enclose device graph replay; the
admission gate requires three consecutive 0%-utilization, 0-MiB readings.

| M/K | One-block us | Multiblock us | Speedup | Better |
|--:|--:|--:|--:|:--:|
| 1/2048 | 9.133 | 3.495 | 2.613x | Yes |
| 2/2048 | 8.084 | 3.646 | 2.217x | Yes |
| 4/2048 | 7.415 | 3.610 | 2.054x | Yes |
| 8/2048 | 7.172 | 3.695 | 1.941x | Yes |
| 16/2048 | 6.984 | 3.762 | 1.857x | Yes |

The isolated geometric-mean speedup is **2.1207x**, and all five M values
improve byte-exactly.

## Complete Llama 3.2 1B MLP

Every row includes shared input recovery, grouped gate/up P32, exact output
recovery, SiLU/product/down preconditioning, split-16 down P32, and exact
down recovery. `vs` is baseline latency divided by QVQ latency; values below
one mean the W4 baseline is faster. `Better` is the strict median comparison
with the merged-tip Phase-25 artifact.

| W | M/K/N | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 25 | Better? |
|--:|:--|--:|--:|--:|--:|--:|:--:|
| W2 | 1/2048/8192 x2; 1/8192/2048 | 53.380 | 1.886 | 0.547x | 0.929x | 1.108x | Yes |
| W2 | 2/2048/8192 x2; 2/8192/2048 | 53.656 | 3.752 | 0.582x | 0.921x | 1.079x | Yes |
| W2 | 4/2048/8192 x2; 4/8192/2048 | 54.324 | 7.412 | 0.578x | 0.920x | 1.066x | Yes |
| W2 | 8/2048/8192 x2; 8/8192/2048 | 54.759 | 14.706 | 0.536x | 0.915x | 1.062x | Yes |
| W2 | 16/2048/8192 x2; 16/8192/2048 | 56.110 | 28.705 | 0.580x | 0.895x | 1.052x | Yes |
| W2.5 | 1/2048/8192 x2; 1/8192/2048 | 54.495 | 1.847 | 0.535x | 0.910x | 1.099x | Yes |
| W2.5 | 2/2048/8192 x2; 2/8192/2048 | 54.677 | 3.682 | 0.571x | 0.904x | 1.078x | Yes |
| W2.5 | 4/2048/8192 x2; 4/8192/2048 | 55.346 | 7.275 | 0.567x | 0.903x | 1.067x | Yes |
| W2.5 | 8/2048/8192 x2; 8/8192/2048 | 55.999 | 14.381 | 0.524x | 0.895x | 1.058x | Yes |
| W2.5 | 16/2048/8192 x2; 16/8192/2048 | 57.037 | 28.238 | 0.570x | 0.880x | 1.053x | Yes |
| W3 | 1/2048/8192 x2; 1/8192/2048 | 51.633 | 1.950 | 0.565x | 0.961x | 1.108x | Yes |
| W3 | 2/2048/8192 x2; 2/8192/2048 | 51.826 | 3.885 | 0.602x | 0.953x | 1.082x | Yes |
| W3 | 4/2048/8192 x2; 4/8192/2048 | 52.244 | 7.707 | 0.601x | 0.957x | 1.068x | Yes |
| W3 | 8/2048/8192 x2; 8/8192/2048 | 52.896 | 15.224 | 0.554x | 0.947x | 1.062x | Yes |
| W3 | 16/2048/8192 x2; 16/8192/2048 | 54.021 | 29.815 | 0.602x | 0.929x | 1.056x | Yes |
| W3.5 | 1/2048/8192 x2; 1/8192/2048 | 54.601 | 1.844 | 0.534x | 0.909x | 1.091x | Yes |
| W3.5 | 2/2048/8192 x2; 2/8192/2048 | 54.614 | 3.686 | 0.572x | 0.905x | 1.077x | Yes |
| W3.5 | 4/2048/8192 x2; 4/8192/2048 | 55.075 | 7.311 | 0.570x | 0.908x | 1.074x | Yes |
| W3.5 | 8/2048/8192 x2; 8/8192/2048 | 55.295 | 14.564 | 0.530x | 0.906x | 1.073x | Yes |
| W3.5 | 16/2048/8192 x2; 16/8192/2048 | 56.468 | 28.522 | 0.576x | 0.889x | 1.058x | Yes |

Geometric means are **1.0734x versus Phase 25**, **0.5644x versus Marlin W4**,
**0.9166x versus Machete W4**, and **2.5740x versus ordinary per-module
QVQ**. All 20 cells improve. The current range is 51.633--57.037 us. W4 is a
figurative dense-equivalent efficiency baseline, not an equal-rate or
equal-quality comparison.

## Nsight Compute and SASS

Nsight Compute 2026.2.1 collected 19 hardware-counter replay passes per
control using `SpeedOfLight`, `LaunchStats`, `Occupancy`, `SchedulerStats`,
`WarpStateStats`, `InstructionStats`, and `MemoryWorkloadAnalysis`.

| Metric, M1 | One-block | Multiblock low | Multiblock high |
|:--|--:|--:|--:|
| NCU replay duration | 10.304 us | 3.296 us | 2.752 us |
| Executed warp instructions | 43,520 | 7,072 | 1,664 |
| Eligible warps/scheduler/cycle | 1.878 | 0.113 | 0.062 |
| Achieved occupancy | 48.26% | 11.59% | 1.75% |
| Long-scoreboard cycles/issued instruction | 1.490 | 7.564 | 1.782 |
| DRAM throughput | 0.090% | 0.189% | 0.116% |
| Combined memory throughput | 3.257% | 9.945% | 11.857% |

The two candidate kernels execute 8,736 warp instructions combined, **79.9%
fewer** than the reference. Source-correlated SASS for the low stage includes
640 `HADD2`, 320 `HFMA2`, 320 `SHFL`, and 320 `FFMA`; the high stage uses 48
`HADD2`, 48 `HFMA2`, and 32 `LDG` instructions for the one valid row. Its 512
`STG` instructions include exact zero stores for all fifteen padded rows.
There are no local or shared spills.

The low stage's higher long-scoreboard value does not imply an HBM limit:
DRAM throughput stays below 0.2%. The normal CUDA-event result, not replay
duration, is the promotion timing.

Reports remain outside Git:

```text
/root/qvq-profiler-artifacts/phase26-input-hadamard/one_block.ncu-rep
/root/qvq-profiler-artifacts/phase26-input-hadamard/multiblock.ncu-rep
```

## Nsight Systems graph result

Nsight Systems 2026.4.1 captured one warmed W3/M1 production CUDA Graph with
node tracing enabled. The two input nodes are 1.600 us (low) and 1.088 us
(high), or 2.688 us combined. The complete ten-node instrumented GPU sum is
52.703 us, down from Phase 25's 58.495 us. The dominant nodes are now grouped
gate/up P32 at 25.792 us and split-16 down P32 at 12.736 us.

```text
/root/qvq-profiler-artifacts/phase26-input-hadamard/full_mlp_graph_nodes.nsys-rep
```

## Validation

- all M1/M2/M4/M8/M16 outputs are bit-exact to the established mode-2 direct
  padded transform and exact under repeated CUDA Graph replay;
- a pre-scale overflow rescued by post-normalization is finite and bit-exact
  on a non-default CUDA stream;
- the real Llama layer preserves logits inside the existing 2e-3 dense gate,
  exact repeated logits, exact generated tokens, cache lifecycle, and
  production telemetry;
- 86 focused H100 tests pass, including the independent 59-case grouped
  Hopper P32 suite and all Phase-25 split-recovery cases;
- build parallelism is capped at Ninja `-j4`, one NVCC host thread, and one
  split-compile partition;
- Compute Sanitizer is unavailable on this host, so memcheck/synccheck are
  recorded as not run.

Artifacts and drivers:

- `artifacts/a41_phase26_h100/multiblock_input_hadamard_experiment.json`
- `artifacts/a41_phase26_h100/production_mlp_multiblock_input_vs_phase25.json`
- `scripts/benchmark_qvq_phase26_multiblock_input_hadamard.py`
- `scripts/profile_qvq_phase26_input_hadamard.py`

## Next phase

The shared input transform is no longer the largest removable boundary. W3
is within 3--7% of Machete W4 across the target M values, and grouped gate/up
P32 owns roughly half of W3/M1 graph time. Phase 27 should first test whether
an eight-block Hopper cluster can execute the low and high N=2048 stages in
one launch using distributed shared memory. This is permitted only if it
eliminates the entire intermediate/global launch boundary and remains
bit-exact; the earlier partial DSM reductions that still required a final
kernel are explicitly not repeated. If the cluster does not beat the current
3.5-us CUDA-event control, work returns to rate-specific grouped gate/up
decode reuse rather than further Hadamard micro-tuning.
