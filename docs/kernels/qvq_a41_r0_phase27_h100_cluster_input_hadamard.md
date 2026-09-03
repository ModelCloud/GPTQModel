# Phase 27: rejected H100 cluster input Hadamard

Phase 27 tested whether Hopper distributed shared memory could collapse the
accepted Phase-26 two-kernel `M x 2048` input transform into one launch. The
candidate was exact, CUDA-Graph safe, and used no additional persistent or
transient storage, but it was rejected because it regressed the repeat
geometric mean and three of the five material decode batch sizes.

The production source remains identical to Phase 26. No cluster operator,
dispatch, or model-runtime change is retained.

## Mathematical contract

For each logical row, the production transform computes

```text
z = round_fp16_unless_overflow(x * SU)
y = H_2048(z / fp16(sqrt(2048)))
```

in the established ascending butterfly order. `H_2048` factors into eight
independent length-256 low transforms followed by three high butterflies:

```text
H_2048 = H_high(bits 256, 512, 1024) o H_low(bits 1..128).
```

Phase 26 materializes the eight low tiles in the padded `16 x 2048` output,
then launches the high stage in place. The Phase-27 experiment instead used
one Hopper thread-block cluster per logical row:

```text
cluster ranks 0..7
  each computes one 256-column low tile in local shared memory
  cluster barrier
  ranks 0..3 read all eight tiles through distributed shared memory
  ranks 0..3 compute high bits 256/512/1024 and store the valid row
  cluster barrier
  all ranks cooperatively zero padded rows
```

The second cluster barrier is mandatory: no rank may retire and release its
shared allocation while another rank can still read that allocation.

All arithmetic used native packed FP16 addition/subtraction in precisely the
same order as Phase 26. The candidate was byte-exact at M=1,2,4,8,16,
including CUDA Graph replay and the existing pre-scale-overflow rescue case
on a non-default stream.

## H100 CUDA-event result

The repeat below is the promotion decision. It used the physical 132-SM H100
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, an idle-memory gate of 0 MiB, 30
warmups, 200 samples, 50 CUDA Graph replays per sample, and CUDA-event timing.
Marlin and Machete do not expose an equivalent transform-only operation, so
their cells are not applicable; the unchanged Phase-26 full-MLP comparison
remains the production baseline.

| M/K/N | Phase 26 | Cluster | vs Phase 26 | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|---:|---:|---:|---:|---:|---:|:---:|
| 1/2048/2048 | 3.584 us | 3.728 us | 0.961x | N/A | N/A | No |
| 2/2048/2048 | 3.564 us | 3.729 us | 0.956x | N/A | N/A | No |
| 4/2048/2048 | 3.579 us | 3.626 us | 0.987x | N/A | N/A | No |
| 8/2048/2048 | 3.792 us | 3.597 us | 1.054x | N/A | N/A | Yes |
| 16/2048/2048 | 3.834 us | 3.845 us | 0.997x | N/A | N/A | No |

The repeat geometric mean is **0.9904x**, a **0.97% regression**, with only
one of five cells improving. The first independent run was noisy in the
opposite direction at 1.0051x, but it still regressed M1 and M4. The repeated
failure across the latency-critical M1-M4 range rules out promotion without
needing to perturb the production full MLP.

## Nsight Compute and assembly evidence

Nsight Compute 2026.2.1 collected 19 hardware-counter replay passes on M1
using `SpeedOfLight`, `LaunchStats`, `Occupancy`, `SchedulerStats`,
`WarpStateStats`, `InstructionStats`, and `MemoryWorkloadAnalysis`, with
source-correlated SASS enabled.

| Metric, M1 | Phase-26 low + high | Phase-27 cluster |
|:--|--:|--:|
| Executed warp instructions | 8,736 | 15,004 |
| Instruction change | baseline | **+71.7%** |
| Registers per thread | 16 / 24 | 28 |
| Cluster size | none | 8 blocks |
| Grid size | 8 + 64 blocks | 8 blocks |
| Eligible warps/scheduler/cycle | 0.113 / 0.062 | 0.106 |
| Achieved occupancy | 11.59% / 1.75% | 12.88% |
| DRAM throughput | 0.189% / 0.116% | 0.156% |

The cluster remains nowhere near an HBM limit. Its source-correlated dynamic
assembly contains 2,788 `IMAD`, 1,504 branches, 1,120 integer comparisons,
996 address instructions, and 352 synchronization instructions. Each of the
two cluster-wide barriers expands into arrival/wait/error-barrier and cache
invalidation machinery across all resident warps. Mapping and loading eight
remote shared allocations also adds address generation that the Phase-26
global intermediate does not need.

Nsight replay duration alone appears favorable because it excludes the same
normal-launch behavior as CUDA Graph timing and perturbs tiny grids. The
promotion decision therefore uses the ordinary CUDA-event matrix; the
counter result explains why that matrix does not improve.

Profiler report (kept outside Git):

```text
/root/qvq-profiler-artifacts/phase27-cluster-input/m1_cluster.ncu-rep
```

## Decision

The candidate is fully removed. The result reinforces the earlier Hopper
cluster rule: distributed shared memory pays only when it removes enough
global reduction work to compensate for cluster-wide synchronization. A
single 2048-value FP16 intermediate and one tiny launch are cheaper here.

Committed artifacts:

- `artifacts/a41_phase27_h100/cluster_input_hadamard_experiment.json`
- `artifacts/a41_phase27_h100/cluster_input_hadamard_repeat.json`

The experimental operator and its temporary drivers were deliberately
removed so the rejected path cannot be selected or compiled from the current
tree.

## Next phase

Phase 28 returns to the dominant grouped gate/up P32 decode. It should test
rate-specific reuse of decoded state and address work across the two sibling
output segments. Any candidate must retain child-local selector streams,
alternative-bank IDs, split schedules, and deterministic reduction order.
