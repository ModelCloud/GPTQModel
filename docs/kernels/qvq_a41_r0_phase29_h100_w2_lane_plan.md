# Phase 29: rejected H100 W2 lane-static decode plan

Phase 29 independently tested whether Phase 28's lane-static window geometry
should be extended from W2.5 to W2. The candidate was exact and passed all 59
grouped Hopper tests, but grouped gate/up latency regressed by 4.6% geometric
mean. It is fully removed; production remains the Phase-28 source at
`ace9b512`.

## Exact experiment

For transition width `E=4`, the candidate precomputed the same lane descriptor
used by W2.5/W3/W3.5:

```text
pair0 = 16 * (lane & 3) + (lane >> 2)
bit0 = (127 - pair0) * E
bit1 = bit0 - 8 * E

plan = {
  bit0 >> 5, next(bit0 >> 5), bit0 & 31,
  bit1 >> 5, next(bit1 >> 5), bit1 & 31,
  lane & 3
}
```

The hot loop used the descriptor for the unchanged four state extractions.
No quantization state, selector, PGC operation, level lookup, WGMMA order,
launch geometry, or storage changed.

W2's four-bit geometry is special: every state position has simpler power-of-
two relationships. `ptxas` already reduces the compact source to essentially
the same dynamic instruction stream. Forcing the explicit descriptor changes
dependency scheduling without deleting work.

## H100 CUDA-event result

Both children are `K=2048, N=8192`; timing includes grouped inner P32 plus
paired recovery. The physical 132-SM H100 passed the strict 0% utilization / 0
MiB admission gate. Results use 30 warmups, 200 CUDA-event samples, and 50
warmed CUDA Graph replays per sample.

| M/K/N per child | Phase 28 W2 | Candidate | vs Phase 28 | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 32.047 us | 33.445 us | 0.958x | N/A | N/A | No |
| 2/2048/8192 x2 | 31.750 us | 33.316 us | 0.953x | N/A | N/A | No |
| 4/2048/8192 x2 | 32.036 us | 33.670 us | 0.951x | N/A | N/A | No |
| 8/2048/8192 x2 | 32.415 us | 33.981 us | 0.954x | N/A | N/A | No |
| 16/2048/8192 x2 | 33.121 us | 34.746 us | 0.953x | N/A | N/A | No |

The geometric mean is **0.9540x**, a **4.60% regression**, and zero of five
cells improve. Marlin/Machete do not expose the same recovery-inclusive
sub-operation. Because this isolated gate fails decisively, running a full
MLP matrix would only consume H100 time and cannot qualify the candidate.

## Nsight Compute and SASS

Matched Nsight Compute 2026.2.1 profiles used the exact `ace9b512` parent and
the candidate binary, 19 counter-replay passes, and source-correlated SASS.

| W2 M1 metric | Phase 28 | Candidate | Change |
|:--|--:|--:|--:|
| Replay duration | 29.376 us | 30.816 us | **+4.90%** |
| Executed warp instructions | 10,408,466 | 10,408,478 | +12 (neutral) |
| Registers/thread | 48 | 48 | unchanged |
| Static shared memory | 26.240 KiB | 26.240 KiB | unchanged |
| Eligible warps/scheduler/cycle | 0.499 | 0.476 | **-4.55%** |
| Long-scoreboard / issue-active | 1.424 | 1.476 | +3.63% |
| Wait stall / issue-active | 0.952 | 1.108 | **+16.4%** |
| DRAM throughput | 12.01% | 11.45% | -4.69% |

The major dynamic opcode counts are also effectively identical: `IMAD`
changes by 63, branches by 126, while `LDS`, `PRMT`, `LOP3`, `SHF`, `LEA`,
and WGMMA counts do not change. The slowdown therefore comes from worse
instruction/dependency scheduling, not extra mathematical work or memory
capacity.

Profiler reports remain outside Git:

```text
/root/qvq-profiler-artifacts/phase29-w2-lane-plan/baseline_w2_m1.ncu-rep
/root/qvq-profiler-artifacts/phase29-w2-lane-plan/candidate_w2_m1.ncu-rep
```

## Decision and validation

- all M=1,2,4,8,16 results were bit-exact, repeatable, CUDA-Graph stable, and
  within the existing 2e-3 dense-P32 gate;
- all 59 grouped Hopper tests passed across every supported rate;
- no production source or runtime path is retained;
- no persistent/transient VRAM, checkpoint state, or graph node was added;
- build parallelism stayed at Ninja `-j4`, one NVCC host thread, and one
  split-compile partition.

Artifacts:

- `artifacts/a41_phase29_h100/w2_lane_plan_baseline.json`
- `artifacts/a41_phase29_h100/w2_lane_plan_candidate.json`
- `artifacts/a41_phase29_h100/w2_lane_plan_ncu_summary.json`

## Next phase

Phase 30 should stop extending lane-address algebra. The remaining grouped
gate/up cost is dominated by shared level lookup and PGC/state work, and prior
phases closed global state tables, wider N128 tiles, and larger shared tables.
The next experiment should instead test a compile-time two-segment gate/up
launch specialization that removes generic grouped metadata/control from the
kernel prologue while preserving the proven N64 work geometry and completely
independent child payloads.
