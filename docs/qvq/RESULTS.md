# Transition-16 banked CPU dispatch decision

## Decision

**KEEP the existing transition-16 legacy dispatch.** This is an explicit speed/quality tradeoff, not an oracle or correctness claim. A wider independent FP64 sweep slightly favored G-only by configuration, 10 to 8, but paired measurements found G-only 2.67x to 2.91x slower at batches 32 through 128. The kernel comment now states that tradeoff and the batch-16 crossover plainly.

This sweep agrees with the direction of the earlier 5-to-3 adjudication, but the wider result is substantially narrower. Neither FP32 recurrence is an oracle.

## FP64 ground truth

The adjudicator generated the same float32 input and codebook tensors for both native recurrences, then recomputed every divergent returned row's total squared error in float64 directly from the original tensors. It also independently checked state range, every adjacent transition, and circular tail-biting closure.

Matrix: seeds `20260821..20260823`; batches 8/16/32; banks 1/2; steps 16/32; state counts 65,536/131,072; segment lengths 8/16; transition width 16; no overlap, entry, or exit constraints.

| Count | Result |
|---|---:|
| ATTEMPTED | 144 |
| COMPLETED | 72 |
| SKIPPED | 72 |
| DIVERGED | 18 |

The 72 state-count-131,072 attempts were SKIPPED, rather than hidden or allowed to abort the sweep: both unconstrained candidates failed circular closure and are not packable by the production 16-bit trellis format. All 72 production state-count-65,536 configurations completed and both candidates passed range, transition-consistency, and tail-biting checks.

Over the 18 divergent completed configurations, legacy was FP64-better in 8 and G-only in 10 (no ties or mixed-winner configurations). Across all 20 divergent rows, each recurrence won 10. Relative differences ranged from `3.01e-06` to `1.09e-04`. Segment bank IDs did not diverge in this matrix.

## Paired timing

Settings: 3 warmups per arm; 15 samples per arm; arms run back-to-back with alternating order; 32 Torch/OpenMP threads; explicit singleton `OMP_PLACES` for all 32 cgroup CPUs; affinity asserted once per series. Times are milliseconds and the ratio is G-only / legacy.

| Batch | Legacy median | Legacy min | G-only median | G-only min | Median ratio | Min ratio | Idle check usage delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 397.19 | 343.33 | 65.09 | 45.44 | 0.16x | 0.13x | 265,770 us / 1.000 s |
| 32 | 31.62 | 23.42 | 84.38 | 83.33 | 2.67x | 3.56x | 282,716 us / 1.000 s |
| 64 | 60.26 | 57.37 | 166.27 | 165.81 | 2.76x | 2.89x | 539,855 us / 1.000 s |
| 128 | 116.24 | 90.42 | 338.09 | 331.03 | 2.91x | 3.66x | 306,403 us / 1.000 s |

The cgroup idle checks use `/sys/fs/cgroup/cpu.stat` usage deltas, not host load average. The batch-16 crossover is real in this series and is not generalized to the larger batches.

## Correctness and provenance

- Baseline commit: `df80f33efdd43c90508dbb4c67bb85fa5d2bdc07` (`origin/main` when captured).
- A commit-tagged baseline artifact contains selected states, segment bank IDs, packed words, and packed bank selectors for batches 16/32/64/128.
- The baseline JIT root was initially empty and its log shows all seven translation units compiling; the extension reported ready after 95 seconds.
- Baseline pytest gate 1: `803 passed, 260 skipped`.
- Baseline pytest gate 2: `178 passed, 8 skipped` (the current baseline, three passes above the older count in the task brief).
- No test was added: dispatch behavior is intentionally unchanged, so a test that failed before the comment correction would be artificial.

Hardware and software are recorded in `CPU_KERNEL_LOG.md`.
