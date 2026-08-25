# Transition-16 banked CPU dispatch decision

## Decision

**KEEP the existing transition-16 legacy dispatch.** Paired measurements found G-only 2.67x to 2.91x slower by median at batches 32 through 128 (2.89x to 3.66x on minima). The FP64 sweep found no measurable quality difference: its raw 10-to-8 tally is aliased and collapses to 3-to-2 across distinct generated data instances, with every difference in the near-tie range. Performance alone supports keeping the current production dispatch.

The sampled regimes split rather than trade axes: G-only won every divergent configuration at batches 8/16 and was faster at batch 16, while legacy won every divergence at batch 32 and was faster there. This is descriptive coverage, not evidence for an accuracy claim. Neither FP32 recurrence is an oracle.

## FP64 ground truth

The adjudicator generated the same float32 input and codebook tensors for both native recurrences, then recomputed every divergent returned row's total squared error in float64 directly from the original tensors. It never used either kernel's returned `squared_error`. Quality adjudication ran with `torch.set_num_threads(1)` only; production and the timing measurements used 32 threads, so the near-tie winner set is not established as thread-count invariant.

Matrix: seeds `20260821..20260823`; batches 8/16/32; banks 1/2; steps 16/32; state counts 65,536/131,072; segment lengths 8/16; transition width 16; no overlap, entry, or exit constraints.

| Count | Result |
|---|---:|
| ATTEMPTED | 144 |
| COMPLETED | 72 |
| SKIPPED | 72 |
| DIVERGED | 18 |

The 72 state-count-131,072 attempts were SKIPPED, rather than hidden or allowed to abort the sweep. The adjudicator incorrectly demanded circular closure from unconstrained calls, although the kernel promises closure only when overlap or entry/exit constraints supply it. That predicate has content at 131,072 states and failed symmetrically for both arms, so the skips introduce no arm-selection effect. At 65,536 states `suffix_count == 1`: the transition and closure predicates reduce to `0 == 0` and are vacuous by construction; only the state-range check has content.

The 131,072-state shape is separately unreachable from today's production Python entry point, which requires `codebooks.shape[1] == (1 << 16)` in `gptqmodel/quantization/qvq.py:1590`. The C++ dispatcher itself has no state-count condition, however, so a direct t16/131,072-state call still routes to legacy. That was the only sampled shape with a non-trivial suffix frontier, and this sweep left it unmeasured: it is an explicit coverage gap, not a kernel finding.

Raw, over the 18 divergent completed configurations, legacy was FP64-better in 8 and G-only in 10. That tally is inflated by matrix aliasing: `segment_steps` does not participate in input generation, so segment-8/16 pairs reuse tensors; and the one-bank codebook aliases bank zero of some two-bank cases because both are generated from the same stream. Collapsing 18 configurations to 9 distinct `(seed,batch,banks,steps)` cases and then to 5 distinct `(seed,batch,steps)` data instances gives G-only 3 and legacy 2. At face value even the uncollapsed 10-of-18 result is not significant (two-sided `p ~= 0.81`). The honest conclusion is **no measurable quality difference; observed differences are near-tie noise**. Across all 20 raw divergent rows each recurrence won 10, relative differences ranged `3.01e-06` to `1.09e-04`, and segment bank IDs did not diverge.

By raw divergent configuration, the batch split was exact: G-only won 8/8 at batch 8 and 2/2 at batch 16; legacy won 8/8 at batch 32.

## Paired timing

Settings: 3 warmups per arm; 15 samples per arm; arms run back-to-back with alternating order; 32 Torch/OpenMP threads; explicit singleton `OMP_PLACES` for all 32 cgroup CPUs; affinity asserted once per series. Times are milliseconds and the ratio is G-only / legacy.

| Batch | Legacy median | Legacy min | G-only median | G-only min | Median ratio | Min ratio | Idle check usage delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 397.19 | 343.33 | 65.09 | 45.44 | 0.16x | 0.13x | 265,770 us / 1.000 s |
| 32 | 31.62 | 23.42 | 84.38 | 83.33 | 2.67x | 3.56x | 282,716 us / 1.000 s |
| 64 | 60.26 | 57.37 | 166.27 | 165.81 | 2.76x | 2.89x | 539,855 us / 1.000 s |
| 128 | 116.24 | 90.42 | 338.09 | 331.03 | 2.91x | 3.66x | 306,403 us / 1.000 s |

The cgroup idle checks use `/sys/fs/cgroup/cpu.stat` usage deltas, not host load average. The batch-16 crossover is real in this series and is not generalized to the larger batches.

The measurement build selected the two native arms with a temporary `QVQ_TEST_FORCE_BANKED_LEGACY` / `QVQ_TEST_FORCE_BANKED_G_ONLY` source gate. That gate was deliberately not shipped; the final clean build contains neither environment name. The retained measurement harness is `/home/ubuntu/work/qvq-findings/t16_dispatch_measure.py` on the measurement host.

## Correctness and provenance

- Baseline commit: `df80f33efdd43c90508dbb4c67bb85fa5d2bdc07` (`origin/main` when captured).
- A commit-tagged baseline artifact contains selected states, segment bank IDs, packed words, and packed bank selectors for batches 16/32/64/128.
- The baseline JIT root was initially empty and its log shows all seven translation units compiling; the extension reported ready after 95 seconds.
- Baseline pytest gate 1: `803 passed, 260 skipped`.
- Baseline pytest gate 2: `178 passed, 8 skipped` (the current baseline, three passes above the older count in the task brief).
- No test was added: dispatch behavior is intentionally unchanged, so a test that failed before the comment correction would be artificial.
- The transition-15 G-only rationale in the dispatcher is inherited from earlier work and was not re-derived by this measurement.

Hardware and software are recorded in `CPU_KERNEL_LOG.md`.
