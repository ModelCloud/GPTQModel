# Experiment 36: historical calibration stability

The fixed native base and read-only F6 seed-7 teacher are retained. Three shuffled
historical Fisher subsets (sampling seeds 71, 72, 73) each contain 64 distinct
eligible documents, with 512 tokens per document. Sampling seeds do not change
the teacher's quantization seed. The verified source hash remains
`5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39`.

Each subset supplies nested prefixes of 2,048, 4,096, 8,192, 16,384, and 32,768
tokens. Sizes within a subset are correlated by construction. Independently
sampled subsets overlap: seeds 71/72 share two documents, 71/73 share one,
and 72/73 share none. They must not be described as mutually disjoint.
The exact selected indices and manifest hashes are in
[the provenance record](results/low-rank/stability-provenance.json).

Three canonical model captures feed 15 fixed-base fits. Each fit tests ranks
2, 4, 6, 8, 12, 16, L2 and tail-alpha-1 fitting, and FP32/FP16 factors.
The tail fit uses 80 steps and calibration-only selection. Every exported
candidate gets all nine localized row counts and export-reload equality.
Fifteen dependent broader-replay jobs evaluate every candidate on the existing
16-document C4 capture; those documents never enter fitting. This replay is an
existing development holdout, not a newly untouched final acceptance set.

The jobs are queued/running, not completed accuracy evidence. Cross-subset pass
rates, spectra and selected ranks remain to be evaluated after completion.

## Smaller sparse recovery follow-up

Layer-0 rank-6 plus 32 sparse exceptions completed its bounded C4 model run:
perplexity 26.97970508 versus window 26.99152524. This small-slice difference
alone does not establish a quality improvement. Full ARC remains in progress.
The raw [model report](results/rank8-targeted/rank6-sparse32-c4.json) retains all
timing samples. M=2 contains large latency excursions (roughly 40–337 ms),
so this run cannot support a stable latency claim without matched remeasurement.

## First completed subset: seed 71, 2,048 tokens

The 24 exports (six ranks × two fits × two factor dtypes) completed all nine
row counts and exact export-reload comparisons. Source export and all read
teacher-shard hashes remained unchanged. [Raw evidence](results/low-rank/stability-seed71-2048.json).

For FP16 factors, combined teacher/window passes are:

| Rank | L2 | Tail alpha 1 |
| ---: | ---: | ---: |
| 2 | 7/9 | 7/9 |
| 4 | 8/9 | 8/9 |
| 6 | 9/9 | 9/9 |
| 8 | 8/9 | 8/9 |
| 12 | 7/9 | 7/9 |
| 16 | 7/9 | 7/9 |

Rank-6 tail has worst teacher MAE 0.00201621 and maximum error 0.04439910.
Rank-16 tail lowers worst MAE to 0.00196325 but raises maximum error to
0.12286758. This shows rank selection is sensitive to calibration composition
and objective; it does not yet establish the cause of the large outlier or
prove rank 6 stable. Broader replay and C4/full-ARC model checks are queued.
No accuracy gate or production default changed.

## Human exception review: rank-2 tail, subset 71 / 2,048 tokens

At M=512, FP16 rank-2 tail reports 1.33298x full-linear speedup over production
window on GPU UUID `GPU-8be4c651-4058-83df-154b-291f1b86add8` (sm80).
Median latency is 0.977920 ms versus 1.303552 ms; the 20-sample ranges are
0.972800–0.988160 ms and 1.262592–1.343488 ms respectively. These are one-run
sample ranges, not confidence intervals or independent timing replications.
Window-relative MAE 0.002065994 passes 0.003; maximum 0.049743652 fails 0.046875.
Teacher-relative MAE/max are 0.002065236/0.049565036. Window itself passes with
teacher-relative MAE/max 0.000024957/0.018733978.

The >25% gain triggers human review under AGENTS.md. A possible exception would
be restricted to this layer/operator and shape, with window fallback elsewhere,
after broader replay, model checks, and repeat timing. It is not approved or
enabled. This candidate is only 1.0094x faster than the passing rank-16 recovery
at the same shape, so the evidence does not establish a compelling tradeoff
against that alternative. Broader/model evidence for this specific fit remains
missing. Other fast failing cases are retained in the machine-readable summary.

`scripts/p32_twenty/summarize_stability.py` checks the complete candidate/row grid
and read-only hash outcomes before counting a local fit as complete. Its output
explicitly separates partial/missing fits from completed local evidence; it does
not certify full-model acceptance or completion of experiment 36.

## Rank-6 sparse model follow-up completed

The earlier rank-6 + 32-exception export now completed all 1,172 ARC examples:
392 raw / 431 normalized correct, versus window 390 / 428. Paired raw wins/losses
are 10/8 (exact p=0.81453), normalized 9/6 (p=0.60724). Prompts, targets, and
sample indices matched exactly before comparison. These results are consistent
with noise, not an established improvement. This is the original 8,192-token
rank-6 sparse candidate, not the new subset-71 rank-6 fit.

[Archived report](results/rank8-targeted/rank6-sparse32-arc.json) and
[paired statistics](results/rank8-targeted/rank6-sparse32-arc-paired.json).
The original generic `recovery_contract` text incorrectly calls all corrections
FP32; the archive annotates this without rewriting the recorded run. Future
reports now record A/B dtypes, logical rank, and sparse count explicitly.
