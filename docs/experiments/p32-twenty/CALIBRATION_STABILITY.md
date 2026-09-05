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

## Large-capture percentile correction

Seed-71 16K/32K workers terminated at `torch.quantile`: the residual arrays exceed
its 2^24-element input limit. Their partial exports/reports are retained and are
not counted as completed fits. Explicit retry1 fits and replay jobs use separate
output directories. Original dependent replay jobs remain unsatisfied and must
not be mistaken for runnable work or successful validation.

Above the limit, tail fitting now computes the same linear-interpolated 99.9th
percentile over every residual using CPU NumPy with explicit FP64 conversion.
No residual subsampling is used. Smaller captures retain their original Torch
path; rounding can differ between paths and is not claimed bit-identical.
A 2^24+1-element ordered-array check and a small Torch FP64 comparison validate
the fallback. The first check caught NumPy preserving FP32 arithmetic; explicit
FP64 conversion fixed that discrepancy before this commit. GPU model validation
of the large-fit path remains pending the retry runs.

## Five-fit interim evidence

Five local fits are complete. Seed 72 at 2,048 tokens has no FP16 candidate
passing all cases, unlike seed 71 at the same size. Seed 72 at 4,096 tokens
passes rank-6 tail and ranks 8/12/16 with either fit. Thus the smallest passing
rank varies with sample composition; the seed-71 rank-6 result is not a stable
cross-subset acceptance claim. Broader replay and the remaining sizes are still
required. The progress summary records every observed attempt and uses the
newest retry directory without erasing original partial reports.

## Local calibration grid completed

All 15 fits completed, including the separately recorded seed-71 large-capture
retries: 360 exports and 3,240 local export-reload cases. Every completed fit
verified its source export and read teacher-shard hashes unchanged. The full
reports for every seed/size are archived beside the progress summary.

No FP16 rank/fit passes all 15 subsets/sizes: seed72/2048 has no passing choice.
At 8,192 tokens, rank6 tail passes all three subsets; rank8 does not pass seed71.
At 16,384 and 32,768 tokens, rank6 L2 and tail pass all three subsets. This is
local evidence, not completion of the full calibration-stability scorecard:
broader document pass distributions, model checks and sensitivity analysis remain.

## Broader replay interim: 14/15 completed

Each completed replay evaluated 24 exports on 16 document cases plus nine
prefix row counts. Seed72/8192 stopped at the idle-GPU preflight before evaluation;
an explicit separate-output retry is queued. The failure remains in queue history.
[Interim candidate-level pass counts and failures](results/low-rank/stability-broader-interim.json)
include source paths/hashes; missing replay is not counted as passing.

Rank6 tail at seed72/4096 passes the original nine cases but only 23/25 replay
cases. In contrast, rank6 L2 and tail at 16K/32K pass both original and replay
cases across all three subsets. These are still development activation tests,
not untouched full-model confirmation or proof of P32's advantage over BF16.

The final seed72/8192 retry completed. All 15 broader replays now cover 360
exports and 9,000 cases; [completed replay summary](results/low-rank/stability-broader-completed.json)
retains every failed case. Completion here means evaluation ran, not every
candidate passed or experiment36 met its full model/stability scorecard.

## Three-subset 16K rank6 model results

All three rank6 tail FP16 candidates completed C4 and full ARC, replacing only
layer0 down and retaining window elsewhere. C4 PPL for seeds71/72/73 is
26.98711266 / 26.98697176 / 26.98477121 (window 26.99152524).
Raw ARC correct is 391 / 393 / 392; normalized correct is 433 for every subset
(window raw390 / normalized428). Normalized paired wins/losses are 12/7, 12/7,
and 11/6, with exact p=0.35928, 0.35928, 0.33231. No statistically established
improvement follows. This supports bounded consistency across these fits, not
universal stability, post-quant advantage, or replacement of other layers.

[Reports and paired summary](results/low-rank-model/stability/interim-summary.json)
retain complete metrics; 32K model arms are separate and still running.
