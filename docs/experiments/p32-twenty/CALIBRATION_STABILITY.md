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
