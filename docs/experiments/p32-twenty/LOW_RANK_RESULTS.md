# Fixed-base small-rank results

All 42 exports (seven ranks × three fits × two factor dtypes) completed all nine
row-count checks: **378/378 export-reload comparisons were bit-identical**. Native
base fingerprints match the original passing joint-rank16 export in every case.
Source export and read teacher-shard hashes are unchanged. This is partial evidence
for experiments 31/32/33/35, not full model acceptance.

The table gives passes against canonical FP32 / deployed window P32. All rows use
the same historical 8192-token Fisher capture; held-out cases are prefixes of one
separate 2048-token C4 activation capture, not nine independent datasets.

| Rank | L2 FP32 pass | Tail FP32 pass | Tail FP16 pass | L2 FP32 BPW | L2 FP16 BPW |
|---:|---:|---:|---:|---:|---:|
|0|0/9 / 0/9|0/9 / 0/9|0/9 / 0/9|4.251583|4.251583|
|2|7/9 / 7/9|7/9 / 7/9|7/9 / 7/9|4.290706|4.271175|
|4|7/9 / 7/9|7/9 / 7/9|7/9 / 7/9|4.329769|4.290706|
|6|8/9 / 8/9|8/9 / 8/9|8/9 / 8/9|4.368831|4.310237|
|8|8/9 / 8/9|9/9 / 8/9|9/9 / 8/9|4.407894|4.329769|
|12|9/9 / 9/9|9/9 / 9/9|9/9 / 9/9|4.486023|4.368836|
|16|9/9 / 9/9|9/9 / 9/9|9/9 / 9/9|4.564148|4.407898|

**Rank12 is the smallest tested rank passing both references** with the current
L2 or tail fits and either factor dtype. Tail rank8 is borderline: its M2048 max
is 0.04684423 versus canonical FP32 but 0.04691315 versus window (FP32 factors).
FP16 factors yield 0.04683660 and 0.04690552 respectively. Window max exceeds
0.046875 by 0.00003815 / 0.00003052; no exception has been applied.

Weight-space SVD truncation fails all nine cases for ranks 2–12 with either
factor dtype. Rank16 truncation passes, showing the control preserves the complete
correction; its lower-rank components are not the output-aware reduced-rank fit.

The stored projected-residual energy fractions are the **optimal L2 energy bound
for each rank**, not measured energy recovery of tail or weight-space-truncated
factors. Complete spectra are retained in raw reports. Future source output names
this quantity explicitly to prevent confusion.

Production-window full-layer speedups for ranks 2–16 are modest, approximately
0.88–1.10x at M1/16/2048 across fits and dtypes. They must not inherit the larger
speed claims measured with the earlier Python/FP32-transform comparator.
FP16 factors reduce bytes but their casts and small GEMMs do not reliably improve
latency. Rank-specific fusion/graphs remain to test.

## Scoped exception review: rank-zero negative control

On GPU1 (PG506 sm80), layer0 down, M16/K8192/N2048, the fixed native base without
correction takes median 0.193536 ms versus production window 0.277504 ms: **1.434x**.
Twenty warmed CUDA-event samples span 0.186368–0.208896 ms versus
0.266240–0.294912 ms. This is full local linear latency, not model speed.
Canonical MAE/max are 0.00545906/0.231266 versus limits 0.003/0.046875;
window-relative MAE/max are 0.00545857/0.25. Window itself passes with
canonical MAE/max 0.00005529/0.01873398. This is a substantial quality tradeoff,
not the narrow rank8 max miss. A human may explicitly authorize an exception
scoped to this control, but no such approval is assumed and window remains the
default. Broader/model evidence is pending; negative-control model jobs are
diagnostic only. No ranked candidate here establishes the requested 2x target.

The automatic queue has started same-device window baselines and single-module
model replacements. PPL/logits/prefill/decode and subsequent downstream tests
must be evaluated before choosing a deployable rank.

Raw reports and exact serialized byte counts: [results/low-rank](results/low-rank).
