# Small-rank single-projection model comparison

All 28 L2/tail × FP32/FP16 × rank0/2/4/6/8/12/16 model runs completed.
Only layer-0 down is replaced; all remaining projections retain the window
model's execution. Each GPU has a separately measured production-window baseline.
The fixed F6 seed-7 teacher and native source export remain unchanged.

| Candidate | C4 PPL (16 documents, 4080 scored tokens) |
|---|---:|
| Window | 26.99152524 |
| Rank8 tail FP16 | 26.97698895 |
| Rank12 L2 FP16 | 26.99006036 |
| Rank16 L2 FP32 | 26.98990935 |

All 28 mean document-NLL differences have exploratory paired bootstrap 95%
intervals containing zero (10,000 resamples of 16 documents, seed7). These are
noise-consistent effects on a small slice, not a predeclared model acceptance
result or proof of a P32 post-quant advantage. Original BF16 on the same slice
was better at PPL 25.38769779.

Rank8 tail FP16 versus window has mean logits KL 0.0018380879, top1 agreement
0.9775391, top5 0.9740723, and top10 0.9757568. These are propagated model
diagnostics, not localized kernel gates. Its tiny local window max-error failure
remains; the model PPL observation does not waive that gate.

Raw model reports retain all nine prefill row counts and growing-cache decode
samples. No 2x model speedup is established by these single-projection replacements.
CPU logits comparisons for the remaining candidates are being collected; a missing
comparison in the current summary is pending, not zero error.

[Raw model results and exploratory intervals](results/low-rank-model/summary.json)
include all 28 candidates and same-device prefill ratios. Timing changes this small
need repeated matched runs before a latency claim; the decode trace retains the
previous cold-first-step limitation.

## Next validation stage

Full ARC-Challenge (all 1172 test examples) is queued for window, original BF16,
canonical FP32, rank12 L2 FP16, rank16 L2 FP32 and borderline rank8 tail FP16.
These task examples never enter fitting. Broader local replay is also queued for
all 42 exports on 16 independently contextualized C4 documents (4096 tokens).
The first eight source documents overlap the original concatenated capture, so
this is expanded-context/document coverage, not an entirely unseen test corpus.

For experiment37, both historical calibration and held-out activation captures
now cover all 16 down projections: 8192 and 4096 tokens respectively. Native
fitting and smallest-rank selection across those layers remain outstanding.

The dispatcher now supports append-only manifest updates while running. The
active extended manifest/state are `/root/p32-low-rank/extended-queue.json` and
`extended-queue-state.json`. The previous dispatcher was replaced only after all
its child workers had exited; no GPU job was interrupted. It immediately started
the captures and full-ARC jobs. All four GPUs remained assigned, including the
long-running canonical GSM8K teacher, which is still being allowed to finish.
