# H100 M960 W2 row-five single-stage TMA gate (2026-09-24 UTC)

The W2 M960 row-five Q projection now uses one TMA stage, matching the
existing W3 row-five schedule. Other geometries retain their established
stage counts. This changes transient CTA staging only: compressed P32 weights,
FP32 output arithmetic, checkpoint representation, and the raw ABI are
unchanged.

On the seed-7 Llama 3.2 1B checkpoint's layer-0 W2 Q metadata (M960,
K2048, N2048, BM80/BN64), 100 warm paired CUDA-event rounds measured
51.168 us for two stages and 45.248 us for one stage (13.08% faster).
The raw FP32 output was bitwise equal, with zero MAE and maximum error.

The production-relevant gate used the merged automatic dead-row and Rank-8
phase defaults: Rank-8 disabled for prefill, enabled for decode. Same H100
GPU UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, GPU-local CPU affinity,
optimized ZML runner, snapshot, GSM8K-Platinum 1,209 requests, B128/M960,
FA2, 544 KV pages, dataset, and reference. Only the QVQ kernel library
changed. Each server was warmed before its full run. The matched order was
control / candidate / candidate / control.

| Arm | Useful prefill tok/s | Padded prefill tok/s | Useful decode tok/s | Padded decode tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control A | 42,639.87 | 49,181.27 | 10,169.47 | 11,872.17 | 35.476 |
| Candidate A | 42,714.73 | 49,267.62 | 10,167.66 | 11,870.07 | 35.441 |
| Candidate B | 42,741.57 | 49,298.58 | 10,171.64 | 11,874.71 | 35.410 |
| Control B | 42,663.84 | 49,208.92 | 10,193.70 | 11,900.47 | 35.439 |

All four arms scored 544/1,209, with zero invalid outputs. Both candidate
runs matched the control's token stream on all 1,209 requests. The candidate
improves full-suite useful prefill by approximately 0.18-0.24% against the
matched controls. This is a small forward step, not a material advance toward
the model-level 2x target; the larger isolated Q speedup is diluted by the
other prefill work. Decode differences are within the observed run band.

Raw full-run records, kept outside git:

- `/var/tmp/w2-single-stage-baseline-full-20260924.json`
- `/var/tmp/w2-single-stage-candidate-full-20260924.json`
- `/var/tmp/w2-single-stage-candidate-repeat-20260924.json`
- `/var/tmp/w2-single-stage-control-after-full-20260924.json`

The first control JSON SHA-256 is
`1dced9accf8bbef62e5fd9c6e02725c404cd47c52aef995f8808492d475c53e9`.
The candidate records cite that exact file as their token-stream reference.
