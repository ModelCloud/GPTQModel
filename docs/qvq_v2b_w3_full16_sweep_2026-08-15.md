# QVQ banked-V2 W3/W3.5 full-16-layer sweep (2026-08-15)

This matched sweep compares the W3 and W3.5 extensions of V2B2-P32 and V2B4-P64 on the real Llama 3.2
1B Instruct checkpoint. It supersedes the accidentally launched 64-row run, whose artifacts were isolated under
`qvq_v2b_w3_full16_cal64_mismatch_20260815` and are not included below.

## Configuration

| Setting | Value |
| --- | --- |
| Model | Llama 3.2 1B Instruct |
| Quantized layers/modules | All 16 decoder layers; Q/K/V/O projections |
| Rounding | Block-LDLQ; YAQA disabled |
| Calibration | Dataset rows 0-511; 188,256 non-padding tokens |
| Evaluation | Disjoint dataset rows 512-1023; 172,367 non-padding tokens |
| Sequence handling | Batch 1, no concatenation, no length limit |
| Seed | 18240 |
| Runtime | Python 3.14.6 free-threaded (`-Xgil=0`), PyTorch 2.13.0+cu130 |
| GPUs | Four NVIDIA PG506-230 `sm_80` devices, one arm per GPU |
| Code revision | `c44995c8` |

## Complete results

Top-5 and Top-10 are set-overlap ratios, matching the harness streaming telemetry. Layer KL is measured at
the output of each decoder layer; `Layer-15 KL` is the final decoder-layer value before the LM head.

| Rate | Codec | Effective BPW | Weight MSE | Weight rel-L2 | Local QKVO KL | Live QKVO KL | Mean layer KL | Layer-15 KL | Final-logit KL | Final JSD | Final rel-L2 | Top-1 | Top-5 | Top-10 | Time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W3 | V2B2-P32 | 3.03125 | 1.592692e-5 | 0.161150 | 0.000940 | 0.010423 | 0.014698 | 0.027243 | 0.022030 | 0.005397 | 0.114794 | 94.39% | 90.87% | 90.74% | 1781.71 |
| W3 | V2B4-P64 | 3.03125 | 1.586764e-5 | 0.160828 | 0.000915 | 0.010315 | 0.015951 | 0.025571 | 0.022861 | 0.005567 | 0.115148 | 94.19% | 90.88% | 90.71% | 1518.68 |
| W3.5 | V2B2-P32 | 3.53125 | 8.225882e-6 | 0.115827 | 0.000464 | 0.005222 | 0.007632 | 0.013573 | 0.011569 | 0.002852 | 0.081956 | 95.67% | 93.24% | 93.10% | 1733.42 |
| W3.5 | V2B4-P64 | 3.53125 | 8.150313e-6 | 0.115295 | 0.000466 | 0.005143 | 0.007114 | 0.012825 | 0.011172 | 0.002752 | 0.081320 | 95.86% | 93.29% | 93.16% | 1553.86 |

## V2B4-P64 delta relative to V2B2-P32

Negative loss deltas are improvements. Top-N deltas are percentage points.

| Rate | Weight MSE | Local KL | Live KL | Mean layer KL | Layer-15 KL | Final KL | Top-1 | Top-5 | Top-10 | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W3 | -0.37% | -2.62% | -1.04% | +8.53% | -6.14% | +3.77% | -0.20 pp | +0.01 pp | -0.03 pp | -14.76% |
| W3.5 | -0.92% | +0.47% | -1.51% | -6.79% | -5.51% | -3.43% | +0.19 pp | +0.05 pp | +0.06 pp | -10.36% |

At W3, B4 improves weight/local/live proxies and the last layer, but loses final-logit KL and Top-1; the mean
layer KL also regresses. At W3.5, B4 improves nearly every propagated metric and is faster. The two codecs have
identical selector overhead (0.03125 bpw), so these differences reflect selector geometry rather than payload cost.

## Selector statistics

| Rate | Codec | Entropy (bits) | Nonzero selectors | Selector histogram | B2 module family histogram |
| --- | --- | ---: | ---: | --- | --- |
| W3 | V2B2-P32 | 1.000000 | 50.0000% | 2,621,442 / 2,621,438 | 0 / 22 / 20 / 22 |
| W3 | V2B4-P64 | 1.987409 | 71.6951% | 741,996 / 572,358 / 741,562 / 565,524 | n/a |
| W3.5 | V2B2-P32 | 1.000000 | 50.0414% | 2,619,268 / 2,623,612 | 0 / 25 / 13 / 26 |
| W3.5 | V2B4-P64 | 1.999997 | 74.9980% | 655,413 / 655,680 / 653,455 / 656,892 | n/a |

All final metrics were finite. Raw JSON and logs are stored locally under
`/private/monster/data/model/qvq_v2b_w3_full16_cal512_20260815/`.
