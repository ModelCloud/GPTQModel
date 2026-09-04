# Phase 83: H100 Qwen linear decoded-level prefetch

Phase 83 applies the already proven decoded-level overlap to the fixed
Qwen3.8-27B linear-input grid at W2 and W3. It improves every affected
M1--M16 cell. W2.5 stays on Phase 82 because a broad trial gave only marginal
gains and regressed M1.

## Scheduling change

The P32 state extraction and pseudo-random-code level lookup for the next
fragment do not depend on completion of the previous warp-group matrix
multiply. The fixed linear kernel now evaluates those eight FP16 levels into
an independent packed register fragment before the fragment-reuse wait:

```text
decode next four P32 state pairs
look up and pack eight levels
wait for the old matrix-multiply source fragment
copy four packed registers into that fragment
issue the next matrix multiply
```

This preserves P32 state math, level ordering, and child-local ordered split
reduction exactly. Only independent instruction scheduling moves.

W2.5 has a different transition width and register/dependency balance. Its
broad-prefetch trial was not promoted; compile-time `TransitionBits != 5`
keeps it on the measured Phase-82 path. Selection is visible through
`h100_qwen_linear_decode_prefetch_launches`.

## Correctness and graph safety

- W2, W2.5 fallback, and W3 pass the grouped same-payload child parity test.
- All three rates are bit-exact across five CUDA Graph replays.
- Maximum dense-P32 FP32 Torch-oracle error is `1.774e-5`; maximum mean
  absolute error is `2.479e-6`.

## H100 benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed with CUDA events, 20 warmups, 60 samples, and 50 replays per
sample. `last/new` compares with committed Phase 82. Marlin and Machete are
figurative W4 projection-sum baselines; ratios below one mean the W4 baseline
is faster. W2.5 is an unchanged control, so its sub-percent yes/no variation
is recorded but is not attributed to Phase 83.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x10240 + 1x5120x6144 | 89.549 | 0.2603x | 0.5563x | 1.0133x | yes |
| 2 | 2 | 2x5120x10240 + 2x5120x6144 | 91.026 | 0.2650x | 0.5407x | 1.0157x | yes |
| 2 | 4 | 4x5120x10240 + 4x5120x6144 | 92.061 | 0.2640x | 0.5336x | 1.0094x | yes |
| 2 | 8 | 8x5120x10240 + 8x5120x6144 | 92.124 | 0.2518x | 0.5327x | 1.0089x | yes |
| 2 | 16 | 16x5120x10240 + 16x5120x6144 | 95.821 | 0.2815x | 0.5155x | 1.0112x | yes |
| 2.5 | 1 | 1x5120x10240 + 1x5120x6144 | 90.598 | 0.2572x | 0.5498x | 0.9941x | no |
| 2.5 | 2 | 2x5120x10240 + 2x5120x6144 | 92.548 | 0.2607x | 0.5318x | 1.0004x | yes |
| 2.5 | 4 | 4x5120x10240 + 4x5120x6144 | 93.261 | 0.2606x | 0.5267x | 0.9992x | no |
| 2.5 | 8 | 8x5120x10240 + 8x5120x6144 | 93.534 | 0.2480x | 0.5246x | 1.0000x | yes |
| 2.5 | 16 | 16x5120x10240 + 16x5120x6144 | 97.305 | 0.2772x | 0.5077x | 0.9993x | no |
| 3 | 1 | 1x5120x10240 + 1x5120x6144 | 85.235 | 0.2734x | 0.5844x | 1.0180x | yes |
| 3 | 2 | 2x5120x10240 + 2x5120x6144 | 88.096 | 0.2739x | 0.5587x | 1.0109x | yes |
| 3 | 4 | 4x5120x10240 + 4x5120x6144 | 88.602 | 0.2744x | 0.5544x | 1.0106x | yes |
| 3 | 8 | 8x5120x10240 + 8x5120x6144 | 88.726 | 0.2615x | 0.5531x | 1.0134x | yes |
| 3 | 16 | 16x5120x10240 + 16x5120x6144 | 92.210 | 0.2926x | 0.5357x | 1.0119x | yes |

The affected W2/W3 geometric speedup is **1.0123x**, with 10 of 10 wins.
The all-rate geometric comparison including the unchanged W2.5 control is
1.0077x. Raw results are in
`artifacts/a41_phase83_h100/qwen38_27b_linear_decode_prefetch.json`.
