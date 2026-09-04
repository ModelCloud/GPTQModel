# QVQ A41/R0 Phase 6 H100 split-K benchmark

The benchmark targets the physical NVIDIA H100 at PCI address
`00000000:44:00.0`; the H200 is hidden with `CUDA_VISIBLE_DEVICES=1`.  A strict
idle-device gate runs before Torch import.  Timings use warmed CUDA Graph
replays bracketed by CUDA events, excluding CPU/container launch gaps.

This first probe measures only the P32 inner decoder and matrix multiply for
Llama 3.2 1B down, (M=1,K=8192,N=2048,W3).  It is a scheduling experiment,
so full-module Marlin and Machete values are intentionally not mixed into this
table.  Production comparisons are added after runtime policy integration.

| M | K | N | Split | Blocks | Median (us) | Speedup vs split 1 | Repeatable | Max abs vs dense |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 8192 | 2048 | 1 | 32 | 72.806 | 1.000x | Yes | 0.00007010 |
| 1 | 8192 | 2048 | 2 | 64 | 38.898 | 1.872x | Yes | 0.00003386 |
| 1 | 8192 | 2048 | 4 | 128 | 22.122 | 3.291x | Yes | 0.00001621 |
| 1 | 8192 | 2048 | 8 | 256 | 17.647 | 4.126x | Yes | 0.00000763 |
| 1 | 8192 | 2048 | 16 | 512 | 16.506 | 4.411x | Yes | 0.00000429 |
| 1 | 8192 | 2048 | 32 | 1024 | 18.449 | 3.946x | Yes | 0.00000381 |

Split 16 wins this W3/M1 probe.  Split 32 regresses 11.8% from split 16,
showing that decoder parallelism has crossed the point where extra block and
reduction traffic dominate.  Every ordered split is bit-repeatable over ten
eager launches and the captured CUDA Graph result equals the eager reference.

The raw record is
`artifacts/a41_phase6_h100/llama_down_w3_m1_ordered_probe.json`.

## Full split-selection matrix

The formal sweep uses 20 warmups, 50 samples, and 20 captured replays per
sample.  Split 16 wins every measured rate and logical row count.

| W | M | K | N | Winning split | Split 1 (us) | Winner (us) | Speedup | Better than split 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2 | 1 | 8192 | 2048 | 16 | 72.606 | 15.982 | 4.543x | Yes |
| 2 | 2 | 8192 | 2048 | 16 | 69.731 | 15.994 | 4.360x | Yes |
| 2 | 4 | 8192 | 2048 | 16 | 69.672 | 15.960 | 4.365x | Yes |
| 2 | 8 | 8192 | 2048 | 16 | 69.720 | 15.914 | 4.381x | Yes |
| 2 | 16 | 8192 | 2048 | 16 | 69.677 | 15.889 | 4.385x | Yes |
| 2.5 | 1 | 8192 | 2048 | 16 | 72.882 | 16.115 | 4.523x | Yes |
| 2.5 | 2 | 8192 | 2048 | 16 | 70.462 | 16.198 | 4.350x | Yes |
| 2.5 | 4 | 8192 | 2048 | 16 | 70.410 | 16.121 | 4.368x | Yes |
| 2.5 | 8 | 8192 | 2048 | 16 | 70.513 | 16.266 | 4.335x | Yes |
| 2.5 | 16 | 8192 | 2048 | 16 | 70.535 | 16.273 | 4.335x | Yes |
| 3 | 1 | 8192 | 2048 | 16 | 72.418 | 16.264 | 4.453x | Yes |
| 3 | 2 | 8192 | 2048 | 16 | 70.536 | 16.370 | 4.309x | Yes |
| 3 | 4 | 8192 | 2048 | 16 | 70.596 | 16.203 | 4.357x | Yes |
| 3 | 8 | 8192 | 2048 | 16 | 70.620 | 16.346 | 4.320x | Yes |
| 3 | 16 | 8192 | 2048 | 16 | 70.518 | 16.222 | 4.347x | Yes |
| 3.5 | 1 | 8192 | 2048 | 16 | 72.774 | 16.356 | 4.449x | Yes |
| 3.5 | 2 | 8192 | 2048 | 16 | 71.022 | 16.375 | 4.337x | Yes |
| 3.5 | 4 | 8192 | 2048 | 16 | 70.995 | 16.332 | 4.347x | Yes |
| 3.5 | 8 | 8192 | 2048 | 16 | 71.005 | 16.280 | 4.361x | Yes |
| 3.5 | 16 | 8192 | 2048 | 16 | 71.084 | 16.478 | 4.314x | Yes |

All 120 rate/M/split cases are bit-repeatable, CUDA-Graph stable, and below
`8.8e-5` maximum absolute error versus the reconstructed dense P32 matrix.
The raw record is
`artifacts/a41_phase6_h100/llama_down_all_rates_ordered_sweep.json`.

## Production complete-MLP result

The production benchmark includes grouped gate/up, the model's original SiLU,
the Phase-5 exact MLP fusions, the promoted split-16 down projection, and
child-local recovery.  Marlin and Machete are W4 figurative baselines; the QVQ
rows retain their displayed W2 through W3.5 storage rates.

| W | M | QVQ (us) | vs Marlin W4 | vs Machete W4 | vs Phase 5 | Better than Phase 5 |
|---:|---:|---:|---:|---:|---:|:---:|
| 2 | 1 | 96.278 | 0.308x | 0.536x | 1.542x | Yes |
| 2 | 2 | 96.422 | 0.331x | 0.534x | 1.542x | Yes |
| 2 | 4 | 97.093 | 0.330x | 0.527x | 1.537x | Yes |
| 2 | 8 | 97.918 | 0.306x | 0.522x | 1.533x | Yes |
| 2 | 16 | 92.466 | 0.355x | 0.552x | 1.565x | Yes |
| 2.5 | 1 | 97.087 | 0.306x | 0.531x | 1.541x | Yes |
| 2.5 | 2 | 97.588 | 0.327x | 0.528x | 1.539x | Yes |
| 2.5 | 4 | 98.054 | 0.327x | 0.522x | 1.537x | Yes |
| 2.5 | 8 | 98.893 | 0.303x | 0.517x | 1.532x | Yes |
| 2.5 | 16 | 93.441 | 0.351x | 0.546x | 1.564x | Yes |
| 3 | 1 | 98.034 | 0.303x | 0.526x | 1.531x | Yes |
| 3 | 2 | 98.345 | 0.324x | 0.524x | 1.527x | Yes |
| 3 | 4 | 98.686 | 0.325x | 0.518x | 1.529x | Yes |
| 3 | 8 | 99.170 | 0.302x | 0.515x | 1.529x | Yes |
| 3 | 16 | 93.626 | 0.350x | 0.545x | 1.561x | Yes |
| 3.5 | 1 | 97.164 | 0.305x | 0.531x | 1.544x | Yes |
| 3.5 | 2 | 97.465 | 0.327x | 0.529x | 1.542x | Yes |
| 3.5 | 4 | 97.682 | 0.328x | 0.524x | 1.548x | Yes |
| 3.5 | 8 | 98.658 | 0.304x | 0.518x | 1.537x | Yes |
| 3.5 | 16 | 93.379 | 0.351x | 0.547x | 1.568x | Yes |

The geometric Phase-6 speedup over Phase 5 is 1.542x, or 35.16% lower
latency.  Complete-MLP geometric rates are 0.323x Marlin and 0.529x Machete.
The raw record is
`artifacts/a41_phase6_h100/production_mlp_vs_baselines.json`.

## Clean pre-PR main comparison

A detached worktree at exact `origin/main` commit `b75ed8ad` was benchmarked
with the same payload construction, H100, inputs, warmups, sample counts, CUDA
Graphs, and CUDA-event timing.  The geometric complete-MLP speedup is 11.945x.
That number is the cumulative PR #98 gain, not the isolated split-K gain.
Origin main ranges from 344.431 us for W2/M1 to about 1.5 ms for most W2.5,
W3, and W3.5 cases; Phase 6 ranges from 92.466 to 99.170 us.

The clean baseline is
`artifacts/a41_phase6_h100/origin_main_plain_mlp.json`.

## Current-tip MLP refresh after Phase 7

The complete MLP matrix was rerun at PR tip `2fd89223` after the Phase 7
query/key/value integration.  Phase 7 does not change the MLP kernel; this is a
cross-run stability check.  Every row executes two gate/up matrices
`(M,2048,8192)` followed by one down matrix `(M,8192,2048)` and includes SiLU,
all QVQ transforms, recovery, and runtime coordination.

`vs last` compares with the committed Phase 6 production MLP artifact.  The
strict boolean is `Yes` only when the new median is lower; any increase is
reported as `No`, even when it is small enough to be ordinary run-to-run
variation.

| W | M | QVQ (us) | vs last | Better | Marlin W4 (us) | vs Marlin | Machete W4 (us) | vs Machete | Effective TFLOP/s |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 1 | 96.630 | 0.9964x | No | 29.588 | 0.306x | 51.626 | 0.534x | 1.042 |
| 2 | 2 | 96.900 | 0.9951x | No | 31.871 | 0.329x | 51.473 | 0.531x | 2.078 |
| 2 | 4 | 97.434 | 0.9965x | No | 31.819 | 0.327x | 51.261 | 0.526x | 4.133 |
| 2 | 8 | 98.314 | 0.9960x | No | 29.706 | 0.302x | 51.252 | 0.521x | 8.191 |
| 2 | 16 | 92.742 | 0.9970x | No | 33.020 | 0.356x | 51.220 | 0.552x | 17.367 |
| 2.5 | 1 | 97.587 | 0.9949x | No | 29.588 | 0.303x | 51.626 | 0.529x | 1.032 |
| 2.5 | 2 | 98.026 | 0.9955x | No | 31.871 | 0.325x | 51.473 | 0.525x | 2.054 |
| 2.5 | 4 | 98.242 | 0.9981x | No | 31.819 | 0.324x | 51.261 | 0.522x | 4.099 |
| 2.5 | 8 | 99.149 | 0.9974x | No | 29.706 | 0.300x | 51.252 | 0.517x | 8.122 |
| 2.5 | 16 | 93.855 | 0.9956x | No | 33.020 | 0.352x | 51.220 | 0.546x | 17.161 |
| 3 | 1 | 97.962 | 1.0007x | Yes | 29.588 | 0.302x | 51.626 | 0.527x | 1.028 |
| 3 | 2 | 98.192 | 1.0016x | Yes | 31.871 | 0.325x | 51.473 | 0.524x | 2.050 |
| 3 | 4 | 98.669 | 1.0002x | Yes | 31.819 | 0.322x | 51.261 | 0.520x | 4.081 |
| 3 | 8 | 99.358 | 0.9981x | No | 29.706 | 0.299x | 51.252 | 0.516x | 8.105 |
| 3 | 16 | 94.005 | 0.9960x | No | 33.020 | 0.351x | 51.220 | 0.545x | 17.133 |
| 3.5 | 1 | 97.367 | 0.9979x | No | 29.588 | 0.304x | 51.626 | 0.530x | 1.034 |
| 3.5 | 2 | 97.826 | 0.9963x | No | 31.871 | 0.326x | 51.473 | 0.526x | 2.058 |
| 3.5 | 4 | 98.314 | 0.9936x | No | 31.819 | 0.324x | 51.261 | 0.521x | 4.096 |
| 3.5 | 8 | 98.946 | 0.9971x | No | 29.706 | 0.300x | 51.252 | 0.518x | 8.139 |
| 3.5 | 16 | 93.872 | 0.9948x | No | 33.020 | 0.352x | 51.220 | 0.546x | 17.158 |

The current-tip geometric means are 1.458x versus ordinary per-child QVQ,
0.321x versus Marlin W4, and 0.529x versus Machete W4.  Versus the last MLP
run the ratio is 0.9969x, corresponding to a 0.31% latency regression.  Three
cells improve and seventeen regress under the strict median comparison.  The
cumulative geometric speedup versus the clean pre-PR `origin/main` artifact
remains 11.908x.

The refreshed raw record is
`artifacts/a41_phase7_h100/production_mlp_refresh_vs_baselines.json`.
