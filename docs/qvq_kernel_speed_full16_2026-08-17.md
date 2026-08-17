# QVQ full-model kernel-speed sweep (2026-08-17)

## Purpose

Measure the latest QVQ quantization and inference changes at full Llama 3.2 1B scale, including every decoder
layer and every Q/K/V/O, gate/up, and down linear. The comparison covers canonical V2, V2B2-P32, and
V2B2-P32 with 512-row YAQA.

The tested source revision was `26679f27b243d1ee48e50583ebac1278a2e19512` (`accelerate underfilled QVQ
segmented recurrence`).

## Matched configuration

| Setting | Value |
|---|---|
| Model | `/private/monster/data/model/Llama-3.2-1B-Instruct` |
| Layers/modules | 16 layers; all 112 Q/K/V/O, gate/up, and down projections |
| Embedding / LM head | Dense |
| Hessian mode | Staged all-linear replay |
| Calibration | 512 full rows, offset 0, batch 1, no concatenation or length limit |
| Evaluation | 512 disjoint full rows, offset 512, batch 1 |
| YAQA Sketch-B | 512 further-disjoint rows, offset 1024, batch 8, seed 0 |
| MLP acceptance | 8 rows, offset 1536; KL and Top-N regression limits 5% |
| Main seed | 18240 |
| Dataset SHA-256 | `26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef` |
| Dataset | `/private/monster/data/model/dataset/nm-calibration/llm.parquet` |
| Factor cache | 15.3 GB local cache; metadata and representative tensor hashes validated |
| GPUs | 7x NVIDIA PG506-230 and 1x PG506-232, `sm_80`, 96 GB |
| Driver | 610.43.02 |
| PyTorch | 2.13.0+cu130 |
| CUDA / NVCC | CUDA 13.0; `cuda_13.0.r13.0/compiler.36424714_0` |

The six rates were distributed across GPUs 0-7. Each GPU ran one sweep process at a time. All 18 final result
artifacts completed successfully.

## Full results

| Rate | Arm | Seconds | Effective BPW | Rel L2 | Local KL | Live KL | Layer KL | Final KL | Top-1 | Top-5 | Top-10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | V2 | 397.8 | 1.0000 | 0.438016 | 0.087400 | 0.656277 | 0.629403 | 1.203016 | 58.83% | 53.79% | 53.13% |
| W1 | V2B2-P32 | 1516.8 | 1.0312 | 0.428284 | 0.080903 | 0.755357 | 0.629559 | 1.088000 | 60.98% | 55.15% | 54.33% |
| W1 | V2B2-P32 + YAQA | 2793.8 | 1.0312 | 0.529797 | 0.633308 | 0.935487 | 0.402686 | 0.756969 | 68.90% | 61.73% | 60.65% |
| W1.5 | V2 | 389.6 | 1.5000 | 0.317952 | 0.025754 | 0.347390 | 0.266919 | 0.386202 | 77.37% | 69.16% | 68.49% |
| W1.5 | V2B2-P32 | 1215.4 | 1.5312 | 0.310337 | 0.024768 | 0.313523 | 0.246109 | 0.355999 | 78.45% | 69.85% | 69.10% |
| W1.5 | V2B2-P32 + YAQA | 2226.0 | 1.5312 | 0.397504 | 0.135523 | 0.294137 | 0.141491 | 0.236714 | 82.98% | 74.78% | 74.00% |
| W2 | V2 | 376.0 | 2.0000 | 0.228039 | 0.010518 | 0.156065 | 0.108333 | 0.160293 | 85.10% | 78.29% | 77.76% |
| W2 | V2B2-P32 | 1156.8 | 2.0312 | 0.222241 | 0.009691 | 0.141203 | 0.099418 | 0.153436 | 86.38% | 79.00% | 78.50% |
| W2 | V2B2-P32 + YAQA | 2095.0 | 2.0312 | 0.283073 | 0.043864 | 0.109502 | 0.050926 | 0.085294 | 89.49% | 83.11% | 82.68% |
| W2.5 | V2 | 381.2 | 2.5000 | 0.163174 | 0.004561 | 0.072903 | 0.048801 | 0.078198 | 89.65% | 83.97% | 83.88% |
| W2.5 | V2B2-P32 | 1159.0 | 2.5312 | 0.156940 | 0.004113 | 0.066427 | 0.043957 | 0.070613 | 90.22% | 84.80% | 84.44% |
| W2.5 | V2B2-P32 + YAQA | 2045.0 | 2.5312 | 0.208448 | 0.018757 | 0.053403 | 0.025040 | 0.039548 | 93.01% | 87.74% | 87.48% |
| W3 | V2 | 352.9 | 3.0000 | 0.114047 | 0.002178 | 0.036317 | 0.022919 | 0.038752 | 92.75% | 88.26% | 88.10% |
| W3 | V2B2-P32 | 1177.8 | 3.0312 | 0.111197 | 0.002063 | 0.034181 | 0.020781 | 0.035386 | 93.00% | 88.77% | 88.74% |
| W3 | V2B2-P32 + YAQA | 2106.9 | 3.0312 | 0.135101 | 0.009484 | 0.020622 | 0.008021 | 0.012495 | 95.76% | 92.53% | 92.38% |
| W3.5 | V2 | 371.4 | 3.5000 | 0.080305 | 0.001101 | 0.017111 | 0.010537 | 0.018704 | 94.54% | 91.51% | 91.40% |
| W3.5 | V2B2-P32 | 1241.6 | 3.5312 | 0.079877 | 0.001015 | 0.016817 | 0.009989 | 0.017333 | 94.77% | 91.75% | 91.61% |
| W3.5 | V2B2-P32 + YAQA | 2187.2 | 3.5312 | 0.098740 | 0.004789 | 0.010901 | 0.004278 | 0.006578 | 96.76% | 94.42% | 94.26% |

## Historical timing A/B

Speedup is historical seconds divided by current seconds. The historical base V2/B2 reports used MLP
acceptance rows starting at 1024, whereas this matched three-arm sweep uses offset 1536 so that acceptance is
disjoint from YAQA. Total-time comparisons are useful performance indicators, but they are not strict
same-artifact quality comparisons.

| Rate | Arm | Old total (s) | New total (s) | Total speedup | Old eval (s) | New eval (s) | Eval speedup | Dominant kernel speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | V2 | 692.6 | 397.8 | 1.74x | 59.2 | 48.2 | 1.23x | 1.09x |
| W1 | V2B2-P32 | 1633.8 | 1516.8 | 1.08x | 58.6 | 53.5 | 1.10x | 0.84x |
| W1 | V2B2-P32 + YAQA | 2176.7 | 2793.8 | 0.78x | 43.2 | 41.8 | 1.03x | 1.07x |
| W1.5 | V2 | 710.3 | 389.6 | 1.82x | 62.5 | 49.5 | 1.26x | 1.17x |
| W1.5 | V2B2-P32 | 1723.7 | 1215.4 | 1.42x | 62.6 | 48.8 | 1.28x | 1.16x |
| W1.5 | V2B2-P32 + YAQA | 2143.1 | 2226.0 | 0.96x | 42.5 | 50.5 | 0.84x | 1.00x |
| W2 | V2 | 647.0 | 376.0 | 1.72x | 57.8 | 52.3 | 1.11x | 1.03x |
| W2 | V2B2-P32 | 1438.8 | 1156.8 | 1.24x | 57.9 | 43.6 | 1.33x | 0.92x |
| W2 | V2B2-P32 + YAQA | 2128.8 | 2095.0 | 1.02x | 43.6 | 48.0 | 0.91x | 1.17x |
| W2.5 | V2 | 663.8 | 381.2 | 1.74x | 59.1 | 50.4 | 1.17x | 1.06x |
| W2.5 | V2B2-P32 | 1460.7 | 1159.0 | 1.26x | 59.2 | 45.5 | 1.30x | 0.98x |
| W2.5 | V2B2-P32 + YAQA | 2091.2 | 2045.0 | 1.02x | 41.5 | 41.7 | 1.00x | 0.87x |
| W3 | V2 | 684.7 | 352.9 | 1.94x | 58.9 | 41.4 | 1.42x | 1.30x |
| W3 | V2B2-P32 | 1478.1 | 1177.8 | 1.25x | 58.3 | 43.9 | 1.33x | 0.94x |
| W3 | V2B2-P32 + YAQA | 2138.8 | 2106.9 | 1.02x | 41.9 | 47.8 | 0.88x | 1.39x |
| W3.5 | V2 | 700.9 | 371.4 | 1.89x | 59.5 | 41.6 | 1.43x | 1.06x |
| W3.5 | V2B2-P32 | 1689.1 | 1241.6 | 1.36x | 57.3 | 42.1 | 1.36x | 1.23x |
| W3.5 | V2B2-P32 + YAQA | 2162.9 | 2187.2 | 0.99x | 41.8 | 45.5 | 0.92x | 1.43x |

Dominant phases are `block_ldl_viterbi` for V2, `block_ldl_v2_banked` for B2, and
`yaqa_segmented_viterbi` for B2+YAQA. Geometric-mean speedups across W1-W3.5 were:

| Arm | Total | Quantization stages | Evaluation | Dominant kernel |
|---|---:|---:|---:|---:|
| V2 | 1.81x | 1.07x | 1.26x | 1.11x |
| V2B2-P32 | 1.26x | 1.06x | 1.28x | 1.00x |
| V2B2-P32 + YAQA | 0.96x | 0.95x | 0.93x | 1.14x |

## Interpretation

- The latest cooperative segmented recurrence remains a microkernel win for underfilled calls, but a full
  all-linear model is dominated by large MLP projections. The B2 dominant-kernel geometric mean is therefore
  effectively flat, with rate-dependent results from 0.84x to 1.23x under an eight-GPU concurrent sweep.
- V2 total runtime improved substantially, mostly from harness, acceptance, and evaluation changes. Its native
  recurrence improved by 1.11x geometrically.
- YAQA's segmented native phase improved by 1.14x geometrically, including 1.39x at W3 and 1.43x at W3.5.
  This phase is too small a fraction of full YAQA runtime to move total time materially.
- W1 YAQA is a clear concurrent-run outlier. Its qkv stage improved 1.18x, but gate/up and down slowed to 0.74x
  each while peak allocation approached the 96 GB device limit. This is host/VRAM/staging pressure, not a
  regression in the measured segmented kernel phase.
- The sweep validates end-to-end quality and finiteness, not raw kernel numerical error. Historical quality
  deltas include changed acceptance rows and intervening algorithm changes; they must not be interpreted as a
  direct CUDA accuracy A/B. The focused kernel tests attached to `26679f27` remain the raw exactness gate.

## Artifacts

All 18 JSON files, logs, and status files are under:

`/private/monster/data/model/qvq_kernel_speed_full16_26679f27_20260817`

