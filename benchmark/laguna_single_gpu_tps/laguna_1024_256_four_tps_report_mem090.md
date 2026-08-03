# Laguna-S-2.1 Single-GPU TPS Report

Benchmark code and instructions: [README](README.md).

## Models

| Short Name | Model Path |
|---|---|
| GPTQ-4G64 | `/monster/data/model/Laguna-S-2.1-GPTQ-4G64` |
| CovMix | `/monster/data/model/Laguna-S-2.1-GPTQ-W4G64-CovMix` |
| CovMix-Embed/LMHead-W8G128 | `/monster/data/model/Laguna-S-2.1-GPTQ-W4G64-CovMix_embed_lmhead_w8g128` |

All three checkpoints use a symmetric W4G64 GPTQ model body with `desc_act=false` and FP16 activations. CovMix
uses dense BF16 embedding and LM-head weights loaded in FP16. CovMix-Embed/LMHead-W8G128 stores both endpoints as
symmetric W8G128 GPTQ; its embedding uses a benchmark-local TP=1 Triton packed-W8 dequant-on-lookup path without
dense materialization, while its LM head uses GPTQ-Marlin.

## Benchmark Contract

- Hardware: one NVIDIA PG506-230, 98,304 MiB, Compute Capability 8.0, 124 SMs.
- GPU allocation: physical GPU 0 was used for all reported runs; physical GPUs 4-7 were not used.
- Workload: 1024 input tokens and exactly 256 generated tokens per request, temperature 0, `ignore_eos=true`.
- Method: one warmup followed by three timed runs per successful Batch. TPS values are mean ± sample standard
  deviation. Time columns are the mean duration of one timed run, not the sum of three runs.
- Isolation: vLLM and SGLang ran in separate, non-overlapping processes with prefix caching and CUDA Graphs
  disabled. SGLang overlap scheduling was also disabled.
- CovMix runs: vLLM `gpu_memory_utilization=0.90`; SGLang `mem_fraction_static=0.90` and `stream_interval=1`.
- Historical GPTQ-4G64 runs: vLLM used `gpu_memory_utilization=0.97`; SGLang used
  `mem_fraction_static=0.90` but did not force or record `stream_interval=1`. Its Prefill, Output, and Total TPS
  include legacy frontend delivery delay and are not directly comparable with the corrected CovMix SGLang rows.

Let `B` be the submitted Batch, `Tp` the maximum per-request time to first token, `Td` the maximum per-request
first-to-last-token Decode duration, and `Te` the full workload wall time:

```text
Prefill TPS = B × 1024 / Tp
Decode TPS  = B × 255  / Td
Output TPS  = B × 256  / Te
Total TPS   = B × 1280 / Te
```

The first generated token belongs to Prefill, so Decode counts 255 tokens per request. Output TPS and Total TPS
share the same end-to-end time. `Total TPS` is not the sum of Prefill TPS and Decode TPS; for this fixed workload,
`Total TPS = Output TPS × 5`.

## TPS and Per-Run Time

### vLLM

| Model | Batch | Prefill Time (s/run) | Prefill TPS (tokens/s) | Decode Time (s/run) | Decode TPS (tokens/s) | E2E Time (s/run) | Output TPS (tokens/s) | Total TPS (tokens/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPTQ-4G64 | 1 | 0.168 | 6087.763 ± 33.331 | 21.148 | 12.058 ± 0.209 | 21.316 | 12.012 ± 0.206 | 60.061 ± 1.029 |
| GPTQ-4G64 | 2 | 0.315 | 6502.698 ± 21.368 | 21.414 | 23.817 ± 0.295 | 21.626 | 23.678 ± 0.291 | 118.388 ± 1.453 |
| GPTQ-4G64 | 4 | 0.525 | 7806.667 ± 365.224 | 22.181 | 45.986 ± 1.021 | 22.558 | 45.403 ± 0.783 | 227.013 ± 3.915 |
| GPTQ-4G64 | 8 | 0.952 | 8604.156 ± 265.170 | 22.202 | 91.884 ± 1.200 | 22.924 | 89.346 ± 1.042 | 446.732 ± 5.208 |
| GPTQ-4G64 | 16 | 1.814 | 9030.708 ± 45.730 | 23.375 | 174.542 ± 2.829 | 24.550 | 166.870 ± 2.615 | 834.348 ± 13.074 |
| GPTQ-4G64 | 32 | 3.596 | 9112.371 ± 71.664 | 24.110 | 338.447 ± 9.150 | 25.911 | 316.228 ± 5.542 | 1581.138 ± 27.711 |
| GPTQ-4G64 | 64 | 7.186 | 9120.562 ± 21.888 | 27.941 | 584.080 ± 10.363 | 29.394 | 557.497 ± 9.492 | 2787.484 ± 47.458 |
| GPTQ-4G64 | 128 | 14.570 | 8995.893 ± 8.109 | 35.610 | 916.607 ± 9.393 | 37.443 | 875.208 ± 8.851 | 4376.041 ± 44.256 |
| CovMix | 1 | 0.171 | 5988.234 ± 141.555 | 21.440 | 11.894 ± 0.169 | 21.611 | 11.847 ± 0.169 | 59.237 ± 0.844 |
| CovMix | 2 | 0.297 | 6893.240 ± 397.338 | 23.636 | 21.577 ± 2.892 | 23.899 | 21.695 ± 2.860 | 108.475 ± 14.300 |
| CovMix | 4 | 0.538 | 7608.885 ± 15.742 | 22.093 | 46.168 ± 0.552 | 22.411 | 45.697 ± 0.540 | 228.485 ± 2.702 |
| CovMix | 8 | 0.966 | 8477.064 ± 73.477 | 22.019 | 92.648 ± 0.369 | 22.622 | 90.531 ± 0.328 | 452.657 ± 1.641 |
| CovMix | 16 | 1.816 | 9023.938 ± 35.613 | 22.370 | 182.384 ± 0.334 | 23.600 | 173.560 ± 0.371 | 867.798 ± 1.856 |
| CovMix | 32 | 3.579 | 9155.005 ± 35.062 | 26.092 | 312.739 ± 36.488 | 27.498 | 300.295 ± 31.728 | 1501.476 ± 158.638 |
| CovMix | 64 | 7.182 | 9125.655 ± 34.037 | 28.559 | 571.454 ± 2.193 | 30.055 | 545.141 ± 1.058 | 2725.706 ± 5.290 |
| CovMix | 128 | 14.580 | 8989.652 ± 14.009 | 36.308 | 898.975 ± 3.250 | 38.153 | 858.870 ± 2.908 | 4294.350 ± 14.538 |
| CovMix-Embed/LMHead-W8G128 | 1 | 0.170 | 6010.105 ± 217.672 | 21.896 | 11.646 ± 0.160 | 22.067 | 11.603 ± 0.161 | 58.013 ± 0.804 |
| CovMix-Embed/LMHead-W8G128 | 2 | 0.319 | 6421.464 ± 42.272 | 22.471 | 22.696 ± 0.257 | 22.683 | 22.574 ± 0.254 | 112.869 ± 1.270 |
| CovMix-Embed/LMHead-W8G128 | 4 | 0.545 | 7515.576 ± 119.239 | 22.510 | 45.313 ± 0.235 | 22.852 | 44.812 ± 0.268 | 224.058 ± 1.340 |
| CovMix-Embed/LMHead-W8G128 | 8 | 0.964 | 8499.090 ± 92.357 | 22.800 | 89.475 ± 0.538 | 23.369 | 87.641 ± 0.610 | 438.206 ± 3.052 |
| CovMix-Embed/LMHead-W8G128 | 16 | 1.809 | 9056.770 ± 5.207 | 23.010 | 177.312 ± 3.874 | 24.176 | 169.476 ± 3.515 | 847.378 ± 17.576 |
| CovMix-Embed/LMHead-W8G128 | 32 | 3.581 | 9151.428 ± 13.462 | 24.849 | 328.379 ± 4.684 | 26.151 | 313.299 ± 4.303 | 1566.493 ± 21.517 |
| CovMix-Embed/LMHead-W8G128 | 64 | 7.200 | 9101.690 ± 55.088 | 28.483 | 572.978 ± 18.615 | 30.522 | 536.799 ± 3.290 | 2683.994 ± 16.450 |
| CovMix-Embed/LMHead-W8G128 | 128 | 14.633 | 8957.418 ± 75.974 | 38.502 | 847.743 ± 79.673 | 40.336 | 816.959 ± 73.053 | 4084.797 ± 365.265 |

### SGLang

| Model | Batch | Prefill Time (s/run) | Prefill TPS (tokens/s) | Decode Time (s/run) | Decode TPS (tokens/s) | E2E Time (s/run) | Output TPS (tokens/s) | Total TPS (tokens/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPTQ-4G64 | 1 | 4.413 | 232.052 ± 7.518 | 18.159 | 14.043 ± 0.378 | 22.573 | 11.346 ± 0.293 | 56.731 ± 1.466 |
| GPTQ-4G64 | 2 | 4.728 | 433.141 ± 2.807 | 18.583 | 27.445 ± 0.392 | 23.312 | 21.965 ± 0.233 | 109.825 ± 1.167 |
| GPTQ-4G64 | 4 | 4.982 | 822.160 ± 30.109 | 18.918 | 53.917 ± 1.116 | 23.901 | 42.860 ± 1.008 | 214.300 ± 5.038 |
| GPTQ-4G64 | 8 | 5.363 | 1527.477 ± 33.913 | 19.032 | 107.188 ± 0.526 | 24.396 | 83.952 ± 0.660 | 419.761 ± 3.302 |
| GPTQ-4G64 | 16 | 6.332 | 2587.474 ± 37.752 | 18.614 | 219.184 ± 1.883 | 24.948 | 164.195 ± 1.660 | 820.976 ± 8.302 |
| GPTQ-4G64 | 32 | 8.271 | 3961.829 ± 40.063 | 18.363 | 444.367 ± 6.197 | 26.636 | 307.572 ± 2.736 | 1537.861 ± 13.680 |
| GPTQ-4G64 | 64 | — | OOM | — | OOM | — | OOM | OOM |
| CovMix | 1 | 0.169 | 6075.622 ± 19.805 | 19.336 | 13.188 ± 0.625 | 19.505 | 13.144 ± 0.616 | 65.720 ± 3.079 |
| CovMix | 2 | 0.295 | 6938.724 ± 16.770 | 19.300 | 26.425 ± 0.126 | 19.596 | 26.128 ± 0.123 | 130.642 ± 0.616 |
| CovMix | 4 | 0.540 | 7587.701 ± 24.171 | 19.924 | 51.196 ± 0.216 | 20.465 | 50.037 ± 0.203 | 250.186 ± 1.013 |
| CovMix | 8 | 1.041 | 7866.660 ± 43.511 | 19.585 | 104.163 ± 0.196 | 20.627 | 99.288 ± 0.180 | 496.440 ± 0.898 |
| CovMix | 16 | 2.088 | 7846.172 ± 10.049 | 21.484 | 189.910 ± 22.489 | 23.573 | 175.194 ± 18.833 | 875.968 ± 94.167 |
| CovMix | 32 | 4.227 | 7751.882 ± 11.331 | 21.938 | 371.953 ± 48.778 | 26.167 | 315.779 ± 34.756 | 1578.897 ± 173.782 |
| CovMix | 64 | — | OOM | — | OOM | — | OOM | OOM |
| CovMix-Embed/LMHead-W8G128 | 1 | 0.167 | 6125.260 ± 17.597 | 18.408 | 13.853 ± 0.010 | 18.575 | 13.782 ± 0.010 | 68.908 ± 0.050 |
| CovMix-Embed/LMHead-W8G128 | 2 | 0.295 | 6934.163 ± 37.997 | 22.138 | 23.037 ± 4.411 | 22.434 | 23.414 ± 4.315 | 117.068 ± 21.574 |
| CovMix-Embed/LMHead-W8G128 | 4 | 0.539 | 7597.101 ± 20.709 | 20.361 | 50.096 ± 0.730 | 20.901 | 48.999 ± 0.692 | 244.994 ± 3.461 |
| CovMix-Embed/LMHead-W8G128 | 8 | 1.038 | 7890.165 ± 6.072 | 19.502 | 104.602 ± 0.450 | 20.542 | 99.700 ± 0.405 | 498.499 ± 2.023 |
| CovMix-Embed/LMHead-W8G128 | 16 | 2.085 | 7856.526 ± 9.680 | 19.780 | 206.272 ± 3.242 | 21.866 | 187.346 ± 2.647 | 936.730 ± 13.236 |
| CovMix-Embed/LMHead-W8G128 | 32 | 4.225 | 7755.278 ± 7.297 | 20.191 | 404.132 ± 11.263 | 24.418 | 335.607 ± 7.758 | 1678.036 ± 38.788 |
| CovMix-Embed/LMHead-W8G128 | 64 | — | OOM | — | OOM | — | OOM | OOM |

## Measured Benchmark Runtime

| Model | Framework | Successful Batches | Warmup Time (s) | Timed Time (s) | Recorded Workload Time (s) | Recorded Workload Time (min) |
|---|---|---:|---:|---:|---:|---:|
| GPTQ-4G64 | vLLM | 8 | 210.187 | 617.165 | 827.352 | 13.79 |
| CovMix | vLLM | 8 | 208.735 | 629.544 | 838.279 | 13.97 |
| CovMix-Embed/LMHead-W8G128 | vLLM | 8 | 216.360 | 636.466 | 852.825 | 14.21 |
| GPTQ-4G64 | SGLang | 6 | 148.225 | 437.292 | 585.517 | 9.76 |
| CovMix | SGLang | 6 | 130.993 | 389.799 | 520.792 | 8.68 |
| CovMix-Embed/LMHead-W8G128 | SGLang | 6 | 127.599 | 386.212 | 513.811 | 8.56 |

`Recorded Workload Time` is the sum of one warmup and three timed end-to-end runs for every successful Batch. It
excludes model loading, engine initialization, teardown, and the failed SGLang Batch 64 OOM attempt. It is therefore
the reproducible measured workload time, not the complete process lifetime.

## Scheduling and Validity Notes

- vLLM used its default `max_num_seqs=1024` and `max_num_batched_tokens=16384`. Batch 32 and above used chunked
  Prefill and scheduler waiting. Their Output and Total TPS remain valid end-to-end workload throughput because E2E
  time includes that waiting; Prefill and Decode TPS are per-request-envelope diagnostics rather than one unsplit
  resident phase.
- SGLang admitted Batch 64 as a single-resident workload. Every model reached a real CUDA OOM at Batch 64; the
  failed attempt did not produce a valid runtime or TPS row.
- All 42 successful rows were checked against their raw JSON results. Every successful request generated exactly
  256 output tokens, and no partial output entered an aggregate.
