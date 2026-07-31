# Laguna-S-2.1 Single-GPU TPS Report

Benchmark code and instructions: [README](README.md).

## Metric Definitions and Relationship

This benchmark uses fixed lengths of `Input Tokens = 1024` and `Output Tokens = 256`. Let \(B\) be the submitted batch size, and let \(T_p\), \(T_d\), and \(T_e\) be the Prefill, Decode, and end-to-end durations:

```text
Prefill TPS = B × 1024 / Tp
Decode TPS  = B × 255  / Td
Output TPS  = B × 256  / Te
Total TPS   = B × 1280 / Te
```

The first generated token is attributed to Prefill, so Decode accounts for 255 tokens per request.

`Total TPS` is not `Prefill TPS + Decode TPS`. Prefill TPS and Decode TPS use their respective phase durations as denominators. Total TPS uses the full end-to-end duration and counts both input and output tokens in the numerator. It therefore includes the elapsed time of both phases, but it does not add their rates together.

Because the input/output length ratio is fixed in this benchmark:

```text
Total TPS = Output TPS × (1024 + 256) / 256 = Output TPS × 5
```

## Test Configuration

- Model: `/monster/data/model/Laguna-S-2.1-GPTQ-4G64`
- Quantization: symmetric 4-bit GPTQ, group size 64, `desc_act=false`; FP16 activations
- GPU: One NVIDIA PG506-230, 98,304 MiB reported by `nvidia-smi`, 95.17 GiB visible to PyTorch,
  Compute Capability 8.0, 124 SMs
- NVIDIA driver: 610.43.02
- Workload: 1024 input tokens per request and exactly 256 generated output tokens per request
- Maximum model length: 1,288 tokens, including an 8-token context margin
- Method: One warmup followed by three timed runs for each batch size; values are mean ± sample standard deviation
- Prefix caching and CUDA Graphs were disabled; vLLM and SGLang ran separately
- vLLM: source revision `7aea73d83d6064449ff8147899de161e6eb68a20`, Python 3.12.13,
  PyTorch 2.13.0+cu130, `gpu_memory_utilization=0.97`
- SGLang: source revision `7c248dde7fe1f3b5100966f8143f97a9932c22a4`, Python 3.11.14,
  PyTorch 2.11.0+cu130, `sglang-kernel=0.4.5`, `mem_fraction_static=0.90`
- FlashInfer: 0.6.15.post1 in both framework environments
- Build: existing framework environments and kernels were used; no framework or kernel rebuild occurred during the
  measurement

## vLLM

| Batch | Prefill TPS (tokens/s) | Decode TPS (tokens/s) | Output TPS (tokens/s) | Total TPS (tokens/s) |
|---:|---:|---:|---:|---:|
| 1 | 6087.763 ± 33.331 | 12.058 ± 0.209 | 12.012 ± 0.206 | 60.061 ± 1.029 |
| 2 | 6502.698 ± 21.368 | 23.817 ± 0.295 | 23.678 ± 0.291 | 118.388 ± 1.453 |
| 4 | 7806.667 ± 365.224 | 45.986 ± 1.021 | 45.403 ± 0.783 | 227.013 ± 3.915 |
| 8 | 8604.156 ± 265.170 | 91.884 ± 1.200 | 89.346 ± 1.042 | 446.732 ± 5.208 |
| 16 | 9030.708 ± 45.730 | 174.542 ± 2.829 | 166.870 ± 2.615 | 834.348 ± 13.074 |
| 32 | 9112.371 ± 71.664 | 338.447 ± 9.150 | 316.228 ± 5.542 | 1581.138 ± 27.711 |
| 64 | 9120.562 ± 21.888 | 584.080 ± 10.363 | 557.497 ± 9.492 | 2787.484 ± 47.458 |
| 128 | 8995.893 ± 8.109 | 916.607 ± 9.393 | 875.208 ± 8.851 | 4376.041 ± 44.256 |

> **Note (vLLM):** Neither `max_num_seqs` nor `max_num_batched_tokens` was passed explicitly. vLLM resolved them to its defaults of 1024 sequences and 16,384 scheduled tokens. The sequence-count limit therefore did not constrain this benchmark.
>
> Starting at Batch 32, the aggregate prompt contains more than 16,384 tokens: 32,768 tokens at Batch 32, 65,536 at Batch 64, and 131,072 at Batch 128. vLLM consequently performs chunked Prefill across multiple scheduler iterations. Some requests may remain in the waiting queue temporarily while the scheduler admits token chunks that fit the current token and KV-cache budgets. Batch 128 was still submitted as 128 requests, and the scheduler logs eventually reached `Running: 128, Waiting: 0`.
>
> When additional KV-cache blocks cannot be allocated safely, vLLM applies scheduler backpressure by leaving requests waiting or by preempting and recomputing requests. It does not blindly over-allocate GPU memory, which is why Batch 32–128 completed without a fatal CUDA OOM. Every request produced exactly 256 output tokens, and every table row contains one warmup plus three valid timed runs.
>
> These Batch 32–128 results are valid for a submitted request batch under vLLM's default scheduler. All requests were submitted in one engine call, and the measured wall time includes scheduler waiting and chunked execution; no waiting time was discarded. They should not be interpreted as measurements of one unsplit, single-resident Prefill batch. Prefill TPS covers the scheduler-managed chunked Prefill makespan, Decode TPS covers the scheduler-managed Decode makespan, and Output TPS and Total TPS are end-to-end workload throughput metrics. Comparisons with SGLang at these batch sizes must therefore account for the fact that SGLang was measuring a single-resident batch while vLLM was allowed to schedule the submitted batch internally.

## SGLang

| Batch | Prefill TPS (tokens/s) | Decode TPS (tokens/s) | Output TPS (tokens/s) | Total TPS (tokens/s) |
|---:|---:|---:|---:|---:|
| 1 | 232.052 ± 7.518 | 14.043 ± 0.378 | 11.346 ± 0.293 | 56.731 ± 1.466 |
| 2 | 433.141 ± 2.807 | 27.445 ± 0.392 | 21.965 ± 0.233 | 109.825 ± 1.167 |
| 4 | 822.160 ± 30.109 | 53.917 ± 1.116 | 42.860 ± 1.008 | 214.300 ± 5.038 |
| 8 | 1527.477 ± 33.913 | 107.188 ± 0.526 | 83.952 ± 0.660 | 419.761 ± 3.302 |
| 16 | 2587.474 ± 37.752 | 219.184 ± 1.883 | 164.195 ± 1.660 | 820.976 ± 8.302 |
| 32 | 3961.829 ± 40.063 | 444.367 ± 6.197 | 307.572 ± 2.736 | 1537.861 ± 13.680 |
| 64 | OOM | OOM | OOM | OOM |

Batch 64 failed with a CUDA OOM during warmup Prefill in both the continuous sweep and a fresh-process
confirmation run. The failed allocations requested an additional 3.75 GiB with only 1.94 GiB and 2.27 GiB free,
respectively.

Compared with `mem_fraction_static=0.97`, using 0.90 reduced the SWA/KV pool from approximately 30.74 GB to
24.12 GB and increased post-pool free memory from approximately 2.89 GB to 9.50 GB. The largest successful
single-resident SGLang batch consequently increased from Batch 16 to Batch 32, but Batch 64 still OOMed.

## Validation

- vLLM completed eight batch sizes and 24 timed samples; every request produced exactly 256 output tokens.
- SGLang 0.90 completed six batch sizes and 18 timed samples; every successful request produced exactly 256 output
  tokens.
- No unrelated GPU process appeared during the exclusivity checks for any successful SGLang batch.
- Both SGLang Batch 64 attempts ended in a real CUDA OOM.
- The target GPU returned to 0 MiB used and 0% utilization after the tests.
