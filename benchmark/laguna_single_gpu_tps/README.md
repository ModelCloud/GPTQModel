# Laguna Single-GPU TPS Benchmark

This directory contains the benchmark harness and measured 1024-input/256-output results for the Laguna-S-2.1
GPTQ checkpoints covered by the reports below.

## Files

- `laguna_prefill_decode_tps.py`: the Prefill, Decode, Output, and Total TPS benchmark used for the report.
- `laguna_benchmark_common.py`: shared workload, validation, GPU preflight, metadata, and reporting utilities.
- `laguna_gptq_embedding.py`: TP=1 packed-W8G128 embedding lookup used by the endpoint-quantized checkpoint.
- `laguna_vllm_runtime_model.py`: benchmark-local vLLM Laguna/W8 embedding adapter.
- `laguna_sglang_runtime_models/`: benchmark-local SGLang Laguna/W8 embedding adapter.
- `laguna_covmix_single_gpu_tps_report_20260803.md`: current CovMix and W8-endpoint results plus historical data.
- `laguna_1024_256_four_tps_report_mem090.md`: historical GPTQ-4G64 report.

## Run the Benchmark

Run vLLM and SGLang in separate processes, never concurrently. Each process requires one exclusively leased GPU;
the benchmark performs three idle-GPU checks before loading the engine.

The current benchmark default is `gpu_memory_utilization=0.90` for vLLM and `mem_fraction_static=0.90` for
SGLang. The dated CovMix report uses 0.90 for both new frameworks/models. The older GPTQ-4G64 report is a
historical artifact: its vLLM rows were measured at 0.97, while its SGLang rows were measured at 0.90.

Run vLLM from its Python environment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<single-gpu-uuid> \
LAGUNA_BENCH_BATCH_SIZES=1,2,4,8,16,32,64,128 \
LAGUNA_BENCH_INPUT_LEN=1024 \
LAGUNA_BENCH_OUTPUT_LEN=256 \
LAGUNA_BENCH_CONTEXT_MARGIN=8 \
LAGUNA_BENCH_REPEATS=3 \
LAGUNA_PHASE_RUN_LABEL=isl1024_osl256_four_tps_default_vllm \
LAGUNA_VLLM_GPU_MEMORY_UTILIZATION=0.90 \
hub/vllm/.venv/bin/python benchmark/laguna_single_gpu_tps/laguna_prefill_decode_tps.py vllm
```

After the vLLM process exits and the GPU is idle again, run SGLang from its Python environment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<single-gpu-uuid> \
LAGUNA_BENCH_BATCH_SIZES=1,2,4,8,16,32,64 \
LAGUNA_BENCH_INPUT_LEN=1024 \
LAGUNA_BENCH_OUTPUT_LEN=256 \
LAGUNA_BENCH_CONTEXT_MARGIN=8 \
LAGUNA_BENCH_REPEATS=3 \
LAGUNA_PHASE_RUN_LABEL=isl1024_osl256_four_tps_sglang_mem090 \
LAGUNA_SGLANG_MEM_FRACTION_STATIC=0.90 \
<sglang-python> benchmark/laguna_single_gpu_tps/laguna_prefill_decode_tps.py sglang
```

For SGLang, `max_running_requests`, `prefill_max_requests`, `max_prefill_tokens`, and `chunked_prefill_size` are
derived from the largest requested batch and the input length. There is no separate
`LAGUNA_PHASE_MAX_RESIDENT_REQUESTS` setting. These derived limits intentionally let the largest requested
SGLang batch enter Prefill without scheduler-side request splitting; this preserves the single-resident workload
used by the report, but it can increase peak activation memory. Remove those four engine arguments when measuring
SGLang's default scheduler-managed chunking instead.

The runner forces `SGLANG_FORCE_STREAM_INTERVAL=1` before importing SGLang so that first-token delivery does not
add a multi-token frontend buffering delay to phase timestamps.

For `/monster/data/model/Laguna-S-2.1-GPTQ-W4G64-CovMix_embed_lmhead_w8g128`, the runner detects packed W8G128
embedding and LM-head tensors from the dynamic quantization metadata and checkpoint index. The benchmark-local
embedding path keeps the tensor packed and dequantizes only requested rows with Triton; it supports TP=1 only. The
LM head remains on the framework GPTQ-Marlin path. Other checkpoints retain their normal dense embedding and LM
head behavior.

Set `LAGUNA_BENCH_ROOT`, `LAGUNA_BENCH_MODEL`, `LAGUNA_BENCH_SGLANG_REPO`,
`LAGUNA_BENCH_VLLM_REPO`, or `LAGUNA_BENCH_OUTPUT_DIR` to override the repository, model, framework checkout, or
JSON output locations. Defaults match this repository layout and the checkpoint used for the published report.

The script uses one warmup and three timed repetitions per batch size. It requires each request to produce exactly
256 output tokens and writes the complete runtime, GPU, engine configuration, per-repeat timing, and aggregate
statistics to JSON.
