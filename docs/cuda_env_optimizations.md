# CUDA Environment Optimizations for GPT-QModel

This document collects environment settings that improve PyTorch CUDA memory behavior for GPT-QModel workloads, especially quantized loading and batched evaluation.

## TL;DR recommended allocator config

```bash
export PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.5'
```

`gptqmodel/models/auto.py` sets this automatically when the variable is unset. Override it before importing `gptqmodel` if you want to experiment.

## Problem: `CUDACachingAllocator` expandable-segments warnings

During batched GGUF evaluation (e.g. `tests/models/test_llama3_2_gguf.py`) you may see lines like:

```text
CUDACachingAllocator.cpp:508: expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 6553600, total: ...)
```

These are **retry warnings, not fatal errors**. The allocator is attempting to map a 20 MiB expandable segment page when only ~6–13 MiB of GPU memory is free, and is falling back to a smaller or alternative allocation. The test still passes, but the retries add overhead and log noise.

## What does not work

- **Removing `expandable_segments:True`** makes the problem much worse. The same memory pressure produces generic `cudaMalloc` OOM retry warnings (776 observed vs ~52 expandable-segment warnings) because the allocator can no longer grow segments by pages.
- `expandable_segments` should stay enabled.

## Tuning `max_split_size_mb` and `garbage_collection_threshold`

The two effective knobs for this workload are:

| Knob | Default | Tuned | Effect |
|---|---|---|---|
| `max_split_size_mb` | `256` | `1024` | Prevents the allocator from splitting large cached blocks into tiny pieces, preserving contiguous free regions large enough for 20 MiB expandable page mappings. |
| `garbage_collection_threshold` | `0.7` | `0.5` | Triggers earlier release of unused cached memory, keeping more contiguous free memory available. |

### A/B results on `tests/models/test_llama3_2_gguf.py` (GPU 6, `batch_size=64`)

| `max_split_size_mb` | `garbage_collection_threshold` | Full test time | MMLU-only time | `expandable_segments` warnings |
|---|---:|---:|---:|---:|
| `256` | `0.7` | 548 s | 321 s | ~52 |
| `1024` | `0.7` | — | 223 s | 0 |
| `1024` | `0.5` | **400 s** | **213 s** | **2** (full) / **0** (MMLU-only) |
| `512` | `0.5` | — | 214 s | 2 |

The tuned config is `expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.5`.

- Eliminates almost all allocator retry warnings.
- Speeds up the full GGUF eval test by ~2 minutes (~27%).
- All evaluation metrics (`gsm8k_platinum_cot`, `mmlu_stem`, `arc_challenge`) remain within tolerance.

## How to set or override

### Bash / shell

```bash
export PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.5'
```

### Python

Set **before** importing `torch` or `gptqmodel`:

```python
import os
os.environ["PYTORCH_ALLOC_CONF"] = (
    "expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.5"
)
```

### Per-test override

`tests/models/model_test.py` uses `os.environ.setdefault`, so an existing `PYTORCH_ALLOC_CONF` in the environment is respected. This makes A/B allocator tuning easy:

```bash
CUDA_VISIBLE_DEVICES=6 PYTORCH_ALLOC_CONF='...' pytest -q tests/models/test_llama3_2_gguf.py
```

## When you might still see warnings

The two remaining warnings in the full `test_llama3_2_gguf.py` run occurred near the end of `gsm8k_platinum_cot`, when transient free memory briefly dropped below the 20 MiB expandable page size. They were non-fatal and did not affect eval metrics. To push further you can:

- Reduce `EVAL_BATCH_SIZE` (trades throughput for lower peak memory).
- Tune `expandable_segments_page_size` (requires newer PyTorch support).
- Lower `garbage_collection_threshold` further (may add GC overhead).

## Related environment variables

- `CUDA_DEVICE_ORDER=PCI_BUS_ID` — GPT-QModel sets this automatically so physical GPU IDs are stable across runs.
- `PYTORCH_CUDA_ALLOC_CONF` is a backward-compatible alias for `PYTORCH_ALLOC_CONF`.

## References

- PyTorch CUDA memory management: https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
- `CUDACachingAllocator` source and `PYTORCH_ALLOC_CONF` parsing: `aten/src/ATen/cuda/CUDACachingAllocator.cpp`
