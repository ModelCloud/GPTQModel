---
name: gptqmodel-telemetry
description: Inject and interpret lightweight telemetry in GPT-QModel. Use for region timings (module_load, module_move, torch_sync), NVTX annotations, disk telemetry, and logbar throttling during debug or profiling.
---

# GPT-QModel telemetry

Use when asked to add timing, log regions, NVTX annotations, or disk telemetry to find where time is spent.
Not for permanent production logging.

## When to use

- Adding `module_load`, `module_move`, `torch_sync`, or other region timings.
- Instrumenting a long quantization or inference pipeline for hotspots.
- Using `nvtx.annotate` to mark phases for `nsys profile`.
- Interpreting `disk_telemetry` for shard I/O or checkpoint conversion.
- Throttling progress bars (`logbar`) so they do not spam headless logs.

## Key files

- `gptqmodel/utils/torch.py` (`torch_sync` helpers)
- `gptqmodel/utils/disk_telemetry.py` (disk I/O timing)
- `gptqmodel/utils/logger.py` (`logbar` / progress throttling)
- `gptqmodel/looper/module_looper.py`, `gptqmodel/models/base.py` (where region timings are injected)
- `scripts/profile_*.py`, `scripts/benchmark_*.py`

## Workflow

1. **Add telemetry before optimizing.**
   - Identify the region you suspect (load, materialize, forward, quantize, save).
   - Wrap it in a context timer or `nvtx` range; do not add per-element logging.

2. **Prefer deterministic wall-clock or CUDA events.**
   - CPU regions: `time.perf_counter()` around the block.
   - GPU regions: `torch.cuda.synchronize()` before and after, or CUDA events.
   - Mixed CPU/GPU pipelines: record both CPU and GPU time and report the gap.

3. **Annotate, do not measure, with NVTX.**
   - Use `@nvtx.annotate` or `with nvtx.annotate(...)` to label `forward`, `backward`, `data_load`, `quant_step`.
   - Do not put `torch.cuda.synchronize()` inside tight loops just for NVTX.

4. **Use `disk_telemetry` for I/O.**
   - Wrap safetensors open/read/write calls to pinpoint slow shard loads.
   - Report bytes and wall time, not just calls.

5. **Keep output usable.**
   - Emit one summary line per region at the end (mean/median/p95, count).
   - Avoid printing every iteration; use live tables only for long-running jobs.

6. **Remove or gate before merging.**
   - Mark debug telemetry as `if log.isEnabledFor(logging.DEBUG):` or remove after diagnosis.
   - Do not leave noisy per-tensor timers in hot paths.

## Anti-patterns

- Do not add `print` inside a hot forward loop.
- Do not call `torch.cuda.synchronize()` on every tensor op.
- Do not measure with `time.perf_counter()` for pure GPU kernels without sync.
- Do not leave `nvtx` annotations that annotate sub-microsecond regions.
