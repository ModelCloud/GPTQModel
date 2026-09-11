---
name: gptqmodel-gpu-testing
description: Run, write, review, or report GPT-QModel GPU correctness tests, model evaluations, and performance benchmarks. Use when a task names GPU IDs, launches CUDA work, compares GPU latency or throughput, requires an idle-device preflight, runs a long evaluation, or asks for periodic live result tables.
---

# GPT-QModel GPU testing

Make the physical-device mapping, idle state, workload contract, and live progress reproducible. Combine this
skill with the relevant backend, kernel, quantization, profiling, architecture, or Evalution skill.

## Resolve physical GPUs before launch

1. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the parent environment before starting any CUDA-aware process.
2. Query `nvidia-smi --query-gpu=index,pci.bus_id,uuid,name --format=csv,noheader`.
3. Treat user-specified GPU numbers as physical IDs from that PCI-bus-ordered inventory. Record the physical ID,
   PCI bus ID, and UUID. Never infer a device capability or identity from a fixed index.
4. Restrict `CUDA_VISIBLE_DEVICES` to exactly the requested devices. Prefer the recorded UUIDs when launching
   long jobs, while retaining the physical IDs as human-facing labels.
5. After the process starts, record its visible-to-physical mapping and verify it matches the request. Stop
   immediately on a mismatch; do not silently substitute another GPU.

Use the same physical ID labels in commands, logs, result JSON, and report tables.

## Put an idle gate at the start of performance code

For every new or modified performance benchmark, place a stdlib-only preflight in the executable entrypoint
before importing Torch, initializing CUDA, loading a model, compiling a kernel, or allocating GPU memory.
Correctness tests should use the same gate when idle hardware is part of their validity contract.

The preflight must:

- inspect only the requested physical GPUs;
- reject any foreign compute process on a target;
- require 0% compute utilization for at least three consecutive samples;
- reject unexpected memory residency, using an explicit small driver-baseline allowance when zero MiB is not
  achievable on that host;
- print the accepted physical ID, PCI bus ID, UUID, utilization, memory use, sample count, and thresholds;
- fail closed with a clear message rather than waiting indefinitely, killing another process, or choosing a
  different GPU.

Implement the check through `nvidia-smi` or NVML using only pre-CUDA dependencies. Keep thresholds configurable
by command-line flags, but default formal performance runs to the strict idle contract above. Recheck immediately
before the timed region after warmup when setup is long; if the device is no longer exclusive, invalidate the run.

## Run and validate

1. Capture `nvidia-smi`, driver, Torch/CUDA versions, compute capability, SM count, memory, dtype, shapes, batch
   and token regime, backend, bits/group size, and exact command.
2. Establish correctness and a dense or higher-precision reference before timing an optimized path.
3. Warm up the exact shape class. Use CUDA events or synchronized timing and report distribution statistics.
4. Monitor only processes launched by the task. Never terminate unrelated jobs to obtain an idle GPU.
5. Preserve CPU and non-target GPU fallbacks in code changes.

## Bound host and device memory

Before launching a test, estimate peak CPU RAM, GPU RAM, and temporary disk use from tensor shapes, dtypes,
retained batches, model copies, worker count, caches, and serialization buffers. Record the estimate and compare it
with current `MemAvailable`, cgroup limits when present, free device memory, and free disk space. By default, do not
let one test retain more than 25% of physical host RAM without explicit user approval; use streaming, chunking,
bounded queues, or disk-backed batches instead. Offloading is not a fix when it merely moves an unbounded tensor
set from VRAM into CPU RAM, page cache, or swap.

Measure peak process RSS and system memory availability during long tests, not only GPU allocation. Stop a run
before it creates host-wide memory pressure, swapping, OOM risk, or an unexpectedly growing cache. Report both the
estimated and measured peaks; never hide memory pressure by omitting CPU-memory telemetry or by labeling
reclaimable cache as free capacity without showing `MemAvailable`.

Release large phase-local objects as soon as their last use completes. Use `del`, context managers, iterator
scopes, subprocess teardown, or the language/runtime's native release mechanism; then clear framework caches where
appropriate. For PyTorch, delete all live tensor references before calling allocator-cache helpers:
`torch.cuda.empty_cache()` does not free live tensors and must not substitute for bounded ownership. Explicitly
clean temporary files and disk-backed mappings after success or failure unless they are required artifacts.

## Emit live results every 60 seconds

For any evaluation or benchmark longer than 60 seconds, emit a complete live table at least once per 60-second
interval. Do not show only changed rows. Preserve every requested baseline and candidate row in every update.

Include, at minimum:

| GPU ID(s) | Candidate | Model/decoder bits | Endpoint bits | Size MB | Metrics | State |
|---|---|---|---|---:|---|---|

- `GPU ID(s)` means physical PCI-bus-ordered IDs, not process-local CUDA ordinals.
- Show the actual running partial aggregate and completed sample count when the evaluator emits one.
- Mark buffered or unavailable metrics as `running` or `pending`; never invent a score.
- Distinguish prefix samples from the live full-run aggregate with footnotes.
- Include invalid-result counts and utilization when they materially explain run health.
- In the final table, replace partial values with exact serialized results and identify interrupted or invalid runs.

When several jobs run in parallel, poll them together so one 60-second update contains the full comparison matrix.

## Handoff

Report exact physical GPU IDs and UUIDs, whether the initial and pre-timing idle gates passed, any threshold
exceptions, commands, artifacts, final metrics, model sizes, and validation status. State clearly when a run was
correctness-only, compiled-only, skipped, interrupted, or invalidated by device contention.

## See also

- [Curated GPU performance engineering resources](references/wafer-gpu-perf-resources.md) — External reading list for serving benchmarks and correctness from wafer-ai's performance engineering index.
