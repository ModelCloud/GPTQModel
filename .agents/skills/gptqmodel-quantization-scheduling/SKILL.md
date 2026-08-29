---
name: gptqmodel-quantization-scheduling
description: Plan, launch, and monitor concurrent model quantization jobs using model-size and measured VRAM demand. Use when assigning quantization or evaluation arms to GPUs, deciding whether multiple quantizers can share a GPU, managing queues, or recovering safely from OOM and stale jobs.
---

# VRAM-aware quantization scheduling

Quantization and evaluation have different memory profiles. A small model may leave enough headroom for two or more quantizers on one GPU, while a large model or a post-quant evaluation may require an exclusive device. Choose concurrency from measured peak memory and a safety reserve; never assume that a fixed “one process per GPU” or “N processes per GPU” policy is portable.

## Classify the workload first

Record these inputs before placing a job:

- model parameter count, architecture (dense/MoE), and number of layers;
- source/checkpoint dtype and temporary reconstruction dtype;
- quantization method, format, bits, group size, calibration rows/tokens, batch size, and replay/search settings;
- whether the process loads a dense reference, teacher logits, candidate banks, or multiple model copies;
- expected output/checkpoint size and filesystem throughput;
- whether the job is quantization, calibration/profile-only, or inference/evaluation.

Treat the following as separate resource classes:

1. **Quantization:** often has a large dense model plus temporary Hessian/reconstruction buffers. It may be shareable for small models, but two jobs can contend for memory bandwidth and CPU/I/O.
2. **Evaluation:** may load both dense and quantized models, tokenizer/evaluator state, and long-lived KV/cache buffers. Default to one evaluation per GPU unless a measured profile proves safe sharing.
3. **Compilation/profiling:** CUDA extension builds and profilers can add transient memory and should use an exclusive device unless explicitly measured.

## Estimate a safe concurrency level

Prefer a short pilot run that records `torch.cuda.max_memory_reserved()` and `max_memory_allocated()` over a static guess. Include startup, calibration, peak reconstruction, serialization, and evaluator warm-up; the steady-state number is not sufficient.

For GPU (g), let (M_g) be total VRAM, (R_g) the driver/framework reserve, and (P_j) the observed or conservative estimated peak for job (j). Admit a set of jobs only when:

```text
sum(P_j) + R_g <= M_g
```

Use a configurable reserve (normally 10–15% of VRAM, or a larger measured host-specific floor). Then choose:

```text
concurrency[g] = min(max_slots, largest safe number of jobs)
```

where `max_slots` is normally 1 for evaluations and can be 2 or more for small quantizers. A useful initial estimate is:

```text
slots = floor((total_vram - reserve) / conservative_peak_per_quantizer)
```

Clamp to at least one and validate with a canary. Do not use free memory alone: another process may allocate after the sample, and CUDA allocators retain cached blocks.

### Example

On a 96-GB A100, the Llama 3.2 1B Wave-7 quantization pilot used about 55.9 GB total while two quantizers were active on one GPU (roughly 28 GB each in that workload). Two quantizers therefore fit with headroom for that exact configuration, but the result is not a universal per-process guarantee. The corresponding evaluation workload approached 84 GB and was kept exclusive.

## Use leases and per-GPU slots

Every launched process must have a unique `(physical_gpu, slot)` lease. A lease records:

- physical GPU ID, PCI bus ID, UUID, and visible CUDA ordinal;
- job/arm ID, checkpoint/config fingerprint, and process PID;
- estimated peak, observed peak, start time, and state;
- lease expiry/heartbeat and the cleanup owner.

Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` and a narrowly scoped `CUDA_VISIBLE_DEVICES`. Prefer UUIDs for long jobs. A GPU can have multiple **quantization** slots, but an **evaluation** slot should acquire an exclusive GPU lease and wait for all quantization slots to release it.

Do not treat a stale lock file as proof that a live process exists. Validate the PID, command line, lease heartbeat, and GPU UUID; expire only a lease whose owner is gone. Never kill an unrelated process to make room.

## Queue algorithm

1. Inventory physical GPUs and memory with `nvidia-smi`/NVML.
2. Run the idle/ownership gate required by the GPU-testing skill.
3. Estimate or pilot each arm’s peak memory.
4. Sort jobs by readiness and priority, then place each on a GPU with enough remaining budget and the least projected contention.
5. For small quantizers, fill a second slot on a GPU only after the first canary passes and the summed peak plus reserve remains safe.
6. When quantization finishes, release its slot and enqueue evaluation. Evaluation must acquire an exclusive lease; never overlap it with a quantizer unless an explicit measured policy permits it.
7. On OOM, mark the attempt failed, reduce that arm’s concurrency requirement (or move it to an exclusive slot), clear only its own lease, and retry once with a larger reserve. Do not silently continue with a partial checkpoint.
8. On repeated OOM or device loss, leave the arm pending/blocked with the diagnostic and keep other GPUs progressing.

The scheduler should be able to use more than two slots when telemetry supports it, but start conservatively. Compute utilization, memory bandwidth, host RAM, CPU threads, disk read/write rate, and checkpoint contention can make three small jobs slower than two even when VRAM permits them.

## Required telemetry and result records

For every arm, append (rather than overwrite) a record containing:

```text
model, model_numel, quant_format, bits, group_size
calibration_rows, calibration_tokens, batch_size
physical_gpu, pci_bus_id, gpu_uuid, visible_cuda_ordinal
slot, concurrency_on_gpu
estimated_peak_vram_mb, observed_peak_vram_mb
quant_start, quant_finish, eval_start, eval_finish
state, retry_count, oom_or_device_error
checkpoint_path, config_fingerprint, result_paths
```

Emit a complete queue table at least every 60–120 seconds for long runs. Show all arms, not only changed rows, and distinguish `queued`, `quantizing`, `evaluating`, `completed`, `failed`, `oom_retry`, and `invalid`. Partial scores must include their sample count and must never replace the final serialized result.

## Correctness and reproducibility gates

- Verify every arm’s resolved per-module configuration/fingerprint before launch; reject duplicate fingerprints unless the duplicate is an intentional seed/reproducibility control.
- Keep quantization calibration and benchmark manifests disjoint according to the project’s contamination protocol.
- Validate that a checkpoint saved by a concurrent quantizer reloads and forwards correctly before spending a full evaluation slot.
- Record the exact concurrency policy and observed peak. A result produced under memory pressure or an OOM-recovered partial run is not comparable until it passes the same reload/evaluation checks.
- Preserve deterministic seeds and separate output directories for every arm and retry.

## Implementation pattern

Keep scheduling policy separate from quantization code. A launcher should accept an explicit physical GPU and slot, acquire a lease, set `CUDA_VISIBLE_DEVICES`, write a heartbeat, sample peak VRAM, and release the lease in a `finally` path. Quantization workers should not inspect or mutate another worker’s locks. The queue/orchestrator owns retries, evaluation handoff, periodic status snapshots, and append-only result commits.

Use the shared GPU allocator when available for physical-device ownership. For a local queue, implement the same contract with per-GPU/per-slot locks, PID validation, and a conservative reserve; do not rely on a single global lock that prevents safe small-model packing.

## Decision rule

The default policy is:

```text
small model + measured headroom + quantization only -> 2 slots, then canary-test more
large model or uncertain peak                     -> 1 slot
evaluation / profiling / compilation              -> exclusive GPU
any OOM or unexplained memory growth              -> reduce concurrency and retry safely
```

The objective is maximum completed, valid experiments per host—not maximum nominal process count. Prefer fewer concurrent jobs when they improve throughput, avoid OOMs, or protect reproducibility.
