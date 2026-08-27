---
name: gptqmodel-metal-profiling
description: Profile MLX custom Metal kernels on Apple silicon with Xcode Instruments and GPU capture, then turn barriers, launch gaps, memory duplication, and occupancy clues into measured kernel optimizations.
---

# GPT-QModel Apple Metal profiling

Use this skill for MLX `fast.metal_kernel` work and Apple GPU performance on M-series Macs. It complements
`gptqmodel-gpu-profiling` for CUDA/portable profiling and `gptqmodel-cuda-kernels` for CUDA implementation work.
Keep the checkpoint format, numerical contract, and non-Apple fallback unchanged unless the user explicitly asks for a
format change.

## Tool map

- **Xcode Instruments — Metal System Trace:** CPU submission, command-buffer intervals, queue ordering, launch gaps,
  synchronization, and coarse GPU activity.
- **Xcode Instruments — Metal GPU Counters:** hardware counters and shader statistics when the selected counter profile
  is supported by the installed OS, Xcode, and device.
- **Xcode GPU Frame Capture:** per-dispatch shader/resource inspection from an MLX `.gputrace` capture.
- **MLX Metal capture:** `mx.metal.start_capture(path)` / `mx.metal.stop_capture()` for a bounded workload capture.
- **Synchronized benchmark:** final latency evidence; profiler timings are diagnostic and include instrumentation cost.

Read [references/xcode-metal-workflow.md](references/xcode-metal-workflow.md) before a formal capture. It contains the
reproducible command templates and report contract.

## Required workflow

1. **State the question.** Identify whether the target is decode, prefill, one GEMV/GEMM shape, a complete quantized
   linear, or end-to-end model latency. Profile prefill and decode separately when both matter.
2. **Prove correctness first.** Run the focused Torch/dense oracle and MLX numerical tests for the exact kernel,
   transition width, dtype, shape, tail, and bank/layout variant. A faster incorrect kernel is not an optimization.
3. **Record the environment.** Capture the commit, model/format/bits, shape and batch/token regime, Python/MLX/Torch/
   Xcode/macOS versions, `mx.metal.device_info()`, and power/thermal state. On a laptop, use AC power and record the
   selected power mode; otherwise label results as potentially frequency-variable.
4. **Warm up before capture and timing.** Complete JIT compilation, allocator growth, and representative shape warmup.
   Every timed run must force completion with `mx.eval(output)` and `mx.synchronize()` (or the exact equivalent used by
   the benchmark).
5. **Capture a bounded trace.** Prefer a short MLX `.gputrace` for dispatch/resource inspection and a matching Metal
   System Trace for CPU/GPU scheduling. Use an absolute interpreter path with `xctrace`; a bare `python3` can make the
   launcher lose the target process on some installations.
6. **Use counters only when available.** If Xcode reports that the GPU counter profile is unsupported, record that fact
   and do not invent occupancy, cache, stall, or utilization percentages. Use the trace, source topology, and matched
   A/B timings for structural conclusions instead.
7. **Map evidence to source.** Match dispatch names/templates to the Python builder and Metal source. Inspect
   `threadgroup_barrier`, SIMD-group communication, shared/constant/device memory, atomics, split-K partials, and
   output epilogues. Count barriers per K tile and identify which SIMD groups are idle during each phase.
8. **Test one change at a time.** Keep a same-process baseline and candidate A/B with identical payloads, warmup,
   samples, synchronization, and ordering. Retain an architecture-gated fallback when a specialization loses on the
   target device.
9. **Re-run correctness and full-module timing.** Report both the inner kernel and the complete `QVQMLXLinear` path;
   include transform, narrowing/scaling, reductions, epilogues, and MLX graph overhead. Do not promote an inner-kernel
   win that loses at the module boundary.

## What to look for

- **Barrier stalls:** a decode producer in one SIMD group followed by barriers while sibling groups wait. Consider
  cooperative decode, SIMD shuffles, or a register-only small-M path only after measuring duplicated decode cost.
- **Duplicate loads:** one selector, code word, activation, or level-table entry loaded by many lanes. Test one-lane
  loads plus `simd_shuffle`/broadcast, preserving the exact lane mapping.
- **Occupancy/underfill:** one SIMD group or a tiny grid for a wide matrix. Compare row tiles, output tiles, split-K,
  and independent work before reaching for matrix instructions.
- **Launch gaps and host boundaries:** repeated Python dispatch, `.item()`, implicit evaluation, graph breaks, or
  materialized split-K reductions. Immutable metadata should be resolved at construction and specialized when safe.
- **Memory/compute overlap:** stage independent decode and activation work only when dependencies permit it. State
  recurrence within one local ring is ordered; do not claim overlap that crosses a required barrier or changes the ABI.
- **Epilogue traffic:** fixed-size split-K reductions, Hadamard transforms, row scaling, and bias/output transforms may
  dominate the complete module even when the quantized GEMV improves.

## Benchmark and reporting contract

Use a complete shape table, not only favorable rows. At minimum include M/K/N, dtype, split/row/output tile, p50, p95,
sample count, and P32/non-local baseline speed ratio. For full modules include the same fields plus total latency and
throughput where meaningful. Report variance across repeated runs when power state or background GPU activity is not
controlled.

For a profiling report, include:

- capture commands and absolute artifact paths/sizes;
- environment and power state;
- correctness result and numerical tolerance;
- dominant dispatch table with calls, total/mean time, and shape;
- overlap/stall table with dependency evidence and confidence;
- fusion candidates with producer/consumer layouts and dtype;
- final synchronized A/B benchmark and the decision to enable, gate, or reject the change.

Do not commit `.trace` or `.gputrace` bundles unless the user explicitly requests them. Captures can contain model
paths, request metadata, resources, and inputs. Do not claim Apple Neural Engine execution for MLX custom Metal
kernels: MLX custom kernels run on the Apple GPU; ANE use would require a separate Core ML/compiler path.
