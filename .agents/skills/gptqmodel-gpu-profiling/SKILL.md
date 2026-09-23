---
name: gptqmodel-gpu-profiling
description: Capture, inspect, or compare GPT-QModel GPU profiles using torch.profiler, Chrome/Perfetto traces, Nsight Systems, or Nsight Compute. Use to attribute quantized inference time, separate prefill and decode, diagnose launch gaps or communication overlap, map kernels to source, identify fusion candidates, or validate whether an optimization changed the real bottleneck.
---

# GPT-QModel GPU profiling

Profile a correct, representative workload and preserve enough context to reproduce it. A trace explains where time went in that run; it does not by itself prove that a proposed optimization is safe or faster end to end.

Read [references/profiling-workflow.md](references/profiling-workflow.md) before capturing or comparing traces. Use `$gptqmodel-cuda-kernels` for kernel changes and the matching Ampere, Hopper, or Blackwell skill for architecture claims.

## Establish the question and baseline

1. State whether the target is quantization, packing/load, prefill, decode, graph capture, communication, or one kernel.
2. Run a quick correctness or model-quality check before profiling. Do not optimize a trace from a numerically broken backend.
3. Record model, method, format, backend, bits, group size, symmetry, `desc_act`, dtype, batch/tokens, cache state, GPU properties, software stack, and exact command.
4. Complete JIT compilation, allocator growth, and representative warmup before the active capture unless startup is the question.

## Choose the tool by scope

- Use `torch.profiler` for Python/operator-to-CUDA attribution and portable Chrome/Perfetto trace artifacts.
- Use Nsight Systems for CPU launch gaps, CUDA graphs, streams, NCCL, synchronization, and end-to-end timeline overlap.
- Use Nsight Compute for focused metrics on a small number of already-identified kernels. Do not collect its heavy metric sets across an entire model run by default.
- Use the repository's synchronized benchmarks for final latency and throughput. Profiler timings include instrumentation overhead.

## Capture representative stages

Prefer separate prefill and decode traces. Keep prompt length, generated length, batch/concurrency, KV-cache state, prefix-cache behavior, and graph mode explicit. Warm up with the same shape class, then capture a small bounded number of active steps.

For a difficult source mapping, collect two matched traces:

1. a mapping trace with graphs or aggressive fusion disabled enough to recover operator and Python attribution;
2. a formal trace with the production configuration enabled for the actual performance conclusion.

Do not compare the two as if disabling graphs were free. In distributed runs, start with one rank; capture all ranks only when communication imbalance or cross-rank ordering is the question.

## Analyze deterministically

1. Rank kernels by cumulative GPU time and call count, separated by stage.
2. Identify uncovered CPU launch gaps, serialization, synchronization, and communication that could overlap, with dependencies stated.
3. Map dominant kernels to the local wrapper, backend, and extension source before recommending a change.
4. List fusion candidates only when the producer-consumer relationship and layout/dtype contract support them. Do not classify by fuzzy kernel-name similarity alone.
5. For a dominant custom CUDA kernel, inspect a source-correlated SASS page rather than reasoning from C++ alone. Use executed instruction mix, scheduler eligibility/stalls, sectors and cache traffic, shared requests/wavefronts/conflicts, resource limits, and tensor-pipe activity together. Low DRAM GB/s does **not** imply bad alignment: compressed/decode-heavy kernels may be instruction-, dependency-, shared-, or Tensor-Core-issue limited. Follow the matched workflow in the reference.
6. Re-run the synchronized benchmark and correctness check after any optimization; then capture a matched follow-up trace if attribution changed.

Return the artifact path, capture command, environment/configuration record, whether evidence is mapping or formal, and complete ASCII tables for dominant kernels, overlap opportunities, and source-backed fusion candidates. Clean up only processes launched by the profiling run, using recorded PIDs and graceful termination before escalation.

## See also

- [Curated GPU performance engineering resources](references/wafer-gpu-perf-resources.md) — External reading list for GPU profiling and correctness from wafer-ai's performance engineering index.


## A100+ diagnosis rules

- **Global alignment/coalescing:** inspect lane addresses and sectors/request.
  Do not infer coalescing from `is_contiguous()`, 128-byte allocation alignment,
  or vector type alone.
- **Shared memory:** report requests, wavefronts and bank conflicts. For ordinary
  accesses use 32 banks / 4-byte granularity; for TMA/WGMMA/TCGen05 operand
  layouts use the architecture-defined/CuTe layout rather than a generic padding rule.
- **Scheduler:** pair occupancy with eligible warps/scheduler/cycle and dominant
  stall reasons. A one-CTA/SM kernel may be optimal if reuse is high; a high-
  occupancy kernel may still be dependency-bound.
- **Async pipelines:** verify `cp.async`/TMA issue and wait/barrier behavior,
  not just memcpy byte counts. Excess waits or a wrong stage depth can dominate.
- **Tensor Core path:** prove HMMA/WGMMA/TCGen05 activity from generated/executed
  code. A source call to WMMA/CuTe or a compatible dtype is not proof.
- **HBM percentage:** compare useful bytes, arithmetic/decode work, cache hits,
  and instruction issue. Optimize for latency/throughput, not for maximizing GB/s.
