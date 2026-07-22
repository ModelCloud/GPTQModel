---
name: gptqmodel-mega-kernels
description: Design, fuse, profile, optimize, review, or debug GPT-QModel CUDA mega-kernels that combine multiple GPU kernels or operator phases into one native launch. Use when reducing launch count, building cooperative or persistent kernels, reusing scratch across phases, scheduling CTA/warp roles, managing grid barriers or atomics, diagnosing fused-kernel tail imbalance, controlling register/shared-memory pressure, preserving CUDA-graph and fallback behavior, or deciding whether multi-kernel fusion is worthwhile.
---

# GPT-QModel mega-kernels

Treat launch reduction as a hypothesis, not the objective. Optimize end-to-end
latency or memory while preserving numerical, dispatch, graph-capture, and
fallback contracts.

Read [references/mega-kernel-playbook.md](references/mega-kernel-playbook.md)
completely before designing or changing a mega-kernel. Also use
`$gptqmodel-cuda-kernels` and `$gptqmodel-gpu-profiling`; add the matching
Ampere or Hopper skill for architecture-specific work. For Marlin+EoRA, inspect
the latest retained and rejected experiments in `eora_marlin.md`.

## Establish the fusion contract

1. Enumerate the current kernels, host operators, allocations, streams, launch
   order, shapes, layouts, dtypes, and numerical accumulation order.
2. Draw the producer-consumer dependency DAG. Mark which phases are independent,
   which require CTA synchronization, and which require device-wide visibility.
3. Establish a dense or unfused numerical reference and a matched production
   baseline before editing.
4. Define the narrow specialization gate from runtime-probed capabilities and
   immutable shape/configuration facts. Never infer capability from a CUDA index.
5. Preserve the existing path for unsupported shapes, dtypes, architectures,
   devices, graph capture, launch failure, and optional features.

## Profile before fusing

Capture a bounded Nsight Systems baseline after JIT and allocator warmup. Report
per-forward kernel count, individual kernel durations, inter-kernel gaps, GPU
span, launch API time, host range, and allocations. Then use Nsight Compute on
the dominant constituent kernels to record:

- registers, static/dynamic shared memory, local memory, spills, and occupancy;
- eligible-warps and issue rates, warp cycles per issued instruction, and stalls;
- DRAM/L2/L1 traffic and hit rates, compute utilization, and instruction count;
- source-correlated hotspots, CTA/SM imbalance, and tail waves.

Answer these questions first:

1. Do launch/API/gap costs occupy a meaningful fraction of the target latency?
2. Is the chain truly serial, or can independent work overlap on streams?
3. Can one launch geometry serve every phase without wasting most warps or CTAs?
4. Does the union of phase resources still permit the required residency?
5. Which phase owns the critical-path tail, and what work is idle at that point?
6. Can dead scratch be reused only after its last reader and required memory fence?
7. Will cooperative launch or graph capture narrow portability or deployment?

Prefer a native operator that still enqueues multiple kernels when Python and
dispatcher overhead is the measured problem. Build a single GPU mega-kernel only
when removing GPU launch/gap costs or enabling phase-local data reuse is material.

## Design from phase boundaries inward

1. Keep phase roles explicit in source even when they share one launch.
2. Budget the worst live registers and launch-reserved shared memory before
   implementation. A later lightweight phase does not release earlier resources.
3. Use block synchronization for shared-memory handoff and cooperative grid
   synchronization only for proven cross-CTA dependencies or scratch aliasing.
4. Reuse global or shared scratch only after the prior phase's final read and the
   visibility boundary required by the next phase.
5. Map remainder tiles onto otherwise-idle warps or threads before adding a
   serial CTA pass.
6. Reduce atomics hierarchically, but prefer coalesced memory access over a lower
   atomic count when the two conflict.
7. Sweep bounded instruction-level parallelism; reject variants that improve
   median while regressing mean, p95, registers, spills, or another dtype.
8. Preserve the exact accumulation grouping when practical. Treat a changed
   reduction order as a correctness change requiring new error evidence.
9. Compile specialization-only phase code out of generic kernels so its resource
   footprint cannot degrade fallbacks.

## Run a disciplined candidate loop

Change one mechanism at a time. Record the exact source revision or JIT
fingerprint and keep rejected variants out of the retained diff.

1. Run the focused correctness reference.
2. Warm up and collect a normal synchronized latency/memory distribution.
3. Capture at least 200 raw launches with matched Nsight Systems settings for
   microsecond-scale kernels; compare p50, mean, p95, minimum, and maximum.
4. Repeat apparent small wins and validate them on the second supported dtype or
   device.
5. Run full Nsight Compute only on credible winners and compare resources and the
   original bottleneck metrics.
6. Reject a median-only win when mean or tail is flat/worse within profiler
   resolution. Do not use NCU replay duration as the sole performance result.

Separate raw-kernel attribution from full-operator impact. CUDA-event buckets,
clock state, allocator reuse, instrumentation, and host dispatch can hide or
exaggerate a one-microsecond kernel change.

## Validate the retained source

Cover:

- dense/unfused reference agreement, shape, dtype, device, and finite output;
- every fused dtype and representative fallback shapes/ranks/row counts;
- repeated calls, current/non-default streams, multiple visible devices, and
  graph capture;
- Compute Sanitizer memcheck plus synccheck/racecheck when barriers or shared
  handoffs change, generated-source checks, and focused tests;
- exact launch count, architecture gate, resource use, and allocator peak;
- compilation and execution on the target hardware, clearly distinguishing each.

Log the environment, commands, baseline, retained deltas, rejected candidates,
profiler artifact paths, JIT fingerprints, errors, resources, and fallback status.
