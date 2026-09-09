---
name: cohere-megakernel
description: Apply lessons from Cohere's decode megakernel to persistent GPU task scheduling, fine-grained dependency counters, warp-specialized pipelines, ragged attention or MoE queues, and graph-safe QvQ/ZML integration. Use for designing or reviewing these mechanisms and evaluating whether they improve decode latency.
---

# Cohere megakernel engineering

Optimize useful work along the measured decode critical path. A single launch
is a design choice, not an acceptance criterion. This skill distills
[cohere-ai/cohere-megakernel](https://github.com/cohere-ai/cohere-megakernel)
at commit `67d0b9ca22ea3652796b715d1d1863459e0e2c3c`.
Read [the source map](references/source-lessons.md) when choosing a mechanism
or checking an upstream claim; it distinguishes observed implementation from
porting guidance. The upstream release targets North Mini Code, H100/SM90a,
BF16 and small batches. Its performance and model schedule do not transfer
automatically to another model, dtype, quantization format or GPU.

## Select the opportunity

Measure a warmed, matched baseline through the actual public entry point.
Record device capabilities, source/build identity, shapes, dtype, batch size,
context lengths, active rows, expert distribution and quality mode. Separate
host gaps, GPU launch gaps, partial waves, dependency waits and memory traffic.
Compare against the existing graph-replayed or fused baseline where available.

Choose a bounded region with measurable idle time, intermediate traffic or
launch overhead. Retain the existing path until the candidate passes numerical,
runtime and full-operation performance gates. Ordinary fusion, existing CUDA
Graphs or a native multi-launch wrapper may already remove the measured cost.

## Schedule dependencies rather than entire operators

1. Draw the actual model DAG at GEMM-tile, KV-head, attention-split or expert-tile
   granularity. Record each task's input region, output region, producer count
   and earliest legal execution. A consumer waits for its own inputs, not
   unrelated tiles. Include reductions and residual joins as real dependencies.
2. Fill partial waves with independent tasks. Cohere overlaps attention and FFN
   work because its model has parallel branches. A sequential transformer FFN
   cannot start from the pre-attention activation merely to obtain this overlap.
3. Use host-prepared per-block instruction lists for predictable work. Prove
   progress for the assigned order: even an acyclic data DAG can deadlock if
   every resident worker waits for a producer queued behind a waiter.
4. Use bounded dynamic queues where runtime imbalance justifies them. Cohere
   drains full-attention and routed MoE tasks dynamically while keeping bounded
   sliding-window attention static. Derive queue lengths and combine thresholds
   from the same live row lengths or routing metadata. Check capacity, empty
   work, inactive rows, and exactly-once task claims.
5. Specialize stable geometry and cache schedules outside execution. Invalidate
   on relevant device, shape, layout, artifact, quality-mode or tiling changes.
   Live lengths and routing must still be refreshed safely for each step.

## Define a task calling convention

Make opcodes, operand layouts, tile coordinates, wait/signal counters and scratch
ownership explicit. Validate the host encoder against the device decoder and
version any separately compiled ABI. Cohere's fixed-width instruction records
and layout-locked launch descriptor illustrate the boundary, not a format the
target repository must adopt.

Separate controller, producer, compute and store roles when that permits useful
overlap. Prefetch immutable weight tiles before activation dependencies resolve;
load activations only after their producer is visible. Sweep a small number of
prefetch stages per operation and batch regime. Extra stages consume shared
memory and can regress small batches.

Budget registers, shared memory, code size and resident blocks for the whole
compiled kernel. Cohere separates compile-time warp-role call paths so register
redistribution can work; verify generated resource usage rather than assuming
role labels lower register pressure. Probe the target device and prove residency
for any spinning cross-block dependency scheme. Never copy the upstream SM count
or treat `__launch_bounds__` as a scheduling guarantee.

## Prove handoffs and state reuse

For every dependency, document:

```text
output writers -> async completion -> publication -> consumer observation
counter: address, initial value, increments, threshold, reset/epoch owner
storage: first writer, last reader, next overwrite, execution-lane owner
```

Publication must follow completion of all relevant stores. Use the target's
supported memory-ordering primitives, including async-proxy handling for TMA;
do not generalize upstream volatile polling and fences into a portable atomic
protocol. Atomic arrival alone does not establish every payload's visibility.

Every participating worker must execute matching logical barriers, including
NOP, inactive-row, short-tail and optional-op paths. A prefetched instruction
ring slot or pipeline stage may be reused only after all of its readers finish;
one warp finishing does not release storage still used by sibling warps.

Keep control counters disjoint from payload scratch unless lifetime proofs cover
every subsequent route. Reset counters, queue heads and reduction buffers before
the next step using ordered execution, or use a proven epoch scheme. Test the
same workspace across repeated calls, shape changes, specialized/generic route
changes and graph replay. A fresh allocation on each test can conceal corruption.

For a native serving loop, define a completed pause/park handshake before host
code changes geometry, slots or pointers. Retain tensors, KV handles, JIT
libraries and callback owners through asynchronous use. A timeout reports a
stalled kernel; it does not cancel device work or make its buffers reusable.

## Apply in these repositories

**QvQ:** start at `gptqmodel/utils/qvq_cuda.py`, the selected implementation in
`gptqmodel_ext/qvq/`, and its caller in `gptqmodel/nn_modules/qlinear/qvq.py`.
Use the existing `$gptqmodel-mega-kernels` workflow for general fusion work.
Distinguish offline quantization/Viterbi work from inference decode. Preserve
packing, trellis semantics, transforms, scales, correction behavior, accumulation
contracts and CPU/non-target fallbacks. Re-derive the memory model for compressed
weights: unpacking or reconstruction can change whether weight prefetch helps.
Use real weights and disjoint real activations for quality decisions; synthetic
fixtures are appropriate for algebra and scheduling edge cases only.

**ZML-Ultra:** start at `integrations/qvq_window/` for the QvQ native bridge, or
the selected implementation under `kernels/` and its ZML caller. Follow the
local `$graph-safe-kernels` skill. Prepare compilation, tuning, validation,
descriptors and storage outside capture. Preserve buffer/stream/library and
nested-graph ownership until enclosing graphs and in-flight work finish. Freeze
quality eligibility before tuning; disabling correction is not an equivalent
faster candidate. Validate the complete StableHLO/PJRT/FFI execution path before
advertising command-buffer compatibility. A leaf CUDA test or capture rejection
does not satisfy this contract. Preserve the existing sharding/TP semantics.

These are starting points, not permission to migrate the runtime to Cohere's
server or impose H100-only execution. Resolve paths against the current checkout.

## Validate and explain the result

- Compare each changed operation and the complete output with the established
  unfused/dense reference under predeclared numerical tolerances. Include real
  model logits or task-quality checks when arithmetic or model behavior changes.
- Exercise uneven tile tails, empty queues, skewed expert assignments, ragged
  lengths, inactive rows, context-bucket changes and repeated state reuse as
  applicable. Use bounded subprocess tests for deadlocks and relevant sanitizer
  checks for changed handoffs; passing a race tool is not a memory-order proof.
- Compare eager and repeated graph execution through the public integration,
  changing input contents at stable addresses. Cover non-default streams,
  allocator pressure, ownership cleanup and the supported fallback routes.
- Use per-SM task timelines to distinguish useful work from dependency waits.
  Cross-check uninstrumented timing: profiler ranges may exclude waits and the
  upstream SM profiler documents timer distortion immediately after barriers.
- Report kernel-only, full decode-step and end-to-end latency separately, with
  p50/mean/p95, throughput, memory and resource usage. Include reset, embedding,
  sampling, metadata transfer and prefill costs where they belong. Match token
  counts or report the difference; synthetic KV timing is not accuracy evidence.
- Ablate the proposed mechanism: schedule overlap, dependency granularity,
  prefetch or queue policy. Retain a win only when repeated full-operation
  evidence supports it without violating quality or runtime contracts.

Deliver the dependency/lifetime argument, exact specialization and fallback
conditions, validation commands/results, matched performance evidence and any
unverified hardware cases. Do not claim target performance from upstream numbers.
