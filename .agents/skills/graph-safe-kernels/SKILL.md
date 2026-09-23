---
name: graph-safe-kernels
description: Implement and validate graph-safe GPU kernels and runtime integrations, including QvQ, other kernel libraries, grouped operators, FFI custom calls, and compiler dispatch.
---

# Graph-safe kernels and integrations

All kernel code and integration with QvQ or any other kernel library must be
made graph safe. This is a required runtime contract, not an optional speed
feature. Apply it to the complete operation, including transforms, grouped
projections, correction branches, dispatch, custom calls and library wrappers.
For CUDA use CUDA Graph capture/replay; validate the equivalent mechanism for
other graph-capable backends. Do not label an untested backend graph safe.

## Separate preparation from execution

Complete compilation, autotuning, artifact/hash validation, packing, lazy library
initialization, immutable conversions and host reads of device values before
capture. Provide an explicit preparation API when needed. The captured path must
not perform host synchronization, timing, device-to-host scalar reads, filesystem
work or lazy setup. Allocation must use capture-aware storage with a lifetime
owned by the executable/graph; ordinary temporary allocation is not automatically
safe just because an eager call succeeds.

Freeze geometry, shape policy and optional correction state per graph. Resolve
quality eligibility before tuning; latency must not turn correction off to win.
Mode or shape changes select a separately prepared graph or invalidate and
recapture outside execution. Keep correction tensors completely unaccessed when
disabled.

## Own lifetimes and ordering

Specify ownership of every input/output, artifact, temporary pool, descriptor,
event, library handle and nested graph. Retain them through all asynchronous
uses and enclosing graph lifetimes. Define output validity across repeated
requests. Prevent workspace reuse across concurrent requests by explicit stream
ordering or independent execution-lane storage. Use the caller's device and
stream; preserve dependencies before and after nested/custom operations.

A child graph clone does not transfer ownership of its device allocations.
Retain its pool owner until all enclosing graphs and in-flight work finish.
Reject stale handles/configuration, incompatible devices or unsupported capture
conditions before submitting unsafe work. Such rejection is containment while
implementation is incomplete, not satisfaction of the graph-safety requirement.

For ZML/PJRT/StableHLO, advertise command-buffer compatibility only after the
actual FFI executable has passed capture/replay tests. Python graph success is
not proof for an external runtime. Preserve sharding/TP ownership independently
of graph support.

## Verify the public path

Compare eager and captured complete outputs under the existing numerical
contract on identical inputs and artifact state. Cover repeated replay with
changed input contents at stable addresses, allocator pressure, non-default
streams, dependency ordering, off/on and independently enabled grouped children,
shape tails and dispatch boundaries. Test invalidation, cleanup and failure
paths, including attempted cross-stream/shared-workspace misuse. Exercise
external-runtime capture when changing its integration, not just a leaf kernel.

Record which devices, shapes, rates, modes and integration paths actually ran.
Keep untested or failing cases open. Do not infer all-QvQ graph safety from one
operator or one device. Existing accuracy, profiling and model-quality gates
continue to apply; graph safety does not authorize numerical changes.

## Memory-aware host compilation

The user authorizes up to half the host's available CPU cores for compilation,
subject to memory headroom and no OOM or swap pressure. Compute the ceiling from
the current CPU affinity/quota, then reduce it for available/cgroup memory,
measured peak compiler memory per job, other active builds, and compiler-internal
threads. Use a conservative first build when per-job memory is unknown; half the
cores is a ceiling, not a mandatory worker count. Set build-system and compiler
parallelism explicitly, record the limits, and monitor memory and swap activity.
Reduce future concurrency or stop the task's own build if pressure appears;
never kill unrelated work. Do not retain an unconditional eight-worker cap.


## CUDA stream and async-pipeline invariants

For NVIDIA A100+ paths, graph safety also requires the same stream/dependency
semantics as eager execution:

- Launch on the caller/framework current stream unless the public API explicitly
  owns another stream.
- If an auxiliary stream is prepared, create its events before capture and encode
  every producer/consumer dependency explicitly. Host submission order is not a
  cross-stream dependency.
- Do not use `cudaDeviceSynchronize()` or host scalar reads to make capture
  deterministic.
- `cp.async`, TMA, mbarrier, WGMMA and TCGen05 pipelines must reach a valid
  completion/reuse point on every replay and every tail path; graph replay does
  not repair a missing device-side barrier.
- Prepared TMA descriptors, repacked execution layouts, scale tables, and tuning
  selections need explicit owners and invalidation keys. Retain them through all
  in-flight graph uses.
- Capture-safe allocation is not permission for unbounded per-replay allocation.
  Prefer stable execution-lane storage or graph-pool-owned buffers.
