# CUDA pipelines: sustain useful throughput and hide critical waits

## Goal and qualifications

For a bandwidth-bound QVQ path, the goal is to deliver useful packed operands
fast enough to keep downstream decoding and computation supplied. Minimize
avoidable traffic, conflicts and critical-path waits; overlap independent work
where the architecture permits it.

The universal objective remains latency/throughput at the required accuracy.
Large prefill GEMMs may be compute-bound; tiny decode grids may be dominated by
launches, dependency latency or insufficient parallelism. Maximizing raw DRAM
GB/s is not always the right target. See [Roofline](roofline.md).

A CUDA stream is an ordered queue of operations, not an executing thread.
Intra-kernel pipelines overlap stages across tiles, warps and hardware units.
The same tile still has data dependencies. Near-zero idle time for every lane,
warp or stream is neither necessary nor generally achievable.

## Primary implementation references

[CUTLASS pipelines](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html)
separate producer/consumer roles and manage stage ownership with synchronization.
Its [GEMM guide](https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/efficient_gemm.html)
describes shared-tile and register-fragment buffering.
[FlashAttention-3](flashattention-3.md) provides a research example of overlap
among specialized units; its schedule is architecture- and operator-specific.

## Proposed QVQ model

Across different tiles, aim to load tile t+2, decode tile t+1 and compute tile t,
subject to storage, dependencies and resource availability. This is a scheduling
model, not a claim that QVQ currently achieves three independent stages.

For independent resources and a sufficiently long steady-state pipeline, a
rough ideal tile interval is bounded below by the largest stage service time.
If stages contend for the same issue, load/store or shared-memory resources,
their demands must be combined; taking only the maximum understates the limit.
Startup, drain, tails, reductions and launch overhead remain outside that ideal.

| Potential limiter | Candidate investigation |
|---|---|
| Global memory delivery | Coalesced packed loads, reuse, adequate outstanding work |
| Dequantization | Exact redundant-work elimination, layout-aware direct decoding |
| Shared memory delivery | Bank mapping, fragment layout, unnecessary write-back |
| Matrix pipeline | Sufficient independent operands, appropriate tiles and accumulation |
| Synchronization | Correct ownership, balanced producers, late consumption waits |
| Small-grid / tail effects | Work partitioning, split overhead, realistic shape dispatch |

Do not remove required waits. Issue asynchronous work early enough to be useful,
wait before consumption, and delay buffer reuse until all consumers finish.
More stages or producer warps can improve overlap but reduce residency or consume
resources needed for arithmetic. Async submission alone does not prove overlap.

The scheduler can hide a waiting warp by issuing another. Optimize waits that
leave the limiting resource starved, rather than trying to make every warp's
stall counter zero. Avoid spinning merely to create apparent activity.

## QVQ application and measurement

The [P32 runtime](https://github.com/ModelCloud/QvQ/blob/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0/gptqmodel_ext/qvq/p32/qvq_p32_cuda.cu)
contains SM80 MMA and asynchronous-copy helpers. It does not establish that
every specialization overlaps perfectly. Keep the SM80 and Hopper/TMA paths
separate, as described in [CUDA execution](cuda-execution.md).

Compare matched stage counts, tiles and layouts using full-operator timing.
Include table initialization, split partial traffic/reduction and EoRA when
enabled. Confirm useful-byte traffic and issue/compute readiness; a kernel that
moves fewer bytes can become faster while reporting lower DRAM throughput.

Use [metric interpretation](cuda-metrics-and-performance.md), retain the
repository accuracy gates, and validate changed ordering against the installed
operator. No pipeline implementation or new performance result is claimed here.
