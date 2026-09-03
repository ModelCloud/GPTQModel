# A41/R0 H100 implementation and optimization audit

This audit closes the planned A41/R0 implementation sequence and the measured
H100 optimization sequence through Phase 61. The accepted production source
remains Phase 56. Phases 57 through 61 were exact experiments that did not meet
the promotion gate; all candidate CUDA source was removed.

## Planned implementation phases

| Phase | Contract | Result |
|:--|:--|:--|
| 1 | Pure Torch grouped-P32 semantics, lossless payload grouping, and R0 legality/fallback | Complete |
| 2 | Exact grouped P32 realization for Ampere | Complete |
| 3 | Segmented grouped P32 realization for Hopper | Complete |
| 4 | Production model/runtime integration with canonical checkpoint storage and fail-closed lifecycle | Complete |
| 5 | Exact grouped MLP recovery and down-input pipeline | Complete |

Torch defines the operator. Ampere and Hopper implement the same operator
without changing the canonical P32 checkpoint. Runtime grouping shares only a
bit-identical input transform; every child keeps its own payload, alternative
bank, output width, split schedule, reduction order, output transform, scale,
and bias.

## H100 production endpoint

Phases 6 through 56 progressively addressed split-K occupancy, grouped QKV
scheduling, exact multiblock transforms, packed FP16 transform arithmetic,
launch/materialization boundaries, decoder scheduling, and the W2.5 N128
dual-consumer path. The final accepted specialization uses three live decode
fragments for the H100 W2.5 N128 gate/up kernel.

The formal complete Llama 3.2 1B MLP benchmark covers

\[
M\in\{1,2,4,8,16\},\qquad
(K,N)_{gate/up}=(2048,8192),\qquad
(K,N)_{down}=(8192,2048).
\]

Across W2, W2.5, W3, and W3.5, Phase 56 measures **45.955--50.118 us**.
The all-rate geometric comparison is **1.0510x versus Machete W4** and
**0.6411x versus Marlin W4**. Relative to the roughly 96 us Phase-5 endpoint,
the accepted H100 path is approximately **2.12x faster**, or **52.7% lower
latency**. The complete 20-cell M/K/N table and exact timing protocol are in
[`qvq_a41_r0_phase56_h100_w25_n128_depth3.md`](qvq_a41_r0_phase56_h100_w25_n128_depth3.md).

These are latency ratios, not equal-bit-rate comparisons: Marlin and Machete
run W4 and are figurative kernel baselines for the W2--W3.5 QVQ paths.

## Closed post-production experiments

| Phase | Experiment | Outcome |
|--:|:--|:--|
| 57 | N128 lane-private and broader-rate specializations | Rejected; no robust end-to-end improvement |
| 58 | 16-bit PGC multiply/add representation | Rejected; all 20 complete-MLP cells regressed |
| 59 | global level table, W3.5 N128, pair packing, and spare-fragment prefetch | Rejected; either inexact, unchanged SASS, or slower |
| 60 | padded and dynamically hashed shared level rows | Rejected; 11.9% and 21.1% slower geometrically |
| 61 | minimally padded 17-word shared level rows | Rejected; complete W2.5 MLP roughly 10% slower |

The post-Phase-56 experiments show that the remaining shared-load conflicts
cannot be attacked profitably by adding a level-index-dependent consumer
address operation. The extra address dependency lies directly on the decode
critical path and costs more than the additional shared-memory wavefront.
Future work would need a producer-side remap with shift/add-only consumer
addressing, or a new representation-level design. That is a new optimization
project rather than an unfinished A41/R0 phase.

## Correctness and profiling closure

The final production source passed the H100-only completion suite:

```text
1554 passed, 75 skipped in 503.03 s
```

The suite includes the Torch oracle, Ampere grouping, grouped runtime,
transform runtime, and CUDA lifecycle/math coverage. The grouped Hopper suite
also passed all 69 cases. Skips are explicit unsupported/multi-device or
environment-dependent cases; only the physical H100 was exposed to CUDA.

Performance timing used warmed CUDA Graph replay measured by CUDA events after
three spaced 0% utilization / 0 MiB admission samples. Instruction and stall
analysis used Nsight Compute 2026.2.1; graph-node timing used Nsight Systems
2026.4.1; emitted SASS and opcode counts were verified with `cuobjdump`.
Profiler reports remain outside Git under `/root/qvq-profiler-artifacts`.

Compilation was capped at four Ninja jobs, one NVCC host thread, and one CUDA
split-compile partition. No quantization math, checkpoint payload, persistent
VRAM, output ordering, or non-H100 dispatch was changed by the rejected
experiments.
