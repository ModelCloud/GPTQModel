# A41/R0 H100 implementation and optimization audit

This audit closes the planned A41/R0 implementation sequence and the measured
H100 optimization sequence through Phase 67. The accepted production source
is Phase 67. Phases 57 through 62, 65, and 66 were exact experiments that did
not meet the promotion gate; all candidate CUDA source from those phases was
removed.

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

Phases 6 through 67 progressively addressed split-K occupancy, grouped QKV
scheduling, exact multiblock transforms, packed FP16 transform arithmetic,
launch/materialization boundaries, decoder scheduling, and the W2.5 N128
dual-consumer path. The accepted decoder specialization uses three live decode
fragments for the H100 W2.5 N128 gate/up kernel. Phase 63 additionally fuses
paired recovery-high with exact SiLU/down preconditioning, removing one launch
and both recovered gate/up tensors. Phase 64 additionally shares the first
four exact recovery-high tree stages between output tiles that differ only in
the final Hadamard bit. The paired kernel is selected only at M16, where the
row grid has enough parallelism to make the reduced instruction count win.
Phase 67 additionally proves once per paired recovery tree that all internal
FP16 results are finite, then omits redundant per-node overflow tests. The
original overflow-preserving tree remains the exact fallback. This is also
M16-only: its same-binary complete-MLP result wins all four rates with a
1.00168x geometric speedup, while M1--M8 retain Phase 64.

The formal complete Llama 3.2 1B MLP benchmark covers

\[
M\in\{1,2,4,8,16\},\qquad
(K,N)_{gate/up}=(2048,8192),\qquad
(K,N)_{down}=(8192,2048).
\]

Across W2, W2.5, W3, and W3.5, Phase 67 measures **44.310--48.465 us**.
The all-rate geometric comparison is **1.0862x versus Machete W4** and
**0.6643x versus Marlin W4**. The four changed M16 cells all win versus the
same-binary Phase-64 control. Relative to the roughly 96 us
Phase-5 endpoint, the
accepted H100 path is approximately **2.18x faster**, or roughly **54% lower
latency**. The complete table and exact timing protocol are in
[`qvq_a41_r0_phase67_h100_bounded_recovery_rounding.md`](qvq_a41_r0_phase67_h100_bounded_recovery_rounding.md).

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
| 62 | paired-recovery high-stage CTA widths 32/128/256 | Rejected; clean complete-MLP confirmation found no winner |
| 65 | adjacent-column `float2` recovery and packed-FP16 low transform | Rejected; 2.7--5.8% slower and 72 registers/thread |
| 66 | four-output recovery-tree sharing at an M16 128-block grid | Rejected; 12.3--20.5% slower despite zero spills |

The post-Phase-56 experiments show that the remaining shared-load conflicts
cannot be attacked profitably by adding a level-index-dependent consumer
address operation. The extra address dependency lies directly on the decode
critical path and costs more than the additional shared-memory wavefront.
Future work would need a producer-side remap with shift/add-only consumer
addressing, or a new representation-level design. That is a new optimization
project rather than an unfinished A41/R0 phase.

## Correctness and profiling closure

Phase 56 passed the H100-only completion suite:

```text
1554 passed, 75 skipped in 503.03 s
```

That suite includes the Torch oracle, Ampere grouping, grouped runtime,
transform runtime, and CUDA lifecycle/math coverage. Phase 67 additionally
passes 48 focused control/candidate exactness/graph cases, an explicit
high-magnitude fallback case, and the real Llama
3.2 layer/logits/cache lifecycle test at both M16 and decode row counts. Its
formal 20-cell benchmark also checks
exact equality with the unchanged plain QVQ result. Nsight Compute measures
3.84% fewer executed instructions in the changed M16 middle kernel. Skips are explicit
unsupported/multi-device or environment-dependent cases; only the physical
H100 was exposed to CUDA.

Performance timing used warmed CUDA Graph replay measured by CUDA events after
three spaced 0% utilization / 0 MiB admission samples. Instruction and stall
analysis used Nsight Compute 2026.2.1; graph-node timing used Nsight Systems
2026.4.1; emitted SASS and opcode counts were verified with `cuobjdump`.
Profiler reports remain outside Git under `/root/qvq-profiler-artifacts`.

Compilation was capped at four Ninja jobs, one NVCC host thread, and one CUDA
split-compile partition. No quantization math, checkpoint payload, persistent
VRAM, output ordering, or non-H100 dispatch was changed by the rejected
experiments.
