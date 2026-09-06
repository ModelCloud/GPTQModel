# XLA: optimization, lowering and tuning boundaries

## Primary documentation

[XLA architecture](https://openxla.org/xla/architecture) describes compilation
from StableHLO through internal HLO optimization into backend code. Optimizations
include common-subexpression elimination, fusion and buffer analysis; backends
choose target-specific operations and library calls. CPU/GPU code generation
uses LLVM. XLA and LLVM therefore occupy different stages of the same pipeline.

[Custom calls](https://openxla.org/xla/custom_call) connect compiled computations
to external implementations through registered handlers and explicit argument,
result and execution contracts.

[Persisted autotuning](https://openxla.org/xla/persisted_autotuning) documents
reuse of tuning results for supported generated fusions. Cache invalidation,
including separation across XLA versions, remains the user's responsibility.

## QVQ evidence

The [P32 ABI](https://github.com/ModelCloud/QvQ/blob/263ed4baf7be5e9547b4e731c9031bef5f48cf69/docs/kernels/qvq_p32_runtime_abi.md) allows native reduction or caller-visible split
partials. Its framework-neutral boundary assigns tuning policy to the caller;
ABI and kernel versions belong in transient tuning keys. The initial library
targets SM80. The document leaves broader graph integration and full-path
performance validation open.

## Proposed integration rules

A custom-call name or a list of tile/stage/split options does not make XLA
automatically benchmark those alternatives. A QVQ adapter must implement
candidate selection, validation and caching, or integrate with an explicit
compiler tuning mechanism. Do not imply standard XLA understands P32 packing.

Expose exact math where practical using [StableHLO](stablehlo.md).
A native packed product can remain opaque while a following reduction or
correction addition becomes visible. Compare the cost of extra partial-output
traffic with any fusion benefit. Whole-graph visibility does not guarantee a
globally optimal schedule.

Preserve scale application, casts, accumulator behavior and reduction ordering.
A fusion that changes rounding can change the residual that an
[EoRA](eora.md) fit was trained to compensate. Recheck the deployed operator,
then propagated model quality; successful compilation establishes neither.

Recommended tuning records include hardware, shapes/layouts, precision,
compiler/runtime versions, candidate identity and correctness evidence.
Keep compilation and tuning latency separate from warm executable timing.
Use [Nsight](nsight-profiling.md) to check whether a proposed optimization
removes an actual bottleneck, and [SSA/SASS](ssa-sass.md) to inspect lowering.
These are proposed QVQ integration practices, not claimed implemented features.
