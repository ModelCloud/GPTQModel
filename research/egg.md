# egg: equality saturation for rewrite exploration

## Primary reference and finding

Max Willsey et al.,
[egg: Fast and Extensible Equality Saturation](https://arxiv.org/abs/2004.03082),
POPL 2021. [Author-maintained implementation](https://github.com/egraphs-good/egg).

Equality saturation represents many equivalent expressions in an e-graph
instead of committing immediately to one rewrite sequence. The paper introduces
rebuilding and e-class analyses to make this approach efficient and extensible.
Its demonstrated results concern the evaluated applications, not QVQ kernels.

## Proposed QVQ use and limits

An exploratory optimizer could search equivalent arrangements of transforms,
scale operations and correction branches, then extract a candidate using a cost
model. This is a research proposal; no egg integration in QVQ or standard XLA is
asserted here.

Only insert rewrites valid for the declared arithmetic. Real-number
distributivity does not establish FP16/FP32 equivalence, and scale cancellation
through quantization is generally invalid. Model rounding and exceptional values
or label an approximation and measure its error.

A cost based only on node count misses register pressure, fusion boundaries,
packed-memory traffic and occupancy. Validate extracted candidates on the
actual backend with [SASS](ssa-sass.md), [Nsight](nsight-profiling.md) and
held-out operator/model checks. Compiler search cannot replace recovery evidence.
