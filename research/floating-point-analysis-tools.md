# Open-source floating-point analysis and rewrite tools

## Primary findings

[Daisy](https://github.com/malyzajko/daisy) combines numerical error analysis
and optimization. Its reviewed README includes a Scala DSL frontend and a
Tree-sitter option for Scala, C and FPCore. A C frontend does not establish
support for arbitrary CUDA templates, inline PTX or concurrent execution.

[FPTaylor](https://github.com/soarlab/FPTaylor) estimates roundoff error using
symbolic Taylor methods and global optimization. The project documents
FPBench interchange and a separate HOL Light verification path.
[The original paper](https://soarlab.org/papers/2015_fm_sjrg.pdf) describes the
method and evaluated scope. Do not describe every ordinary run as a
machine-checked certificate.

[Herbie](https://herbie.uwplse.org/) searches for floating-point expressions
with improved accuracy and supports cost-aware exploration. It proposes
candidates rather than proving that every input improves.
[Combining Tools for Optimization and Analysis of Floating-Point
Computations](https://arxiv.org/abs/1805.02436), FM 2018, studies using Daisy
to check Herbie's proposed rewrites. This supports separating search from
worst-case validation.

[FPCore/FPBench](https://github.com/FPBench/FPBench) supplies a shared numerical
benchmark representation, not a CUDA runtime or automatic kernel extractor.

## Proposed QVQ deployment

Use these tools on extracted scalar scale formulas, normalization subexpressions,
correction additions or short reductions. Preserve the exact cast and rounding
sequence and declare input ranges. Begin with one fragment whose mathematical
specification and CUDA implementation can be checked independently.

The tools' scalar arithmetic models should not be assumed to represent FP4
encoding, stochastic rounding, Tensor Core internal accumulation or GPU
transcendentals. Supply a justified model or mark the operation unsupported.
Model overflow, underflow, subnormals, flush-to-zero, FMA contraction and
exceptional values relevant to the deployed path.

Ranges measured on calibration data define an empirical envelope, not a proof
that all future activations remain inside it. State whether a runtime check,
mathematical invariant or experiment-specific assumption enforces each bound.

## Interpret the result correctly

An error bound against an exact-real function is different from candidate drift
against the deployed reference kernel. If two implementations approximate the
same real function with bounds E1 and E2 over the same domain, their difference
is at most E1 + E2 by the triangle inequality; this may be too loose to be useful.
A direct relational analysis can be preferable when available.

Herbie may improve real-function accuracy while increasing drift from an
existing reference or changing the residual fitted by [EoRA](eora.md).
Evaluate the claimed objective explicitly.

Do not use a scalar bound to certify a full dot product or full-model output.
Keep the [repository accuracy rules](../AGENTS.md), matched operator checks,
held-out model evidence and [performance measurements](nsight-profiling.md)
separate. This note documents candidates; it does not report a QVQ numerical
analysis run.
