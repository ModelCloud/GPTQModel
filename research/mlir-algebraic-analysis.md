# MLIR, Polygeist, Polly and symbolic rewrite exploration

## What is available

[MLIR canonicalization](https://mlir.llvm.org/docs/Canonicalization/) applies
dialect-defined folds and rewrite patterns greedily, with iteration limits.
It is best-effort; canonicalization does not guarantee a unique global normal
form or a speed improvement.
The [pass catalog](https://mlir.llvm.org/docs/Passes/) includes CSE and
dialect-specific loop/affine transformations.

[Polygeist](https://github.com/llvm/Polygeist) raises C/C++ into MLIR and
documents optional CUDA backends and polyhedral infrastructure.
[Polygeist: Raising C to Polyhedral MLIR, PACT 2021](https://doi.org/10.1109/PACT52795.2021.00011)
describes its compilation approach. Generating CUDA code is not proof that
every existing QVQ CUDA construct can be imported.

[Polly](https://polly.llvm.org/) optimizes suitable LLVM loop regions using
polyhedral representations and dependence information. Applicability depends
on the region, effects and analyzable indexing, not simply the presence of loops.

[egg](https://github.com/egraphs-good/egg) and
[egglog](https://github.com/egraphs-good/egglog) offer equality-saturation
infrastructure; egglog combines e-graphs with Datalog-style reasoning.
See [the egg paper note](egg.md).
[SymPy CSE and rewriting](https://docs.sympy.org/latest/modules/rewriting.html)
offer a lighter symbolic route to identify repeated formulas. These systems
need appropriate semantics and extraction to become a CUDA optimization tool.

## QVQ recommendations (inference from these capabilities)

Use MLIR first where operations are already explicit in a graph, such as
transforms, scales and correction branches. For packed kernels, pilot
Polygeist on a small ordinary C/C++ helper or regular packing loop before
attempting an entire translation unit.

Regular strided loops and shared address calculations are plausible polyhedral
targets. Packed data-dependent lookups, inline assembly, warp cooperation and
asynchronous effects make the core kernel a harder target. Preserve dependence
and convergence semantics when changing loop order.

A native [StableHLO custom call](stablehlo.md) does not reveal its internals to
MLIR. Exposing a reference decomposition creates a semantic opportunity, not
automatic replacement by a faster native kernel.

Use SymPy to suggest symbolic factorization or CSE, then translate candidates
into the actual unsigned/FP operation model before acceptance. Exact integer
algebra over unbounded integers is not the same as fixed-width arithmetic.

For egg/egglog, define types, widths, rounding, effect boundaries and valid
rewrite preconditions. Model extraction cost using register pressure, loads,
conversion work and target instruction choices; a minimal expression tree
can have worse GPU scheduling. Cap search resources and retain alternatives
for measurement.

## Deployment boundary

Polygeist/LLVM versions and optional dependencies must be compatible; the
[reviewed source snapshot](cuda-static-analysis-tools.md#source-and-license-snapshot)
is not a tested build recipe. Start in an isolated research environment.
No Polygeist, Polly, egglog or SymPy pipeline was run on QVQ here.

Use [LLVM analysis](llvm-cuda-analysis.md) for the direct CUDA inspection path
and [Alive2/Z3](rewrite-verification.md) for supported local verification.
Do not add a graph rewrite to production solely because it is valid over real
numbers or reduces node count.
