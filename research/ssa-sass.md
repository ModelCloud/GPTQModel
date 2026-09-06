# SSA and SASS: algebraic simplification versus emitted instructions

## Terminology and primary documentation

**SSA** means static single assignment: each IR value has one definition.
It supports dataflow reasoning. **SASS** is NVIDIA device machine assembly;
**PTX** is a virtual instruction set lowered to target machine code.
These are different representations, not alternate spellings.

[LLVM's language reference](https://llvm.org/docs/LangRef.html) defines its
SSA IR and floating-point flags. In particular, reassociation and contraction
are distinct permissions; real-number identities are not unconditional
floating-point rewrite rules.
[NVIDIA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
documents `cuobjdump` and `nvdisasm` for inspecting device binaries and SASS.
The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
defines virtual instructions and their target requirements.

## QVQ analysis procedure (proposed)

| Question | Appropriate evidence |
|---|---|
| Did repeated pure expressions merge? | Optimized graph/SSA IR and def-use relationships |
| Did constants or inverse transforms fold away? | Before/after IR with numeric preconditions |
| Did redundant loads survive? | Alias/effect analysis, SASS and measured memory traffic |
| Did fewer expressions reduce cost? | Registers, spills, dynamic instructions and timing |
| Did the rewrite preserve recovery behavior? | Matched operator parity and held-out model evaluation |

Common-subexpression elimination merges equivalent computations.
Constant folding evaluates compile-time expressions.
Algebraic rewriting changes expression structure under proven conditions.
Fusion changes execution grouping and data movement; it need not reduce math.

For QVQ, examine repeated activation transforms, scale loads, address arithmetic
and correction additions. Equal-looking loads cannot be merged across a possible
write without alias/effect evidence. Stochastic or stateful operations are not
pure expressions. Reusing a value can extend its live range and increase
register pressure, so fewer operations need not mean faster code.

Do not cancel scales across clipping, quantization, rounding or nonlinearities.
Likewise, replacing separate multiply/add with FMA changes rounding. Treat an
allowed approximation as a measured numerical change, not exact deduplication.

Inspect the actual loaded binary with its build flags, target SM and hash.
Static instruction count is not dynamic work or elapsed time. Correlate SASS
with [Nsight](nsight-profiling.md) before attributing a speedup to folding.

## Research and repository evidence

Read [Cytron et al. on SSA](ssa-paper.md) and [egg equality saturation](egg.md).
Neither establishes that QVQ or XLA uses egg.

The [P32 ABI audit](https://github.com/ModelCloud/QvQ/blob/263ed4baf7be5e9547b4e731c9031bef5f48cf69/docs/kernels/qvq_p32_runtime_abi.md) records a source-correlated SASS/addressing
investigation and explicitly avoids interpreting a tiny profiled grid as a
performance win. Follow that separation of correctness and timing evidence.
