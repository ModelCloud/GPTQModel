# Open-source static analysis for QVQ CUDA

Reviewed 2026-09-06 against QVQ main
[`66565c2`](https://github.com/ModelCloud/QvQ/tree/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0).
This is a source-backed deployment assessment. None of these analyzers was
installed or run on QVQ for this note.

## Recommendation

Start with **Clang device LLVM IR plus LLVM analysis/optimization reports**.
Add **Z3 for exact packed-bit identities** and **Alive2 for supported LLVM
rewrite validation**. These address QVQ's repeated decoding and address work
most directly. Use **Daisy/FPTaylor** for scoped floating-point error bounds;
**Herbie, SymPy and egg/egglog** can propose rewrites that still need checking.

There is no verified drop-in tool in this survey that reads arbitrary QVQ CUDA,
understands every inline PTX/Tensor Core/barrier operation, proves all algebraic
optimizations and predicts their final SASS performance.

## Capability and deployment map

Priority is a QVQ engineering recommendation, not a vendor support claim.

| Tool | Input and useful output | QVQ fit / boundary | Priority |
|---|---|---|---|
| [Clang + LLVM](llvm-cuda-analysis.md) | CUDA to SSA IR; value, alias, loop and redundant-work analysis | Inspect visible device arithmetic; does not analyze inline PTX instruction bodies as ordinary IR | First |
| [Alive2](rewrite-verification.md) | LLVM before/after refinement checks | Extract supported functions; unsupported intrinsics, concurrency and bounded loops limit coverage | First, scoped |
| [Z3](rewrite-verification.md) | Explicit bit-vector / IEEE FP formulas; proof query or counterexample | Exact masks, shifts, bank selection and index identities; requires a faithful model | First, scoped |
| [MLIR](mlir-algebraic-analysis.md) | Dialect IR; CSE, canonicalization and structural transforms | Natural for graph-side QVQ operations; not a CUDA source parser | Second |
| [Polygeist](mlir-algebraic-analysis.md) | C/C++ frontend to MLIR; polyhedral/parallel transformations | Pilot selected helpers; full hand-written PTX-heavy kernel coverage unverified | Experimental |
| [Polly](mlir-algebraic-analysis.md) | LLVM loop regions and dependence/schedule transformations | Regular packing/address loops are a better target than irregular decoding and barriers | Experimental |
| [egg / egglog](mlir-algebraic-analysis.md) | Custom expression language and rewrite rules | Candidate search needs QVQ semantics, extraction and a target cost model | Experimental |
| [SymPy](mlir-algebraic-analysis.md) | Symbolic expressions; CSE and algebraic rewriting | Cheap formula exploration; real/symbolic identities are not machine-arithmetic proofs | Optional |
| [Daisy](floating-point-analysis-tools.md) | Numerical program subsets and input domains; error analysis | Extract scalar formulas; current frontend options do not establish CUDA support | Second, scoped |
| [FPTaylor](floating-point-analysis-tools.md) | FP expressions and ranges; rigorous roundoff bounds | Useful for bounded numerical fragments, not entire matrix kernels | Second, scoped |
| [Herbie](floating-point-analysis-tools.md) | FP expressions; candidate accuracy/cost improvements | Search aid; validate worst-case error and GPU lowering independently | Optional |
| [Souper](rewrite-verification.md) | LLVM-derived integer expressions; synthesized peephole rewrites | Relevant to packed decode, but archived upstream and toolchain-bound | Research only |
| [GPUVerify](gpuverify.md) | Supported CUDA/OpenCL kernels; race/barrier-divergence checking | Different problem from math folding; modern PTX/TMA coverage unverified | Isolated experiment |

## What the repository actually exposes

The audited [P32 source](https://github.com/ModelCloud/QvQ/blob/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0/gptqmodel_ext/qvq/p32/qvq_p32_cuda.cu) contains pure integer helpers
`pgc16_mix` and `selected_bank_mask`, paired extraction in
`window_state_pair64`, and inline PTX for `ldmatrix`, `mma.sync`
and `cp.async`. These need different analysis boundaries.

The [Bazel runtime build](https://github.com/ModelCloud/QvQ/blob/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0/gptqmodel_ext/qvq/p32/BUILD.bazel) invokes NVCC with C++17, O3 and
an SM80 target. Therefore a Clang analysis compile is a parallel inspection
path, not a dump of the production NVCC optimizer. Its findings must be
confirmed in the actual built binary. This statement concerns that target,
not every QVQ backend or architecture.

Proposed first units: verify a bank-mask identity, analyze repeated paired-window
indices under the real caller ranges, then inspect duplicated scale/conversion
work around the opaque matrix operations. A helper proof does not prove memory
safety, synchronization or the enclosing kernel.

## Deployment order and acceptance

1. Pin a compatible Clang/LLVM/CUDA toolchain and emit device IR for one existing
   specialization. Record parsing failures and opaque operations as coverage gaps.
2. Run standard analysis and simplification reports without changing production
   code. Start with [the LLVM recipe](llvm-cuda-analysis.md).
3. Extract one proposed integer rewrite into a proof case. Save assumptions,
   widths, source/candidate revisions and solver outcome.
4. For a floating-point change, specify the rounding model and input domain,
   then obtain appropriate error evidence. Exact bit equality and bounded
   approximation are different outcomes.
5. Implement only independently justified candidates. Apply [AGENTS.md](../AGENTS.md),
   inspect source-correlated [SASS](ssa-sass.md), profile with
   [Nsight](nsight-profiling.md), and measure the complete operator.

Run analysis jobs on CPU; GPU hardware is needed later for actual kernel parity
and timing. Record pass, counterexample, unsupported and timeout separately.
A timeout or an erased unsupported operation is never a successful proof.

## Source and license snapshot

The links pin upstream default-branch heads observed during this survey.
They are source-review identifiers, **not a tested compatible bundle**.
License labels describe the project sources; dependency licenses remain separate.

| Project | Reviewed head | Root license / status |
|---|---|---|
| LLVM / MLIR / Polly | [5840db809d3f](https://github.com/llvm/llvm-project/tree/5840db809d3ff2dc194ccc43bd0be2f36c629185) | [Apache-2.0 with LLVM exceptions](https://github.com/llvm/llvm-project/blob/5840db809d3ff2dc194ccc43bd0be2f36c629185/llvm/LICENSE.TXT) |
| Polygeist | [77c04bb2a7a2](https://github.com/llvm/Polygeist/tree/77c04bb2a7a2406ca9480bcc9e729b07d2c8d077) | [Apache-2.0 with LLVM exceptions](https://github.com/llvm/Polygeist/blob/77c04bb2a7a2406ca9480bcc9e729b07d2c8d077/LICENSE) |
| Alive2 | [c5773898e4e4](https://github.com/AliveToolkit/alive2/tree/c5773898e4e40097b377fe9200ccb82447f4289d) | MIT |
| Z3 | [c2c198d553a6](https://github.com/Z3Prover/z3/tree/c2c198d553a6599abd6160b5a1fb9944d442a929) | [MIT](https://github.com/Z3Prover/z3/blob/c2c198d553a6599abd6160b5a1fb9944d442a929/LICENSE.txt) |
| egg | [2f31b28e3f9d](https://github.com/egraphs-good/egg/tree/2f31b28e3f9d78e02273b6c6d4201b5b0720b343) | MIT option |
| egglog | [90635860397c](https://github.com/egraphs-good/egglog/tree/90635860397ce710f8c0a4eeb04154a8ebc3ac05) | MIT |
| Herbie | [52cba77bdd00](https://github.com/herbie-fp/herbie/tree/52cba77bdd002d6a71feecc2e57631c8d462b4b2) | [MIT](https://github.com/herbie-fp/herbie/blob/52cba77bdd002d6a71feecc2e57631c8d462b4b2/LICENSE.md) |
| Daisy | [6a6f47abdd23](https://github.com/malyzajko/daisy/tree/6a6f47abdd231d3009444e9e1b8ce9febc28c27e) | Apache-2.0 |
| FPTaylor | [b5a77cae3484](https://github.com/soarlab/FPTaylor/tree/b5a77cae348400f21f83512210d9f43c4bffb381) | MIT; dependency/build pilot needed |
| Souper | [963d4df436f3](https://github.com/google/souper/tree/963d4df436f3dc0b039cc0e47ada0577a26f5c4e) | Apache-2.0; Google repository archived |
| GPUVerify | [49219770aad0](https://github.com/mc-imperial/gpuverify/tree/49219770aad01231edd0d8e0fa3ed036006cf32a) | [Ms-PL](https://github.com/mc-imperial/gpuverify/blob/49219770aad01231edd0d8e0fa3ed036006cf32a/LICENSE.TXT); historical toolchain |
| SymPy | [License](https://github.com/sympy/sympy/blob/master/LICENSE) | BSD-3-Clause; documentation review, no revision-pinned build |

NVIDIA's NVCC/ptxas, binary utilities, Nsight and Compute Sanitizer are useful
complements, but are not the open-source algebraic analyzers being recommended.
Do not substitute profiling or dynamic sanitizer runs for static proofs.
