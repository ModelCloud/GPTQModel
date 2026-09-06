# Exact rewrite checking: Alive2, Z3 and Souper

## Tool and paper findings

[Alive2](https://github.com/AliveToolkit/alive2) validates LLVM transformations
using refinement checking and SMT. It supplies `alive-tv` and compiler
integration tools. Its README warns that interprocedural transformations are
unsupported and that the LLVM integration tracks a compatible upstream build.
[Alive2: Bounded Translation Validation for LLVM, PLDI 2021](https://web.ist.utl.pt/nuno.lopes/pubs/alive2-pldi21.pdf)
explains bounded validation; finite loop exploration is not an arbitrary-trip
count proof.

[Z3 bit-vectors](https://microsoft.github.io/z3guide/docs/theories/Bitvectors/)
model fixed-width operations with explicit signed/unsigned operator choices.
Its [IEEE FP theory](https://microsoft.github.io/z3guide/docs/theories/IEEE%20Floats/)
can express floating-point operations and rounding modes. Z3 is a solver,
not a CUDA parser; QVQ must supply the correspondence to source semantics.

[Souper](https://github.com/google/souper) searches for missing LLVM peephole
optimizations using SMT. Its
[paper](https://arxiv.org/abs/1711.04422) describes synthesis for integer
expressions. The Google repository is archived, observed 2026-09-06.
Treat it as a pinned research environment, not a new mandatory compiler dependency.

## Proposed QVQ proof targets

The [audited P32 code](https://github.com/ModelCloud/QvQ/blob/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0/gptqmodel_ext/qvq/p32/qvq_p32_cuda.cu) has useful bounded targets:

| Target | Required model and assumptions |
|---|---|
| `selected_bank_mask` | Unsigned word width, allowed bank-bit positions, mask identity |
| `pgc16_mix` | Exact XOR/shift/multiply/add/truncation behavior |
| `window_state_pair64` | Actual caller pair domain, transition rate, circular word layout, funnel-shift semantics |
| Shared address subexpressions | Signedness, range, overflow and equivalence of resulting addresses |

For example, a bit `b` restricted to 0 or 1 permits comparing unsigned
`(0u - b) & mask` with `b * mask`. A solver can check whether inequality
is satisfiable under that domain. This illustrates a proof obligation; it is
not a newly verified QVQ optimization or evidence that either spelling is faster.

Model C/C++ undefined behavior and LLVM poison correctly. SMT shifts and
modular arithmetic are total operations; C++ oversized shifts and signed
overflow cannot be silently assigned those same semantics. A source-to-target
refinement result also differs from bidirectional bit equality.

## Deployment and result handling

Extract a small source and candidate function, preserving types and definedness.
For supported LLVM fragments, run `alive-tv source.ll candidate.ll` using
the pinned build's documented interface. For custom bit-vector models, ask for
a counterexample to the intended identity and retain the complete query.

Record outcome as proved within the stated model/bounds, counterexample,
unsupported, or timeout/unknown. Check model extraction independently. Do not
turn an unsupported MMA or PTX operation into an arbitrary pure function and
then claim the kernel was verified.

Validate FP-to-integer bitcasts, NaNs and signed zero using the intended equality
notion. Neither a real-arithmetic identity nor an SMT floating-point abstraction
automatically captures Tensor Core intermediate accumulation or flush-to-zero.

Use Souper or [egg](egg.md) to generate candidates only after the simple proof
path is working. Rank candidates using the actual target's emitted instructions
and measurements, not expression count alone. Keep proof cases and solver
versions beside experiment evidence; see [the deployment map](cuda-static-analysis-tools.md).
