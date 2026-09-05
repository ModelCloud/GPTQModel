---
name: qvq-kernel-accuracy
description: Preserve QVQ numerical contracts when optimizing linear kernels, decode/reconstruction, reductions, fusion, compiler math, backend dispatch, or MKNE autotuning on CPU, CUDA, ROCm, and Metal. Use before selecting performance transformations or changing precision.
---

# QVQ kernel accuracy

Optimize the implementation of the accepted mathematics first. A speedup is eligible for production only when it
preserves the applicable numerical and runtime contracts. Start with redundant math, repeated decode work, and data
movement; reduced precision is a separate numerical tradeoff, not the default way to make a kernel faster.

## Lock the operation and accuracy contract

Before changing code or selecting an autotune winner, record:

- The retained baseline revision, canonical reference, full operator equation, and boundaries being optimized.
  Include decoding, bank selection, scales, rotations/Hadamards, correction terms, bias, and epilogue as applicable.
- Actual M/N/K and expert/group dimensions, strides, padding, rates/layouts, and dispatch conditions. Spell out
  dimension ordering; a geometry string without axis names is ambiguous. Use E only where the operation has experts.
- Input, decoded-weight, product/compute, accumulator, split-reduction, scratch, epilogue, and output dtypes;
  conversion locations, reduction order, rounding modes, fast-math flags, and denormal/overflow behavior.
- Each metric's formula, normalization, aggregation axes, and acceptance threshold before inspecting candidate results.

Preserve the existing [QVQ contract](../gptqmodel-cuda-kernels/SKILL.md#qvq-accuracy-contract):

- Quantization implementation optimizations must preserve deterministic state/path selection, packed words, bank IDs,
  and metadata exactly. Floating-output tolerance cannot excuse a changed quantization decision.
- QVQ inference requires **maximum absolute output drift <= 2e-3 in every tested case**, against the canonical
  reference for identical packed weights, inputs, and operator semantics. Apply any additional path-specific gates.
  This is an absolute threshold, not 0.2%, relative L2, mean error, or an allclose rtol.
- Do not widen the threshold to make a faster candidate pass, normalize/clip away a failure, modify the oracle to
  mimic the candidate, or hide a failing shape in an average. Revising the contract is a separately scoped task;
  preserve the current baseline and defaults while presenting the evidence and proposed change.
- This implementation-preservation contract does not prohibit separately requested quantization-algorithm research.
  Follow the repository's promotion/escalation policy for those experiments; model-quality proxy tradeoffs do not
  waive an implementation correctness failure.

Read the nearest implementation and applicable tests; instructions are not evidence that a test actually ran.

## Prioritize accuracy-preserving mathematics

Write a short before/after math argument and expected instruction/memory savings for the proposed change. Work
through promising opportunities in the measured hot path before proposing lower precision; an exhaustive search of
every possible configuration is not required.

| Opportunity | What must remain true |
|---|---|
| Decode a packed field once and reuse it | Preserve integer widths, signedness, masks/shifts, bank selection, and every consumer's decoded value. |
| Eliminate repeated address, scale, or conversion work | Reuse the same value at the same numerical boundary; respect mutations, lifetimes, and synchronization. |
| Reuse decoded weights or activations across consumers | Preserve each consumer's layout, scale, output partition, and accumulation contract. |
| Hoist invariants or precompute immutable transforms | Account for storage cost and invalidate on every relevant weight/metadata mutation. |
| Change tiling, vectorization, pipeline, or layout | Cover each logical term exactly once, preserve tails, and classify any changed reduction order. |
| Fuse compatible stages or remove intermediates | Preserve dependencies and intended rounding points; removing a cast or moving a scale can change results. |
| Factor or fold a correction into a GEMM | Prove the real-arithmetic identity and independently validate floating-point rounding and conditioning. |

Classify the transformation explicitly:

1. **Exact representation/value reuse:** prove the integer or already-rounded values are unchanged.
2. **Equivalent real-number algebra with changed floating-point evaluation:** reassociation, FMA contraction,
   split-K, scale movement, or folding can change rounding, cancellation, and range. Validate rather than claiming
   bitwise equivalence. Split-K with FP32 partials is not inherently inaccurate, nor is it automatically identical.
3. **Approximate arithmetic:** narrower inputs/accumulators/partials/output, TF32 or other reduced-mantissa modes,
   approximate reciprocals/activations, or dropped terms change the numerical strategy. Explain why the measured
   precision-preserving options are insufficient and evaluate this as a distinct candidate.

For example, XW + (XA)B and X(W + AB) are equal over real numbers, but forming and storing W + AB introduces different
rounding and can lose a small correction. A lower operation count is not a floating-point proof.

FP32 output alone does not establish FP32 computation or exactness. Inspect library compute modes and generated
instructions. A final FP32 cast cannot recover precision discarded earlier.

## Keep independent references

Use the existing eager/canonical QVQ implementation on the **same reconstructed quantized weights** to isolate
kernel error from quantization error. A comparison against the original unquantized weights answers another question.
For ambiguity, independently reconstruct the values and use FP64 on bounded rows/columns while retaining the full K
reduction; keep the production eager comparison as well. Record the oracle's own precision and backend settings.

To investigate output narrowing, compare the candidate with both the original reference and the reference rounded
at the intended output boundary. This diagnoses an unavoidable format effect; it does not relax the locked gate.
Test full preprocessing/GEMM/correction/recovery/epilogue composition when any boundary changes, not only inner GEMM.

## Validate before ranking speed

- Run focused exact/logical and floating-output checks, including long reductions, cancellation, outliers, rounding
  boundaries, legal tails/layouts/rates, and relevant repeated/stream/graph paths. Verify the public dispatch path.
- Report max-absolute, mean-absolute, relative-L2, finite status, and token/channel tails. Use safe metric accumulation
  precision; report zero/near-zero reference norms rather than presenting unstable relative ratios as evidence.
- Use synthetic fixtures for algebra/kernel corner cases only. Model-quality conclusions require real weights and
  real tokenized activations held out from calibration and tuning; never tune on the benchmark test set.
- When changing arithmetic, precision, fusion boundaries, or broad dispatch, replay representative real modules,
  affected blocks, and the combined model path. Keep non-target operators/checkpoint tensors fixed. Measure
  teacher-forced logits/KL, top-token agreement/margins, and paired task effects as the declared scope requires.
- Retain an accurate fallback for failing or unvalidated cases. A shape-specific pass licenses only that tested scope.
  Full-model quality cannot waive the locked kernel gate; a kernel pass alone does not certify full-model quality.

For propagation diagnosis or a proposed error-budget change, read
[references/measure-propagation.md](references/measure-propagation.md). Do not invent model sensitivity constants or
predict benchmark accuracy from a kernel tolerance.

For hardware runs and generated-code evidence, also follow the applicable GPU-testing, CUDA/AMD/Metal, and profiling
skills routed by AGENTS.md. Reuse their hardware/lease and per-device-code-commit audit procedures.

## Make autotuning correctness-constrained

Filter candidates by supported semantics and the declared accuracy gates before ranking accepted candidates by
warmed latency. Record failed candidates and failing cases; fastest-but-inaccurate is a rejected candidate.
An autotune cache must distinguish the relevant operation/precision contract, hardware, shape/layout/rate, and
implementation/build identity so an accurate selection cannot silently reuse an incompatible result.

Apply this rule whether selection occurs inside a kernel wrapper or in a compiler such as XLA/StableHLO. Exposing a
tunable knob to a compiler does not authorize that knob to weaken numerical semantics.

## Report the retained result

Record the math change/classification, unchanged or explicitly proposed numerical contract, exact revisions and
configuration, per-case accuracy and timing, generated-instruction evidence, measured scope, fallbacks, and limitations.
Separate observed results from hypotheses or estimates. Preserve the baseline and rejected-candidate evidence.
Do not describe an unrun GPU/model check as passing or an instruction-count reduction as an end-to-end speedup.

Repository starting points (inspect current code and supported arguments before running):

- `tests/test_qvq_p32_amd.py`: canonical FP32, folded/recovery, dtype, dispatch, and stream checks.
- `scripts/benchmark_qvq_p32_amd.py` and `scripts/benchmark_qvq_p32_qwen38_27b_amd.py`: AMD shape/rate sweeps.
- `tests/test_qvq_grouped_oracle.py`, `tests/test_qvq_unified_harness.py`: grouped and integrated QVQ checks.
- `verification/README.md`: serving-engine distribution and task evaluation. Its native-versus-quantized comparison
  does not by itself isolate two kernels on identical packed weights; use matched models/backends for that question.
