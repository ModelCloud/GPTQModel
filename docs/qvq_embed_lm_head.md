# Tied QVQ embedding and LM-head design

## Status

This document specifies the intended design for quantizing one matrix shared by an input embedding and an LM head,
as in Llama 3.2. It is a design and validation contract, not a claim that the lifecycle or kernels already support
this path.

The requested name `V2B2-G32` refers here to the existing serialized codec `QVQ_V2B2_P32`: one canonical V2 bank,
one module-selected complementary family, and one binary bank selector for every 32 weights. The implementation and
checkpoint name should remain `V2B2-P32`; introducing a second alias for the same layout would create avoidable format
ambiguity.

## Decision

Quantize the tied endpoint as one logical parameter and one serialized QVQ artifact. Do not untie the embedding and
LM head, do not train independent selector maps, and do not save two packed payloads.

The shared artifact has two runtime views:

```text
token IDs -> QVQ embedding gather --+
                                    +-- one reconstructed matrix Q [vocabulary, hidden]
hidden states -> QVQ LM-head GEMV --+
```

Both views must reconstruct exactly the same `Q`. If model parallelism places the two call sites on different devices,
the immutable packed payload may be cached once per device, but those copies must have the same content hash and still
represent one checkpoint object.

The quantization objective must include both uses of the parameter. Candidate generation uses a scalable tied-gradient
YAQA proxy; authoritative selection uses teacher-forced final-logit replay on disjoint data. An independently encoded,
all-bank-zero V2 artifact is the atomic fallback.

## Why one shared artifact is required

Let the dense tied weight be

\[
W \in \mathbb{R}^{V \times D},
\]

where `V` is vocabulary size and `D` is hidden width. The two uses are

\[
\operatorname{embed}(t)=W_{t,:}, \qquad
\operatorname{head}(h)=Wh.
\]

If the two modules are quantized independently, they become matrices `Q_embed` and `Q_head`. That changes the model
class, doubles the endpoint payload, and removes the regularization implied by tying. A valid tied implementation has

\[
Q_{\text{embed}} = Q_{\text{head}} = Q.
\]

For Llama 3.2 1B, `V=128256` and `D=2048`, so the shared matrix contains 262,668,288 weights. V2B2-P32 adds one
selector bit per 32 weights:

\[
R_{\text{effective}}=R+\frac{1}{32}=R+0.03125\ \text{bpw}.
\]

The selectors occupy about 0.979 MiB. At W2, the planar payload plus selectors is about 63.604 MiB, excluding scales
and small metadata. Untying would require about 127.207 MiB and would no longer be the same tied model.

## V2B2-P32 codec contract

The tied endpoint reuses the existing segmented V2 codec without changing its bitstream:

- 16-bit V2 trellis state and two reconstructed weights per transition;
- bank 0 is the exact canonical V2 codebook;
- one alternative family is selected from family IDs 1, 2, and 3 for the complete shared artifact;
- a binary selector chooses canonical or alternative bank every 32 weights;
- eight selectors cover each canonical 16-by-16 QVQ tile and pack into one byte;
- selector boundaries do not reset the V2 state;
- W1 through W3.5 use the existing rate-specific frozen bank definitions.

P32 segments follow the existing row-major order of the canonical QVQ inner 16-by-16 tile. They are not token rows.
For the normal linear orientation, one selector spans two hidden-axis rows by 16 vocabulary-axis columns. The embedding
kernel must decode the requested vocabulary column from this same layout; it must not reinterpret selectors as one
schedule per token.

The all-zero selector schedule is representable, but a mixed-bank dynamic program may prune the exact canonical
history while merging survivors. Therefore bank 0 must also be encoded independently. Falling back means restoring
that complete independent artifact, not merely zeroing selectors on a mixed traceback.

## Correct objective for a tied parameter

### The ideal shared-gradient Fisher

For one independent sequence `s`, let

\[
G_s^{\text{emb}}=\frac{\partial \ell_s}{\partial W}\bigg|_{\text{embedding use}},
\qquad
G_s^{\text{head}}=\frac{\partial \ell_s}{\partial W}\bigg|_{\text{LM-head use}}.
\]

Because the parameter is tied, its score gradient is the sum before any Fisher or YAQA factor is formed:

\[
G_s=G_s^{\text{emb}}+G_s^{\text{head}}.
\]

The empirical Fisher target is

\[
F=\mathbb{E}_s\left[\operatorname{vec}(G_s)\operatorname{vec}(G_s)^T\right].
\]

This includes embedding/head cross terms. Building separate YAQA losses and adding them omits
`G_emb^T G_head`, `G_head^T G_emb`, and their vocabulary-axis equivalents. A role-separated sum is useful as an
ablation, but it is not the mathematically complete tied objective.

The exact Sketch-B factors for the shared matrix orientation are

\[
H_D=\frac{1}{V}\mathbb{E}_s[G_s^T G_s] \in \mathbb{R}^{D\times D},
\qquad
H_V=\frac{1}{D}\mathbb{E}_s[G_s G_s^T] \in \mathbb{R}^{V\times V}.
\]

For reconstruction error `E=Q-W`, stock YAQA would score

\[
J_{\text{full}}(Q)=\operatorname{tr}(E H_D E^T H_V).
\]

This is the reference math. Tests with a small vocabulary must materialize `G_s`, `H_D`, and `H_V` and compare every
optimized approximation against this oracle.

### Why stock YAQA cannot be applied directly

For Llama 3.2, `H_V` has `128256^2` elements. It would require about 61.3 GiB even in FP32, before factorization or
workspace. Stock two-sided YAQA also assumes one dense input factor and one dense output factor for an ordinary linear
module. Running it independently on the embedding and head would break the shared-parameter objective.

The tied endpoint therefore needs a bounded representation of the shared factor, not a dense vocabulary Hessian.

### Scalable tied-gradient Sketch-B target

The first production representation uses a diagonal vocabulary factor:

\[
d_t=\frac{1}{D}\mathbb{E}_s\left[\lVert G_s[t,:]\rVert_2^2\right],
\qquad
H_V \approx \operatorname{diag}(d).
\]

Its proxy is

\[
J_{\text{diag}}(Q)
=\sum_{t=1}^{V}d_t\,E_{t,:}H_D E_{t,:}^T.
\]

This retains a full hidden-axis factor, gives every vocabulary row an explicit importance, and remains chunkable in
`O(D^2 + V)` factor storage. The score is nonnegative when `H_D` is positive semidefinite and `d_t >= 0`.

Neither full `G_s` nor an exact `G_s^T G_s` should be materialized for a large vocabulary. Use independent randomized
projections. For Rademacher sketches with unit covariance,

\[
R_s \in \mathbb{R}^{V \times r_v},
\qquad
S_s \in \mathbb{R}^{D \times r_h},
\]

\[
\widehat H_D=
\frac{1}{N V r_v}\sum_s
(R_s^T G_s)^T(R_s^T G_s),
\]

\[
\widehat d_t=
\frac{1}{N D r_h}\sum_s
\lVert [G_s S_s]_{t,:}\rVert_2^2.
\]

Both estimates are unbiased for their respective factors. Use independent sketch streams so their product does not
acquire avoidable correlated-sketch bias.

The projections can be formed from the two call sites without constructing `G_s`. If `A_s` is the score derivative
with respect to logits, `H_s` is the final hidden activation, `B_s` is the derivative with respect to embedding output,
and `scatter_ids(B_s)` sums rows of `B_s` by token ID, then

\[
G_s^{\text{head}}=A_s^T H_s,
\qquad
G_s^{\text{emb}}=\operatorname{scatter}_{\text{ids}}(B_s).
\]

Useful projected forms are

\[
R_s^T G_s^{\text{head}}=(A_sR_s)^T H_s,
\qquad
R_s^T G_s^{\text{emb}}=\sum_p R_s[\operatorname{id}_p,:]^T B_s[p,:],
\]

and

\[
G_s^{\text{head}}S_s=A_s^T(H_sS_s),
\qquad
G_s^{\text{emb}}S_s=\operatorname{scatter}_{\text{ids}}(B_sS_s).
\]

Adding each pair before its Gram or row norm preserves the tied cross-role terms. Padding positions are excluded from
the embedding scatter and loss, while the padding token remains a normal vocabulary output unless the model masks it
at the LM head.

A later quality tier may retain a bounded low-rank vocabulary correction

\[
H_V \approx \operatorname{diag}(d)+U\Lambda U^T,
\]

whose extra score is

\[
\lVert \Lambda^{1/2}U^T E L_D\rVert_F^2,
\qquad H_D=L_D L_D^T.
\]

This can capture cross-token curvature without a dense `V x V` matrix. It should be introduced only after the
diagonal collector agrees with the exact small-vocabulary oracle.

### Numerical rules

- Form one score gradient per independent sequence; never average sequences before the Gram update.
- Disable TF32 while collecting or scoring reference factors.
- Accumulate projected Grams in FP64 where supported, or use bounded FP32 accumulation with an overflow flag and
  deterministic chunk reduction.
- Symmetrize `H_D`; clamp only roundoff-sized negative eigenvalues or diagonals and reject materially indefinite or
  non-finite factors.
- Floor the vocabulary diagonal with a documented relative epsilon so rare rows are not unconstrained. Record the
  unfloored mass and affected-row count.
- Normalize objective terms only with fixed dense/bank-zero statistics. Candidate-dependent normalization changes the
  winner and is invalid.
- Recompute near-tied candidate scores with the high-precision reference reduction before choosing a selector map or
  alternative family.

## Quantization and candidate selection

The diagonal tied proxy is not the same geometry as the current dense two-sided YAQA anti-diagonal recurrence. The
initial implementation should keep the proven segmented V2 quantizer as candidate generation and use the tied proxy
for complete-artifact reranking:

1. Encode an independent canonical V2 candidate.
2. For each alternative family ID 1, 2, and 3, generate complete V2B2-P32 candidates using the same rate, seed,
   trellis policy, and hidden transform.
3. Generate candidates biased toward embedding-only, head-only, and tied-gradient factors. These are proposal
   mechanisms; none is authoritative by itself.
4. Score every complete reconstructed matrix with `J_diag` and, when enabled, the low-rank correction.
5. Keep a bounded shortlist including canonical V2, each family winner, and candidates within the configured proxy
   margin.
6. Install each shortlisted matrix at both call sites and run teacher-forced full-model replay.
7. Select by masked final-logit teacher KL subject to Top-1, Top-5, Top-10, finiteness, and margin guardrails.
8. If no candidate passes, restore the independent canonical V2 artifact atomically.

The complementary family ID is selected once for the shared artifact, not once per role. The selectors, trellis,
scales, and transform metadata are also shared. Selector occupancy, selector entropy, family ID, family churn,
bank-zero fallback, tied-proxy loss, final KL, and Top-N deltas must be reported.

An optimized generalized YAQA recurrence may follow later. For the diagonal vocabulary factor its error gradient is

\[
\nabla_E J_{\text{diag}}=2\operatorname{diag}(d)E H_D.
\]

That structure has hidden-axis feedback but no dense vocabulary-axis feedback. It should be implemented as a distinct
tied-endpoint quantizer rather than forcing fake dense factors through the stock YAQA API.

## Transform and scale policy

The current linear QVQ RHT mixes both input and output axes. It cannot be reused unchanged: a Hadamard transform across
the vocabulary axis mixes token identities, so one embedding lookup would require reconstructing or combining the
whole vocabulary.

The tied endpoint permits only a hidden-axis transform and vocabulary-row scales. A suitable shared factorization is

\[
Q=D_s Z R_D^T,
\]

where `R_D` is one orthogonal randomized hidden transform, `D_s` is one finite vocabulary-row scale vector, and `Z`
is the matrix represented by the shared QVQ payload. Then

\[
\operatorname{embed}_Q(t)=s_t Z_{t,:}R_D^T,
\qquad
\operatorname{head}_Q(h)=D_s Z(R_D^T h).
\]

These are two evaluations of the same `Q`. The first implementation may disable RHT entirely as a correctness
baseline. If hidden-only RHT is enabled, both kernels must use the same signs, normalization, and row scales. Separate
embedding and head `SU`/`SV` optimization is forbidden because it would create two effective matrices.

Define the dense transformed target as `T=W R_D` and the scaled transformed reconstruction as `T_hat=D_s Z`. Then

\[
E_{\text{original}}=Q-W=(\widehat T-T)R_D^T,
\qquad
H_D^{\text{transformed}}=R_D^T H_D R_D.
\]

Every proxy is evaluated after reconstructing the original-basis `Q`, or with this exact transformed identity. No
vocabulary-axis transform may appear in the checkpoint.

## Lifecycle ordering

Changing the input embedding changes every downstream activation. The tied endpoint must therefore be chosen before
decoder quantization, then held fixed:

```text
load immutable dense teacher and shell/turtle student
  -> detect the tie by parameter/storage identity and model metadata
  -> collect disjoint tied-endpoint Sketch-B and replay statistics
  -> generate and replay shared endpoint candidates
  -> freeze one shared packed endpoint artifact
  -> recapture decoder Hessians/Sketch-B on the student trajectory
  -> quantize decoder layers true-sequentially
  -> save, reload, and evaluate the packed model
```

The dense teacher remains unchanged for score gradients and final-logit targets. Decoder preparation captured before
the embedding candidate is installed is stale and must not be reused.

One optional outer refinement is valid:

1. quantize the endpoint;
2. quantize the decoder against that endpoint;
3. recollect tied-endpoint statistics through the quantized decoder;
4. select a revised endpoint;
5. requantize the decoder from the pristine source weights.

Updating the embedding after decoder quantization without repeating decoder preparation and quantization is invalid.

Preparation artifacts record dataset identity, split rows, tokenization, masks, seed, sketch ranks, factor checksums,
and the shared parameter identity. They do not retain the raw dataset or source endpoint weight. LazyTurtle
materializes the endpoint once from its canonical checkpoint owner and releases it after candidate generation.

## Checkpoint ownership and loading

- Serialize trellis words, packed selectors, family ID, scales, and transform metadata once under a canonical shared
  endpoint owner.
- Store alias metadata for both the embedding and LM-head module paths.
- On load, create one immutable `QVQTiedEndpointWeight` owner and make both runtime wrappers reference it.
- Validate shape, dtype, codec version, selector count, family range, checksums, and tie metadata before either wrapper
  can execute.
- Save/reload must preserve logical aliasing. It is not sufficient for two independent tensors to happen to contain
  equal values.
- Multi-GPU placement may create immutable per-device caches from the one serialized owner. Cache replication is a
  runtime placement detail, not a second quantized parameter.
- A missing or replaced alias fails closed; it must never silently instantiate a dense or separately quantized head.

## Inference kernels

Two native paths are required:

### Embedding gather

Given integer token IDs, decode only the vocabulary columns and hidden coordinates needed by those IDs from the
canonical P32 tile layout, apply the shared row scale and inverse hidden transform, and return the normal embedding
shape. Preserve repeated IDs, arbitrary leading dimensions, empty inputs, and `padding_idx` behavior.

### LM-head projection

Apply the shared hidden transform to activations, run the P32 banked GEMV, apply the same vocabulary-row scales, and
produce logits. Selector extraction, state decoding, codebook values, FP32 product order, and family selection must
match eager reconstruction.

Quantization kernels require exact states, selectors, family IDs, and packed words relative to the reference path.
Inference may differ from dense FP32 matmul with the fully reconstructed `Q` by at most `2e-3` under the repository's
declared absolute/relative comparison; model-quality loss relative to dense `W` is measured separately.

## Validation plan

### Mathematical unit tests

- Compare explicit tied parameter gradients with `G_emb + G_head` on a toy tied model.
- Materialize the complete small-vocabulary empirical Fisher and exact Sketch-B factors.
- Verify the projected estimates converge to `H_D` and `diag(H_V)` over fixed seeds.
- Verify cross-role terms are present by constructing a case where the independent-role sum chooses a different
  candidate from the tied-gradient objective.
- Compare `J_diag` with an explicit row loop and verify nonnegativity for PSD factors.
- Verify the low-rank formula against a materialized `diag(d) + U Lambda U.T` reference.
- Exercise padding exclusion, repeated token IDs, unseen tokens, tied-gradient cancellation, near ties, non-finite
  inputs, and factor stabilization.

### Codec and persistence tests

- Cover W1, W1.5, W2, W2.5, W3, and W3.5 and all alternative family IDs.
- Require all-bank-zero quantization, packing, reload, embedding gather, and LM-head projection to be bit-for-bit
  canonical V2.
- Check selector boundaries crossing token columns and hidden rows in the canonical inner-tile order.
- Require exact states, selectors, family ID, packed words, and eager reconstructed `Q` across CPU and CUDA
  quantization paths.
- Save only one payload and restore one logical owner with two module aliases.
- Test same-device sharing and cross-device immutable cache replication.
- Reject per-role selector maps, illegal family IDs, malformed selector counts, vocabulary-axis RHT metadata, stale
  artifact fingerprints, and broken aliases.

### Kernel tests

- Compare embedding gather with `torch.nn.functional.embedding(ids, Q_dense)` for unique, repeated, padded, empty,
  and adversarial token IDs.
- Compare LM-head output with `x.float() @ Q_dense.float().T` for decode and prefill shapes.
- Cover FP16 and BF16 inputs, FP32 accumulation, empty batches, non-default streams, save/reload, and every supported
  rate.
- Require deterministic output and no retained temporary allocation growth.
- Profile A100 `sm_80` and RTX 4090 `sm_89`; architecture-specific wins must remain gated.

### Llama 3.2 1B model gate

Use disjoint, recorded populations for endpoint statistics, decoder calibration, and evaluation. The initial gate
should use at least 512 full rows for each population, batch 1 without concatenation or sequence truncation, and record
the exact dataset name, split, row ranges, tokenizer revision, seed, and checkpoint revision.

Compare:

1. dense tied endpoint;
2. shared canonical V2;
3. V2B2-P32 selected by embedding-only proxy;
4. V2B2-P32 selected by head-only proxy;
5. V2B2-P32 selected by tied-gradient YAQA proxy;
6. V2B2-P32 selected by final-logit replay.

Run the packed model through all decoder layers and all normal linear modules. Report endpoint proxy loss, layer KL,
final KL, Top-1/Top-5/Top-10 retention, selector occupancy and entropy, family ID, fallback count, effective BPW,
quantization time, kernel time, and peak CPU/VRAM use.

Promotion requires:

- one serialized artifact and preserved logical tying after reload;
- no regression from the independent V2 fallback under its exact proxy;
- final-logit KL improvement over shared V2 on the disjoint evaluation population;
- no configured Top-N or margin guardrail violation;
- exact quantization metadata and the `2e-3` inference-kernel bound;
- bounded preparation and candidate memory independent of `V^2`.

## Explicit non-goals and rejected shortcuts

- Do not untie and quantize the two roles independently.
- Do not choose a head candidate and copy it to the embedding without joint replay.
- Do not call a sum of independent role losses the exact tied YAQA objective.
- Do not allocate or factor a dense vocabulary-by-vocabulary Hessian.
- Do not use a vocabulary-axis Hadamard/RHT in an embedding gather path.
- Do not maintain per-role trellises, selectors, family IDs, scales, or transforms.
- Do not update the shared endpoint after decoder quantization without re-preparing and requantizing the decoder.
- Do not treat successful dense reconstruction as proof that the packed embedding and head kernels agree.

## Current implementation gaps

The repository does not yet satisfy this contract:

- the calibrated QVQ processor restores only `Linear` and `Conv1D` weights, not `Embedding` weights;
- the existing embedding/lm-head lifecycle either rejects tied LM-head quantization or belongs to weight-only RTN;
- ordinary QVQ serializes and executes a linear module, not a shared endpoint owner with two runtime wrappers;
- current RHT metadata describes transforms on both linear axes and cannot be reused for token gather;
- current YAQA collection assumes one module use and one manageable dense output factor; a shared parameter used twice
  must combine its role gradients before factor formation;
- no native P32 embedding-gather kernel currently consumes the linear-oriented tile stream.

These are implementation tasks, not reasons to untie the model or weaken the mathematical contract.

## Implementation stages

1. Add the small-vocabulary tied-gradient oracle and objective tests.
2. Add the bounded projected Sketch-B collector and compare it with the oracle.
3. Add one shared endpoint artifact, alias-aware save/load, and exact canonical-V2 fallback.
4. Add original-basis reference embedding/head wrappers with no RHT.
5. Add native P32 embedding gather and validate it against eager reconstruction.
6. Add the hidden-only transform and shared row-scale path behind an explicit toggle.
7. Add family/candidate reranking and final-logit replay.
8. Integrate endpoint-first preparation followed by decoder recapture and true-sequential quantization.
9. Run the full Llama 3.2 1B gate before enabling the feature outside research configurations.
