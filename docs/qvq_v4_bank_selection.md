# QVQ V4 propagation-aware bank selection

Status: experimental design. This document records the intended implementation and the decisions that must be
validated before propagation-aware V4 bank selection becomes a production default.

## Objective

QVQ V4 has four implicit PGC16 banks for each supported rate. Every 16x16 weight tile stores a two-bit selector:

```text
b[module, tile] in {0, 1, 2, 3}
```

The selector is fixed at quantization time and serialized with the tile. Inference does not inspect activations or
dynamically choose a bank.

The selector costs `2 / 256 = 0.0078125` effective bits per weight before container metadata. Bank-aware inference
also performs the selector load/mask already present in the four-bank V4 kernel. Propagation-aware optimization adds
no inference tensors or operations beyond that banked format; it must not be described as cost-free relative to the
one-bank V4 control.

The objective is to choose these selectors to reduce error after it propagates through the quantized model,
especially below W3. Every quantization must first produce a deterministic locally selected bank map. The sequential
Block-LDLQ/YAQA solver is greedy under its proxy and has a canonical-bank rollback; it is not a global minimizer over
all mixed maps. The resulting map is a complete, loadable fallback checkpoint, not merely a diagnostic. When replay
is requested, local reconstruction may generate this first map and legal trellis paths *inside* each bank, but it must
not prune the alternate banks or accept a refinement candidate. A locally larger residual can cancel an existing
downstream residual, while a locally smaller residual can rotate into a high-gain direction and damage final logits.

The design must preserve:

- the V4 planar weight payload;
- two selector bits per 16x16 tile;
- zero new inference operations relative to the existing four-bank V4 decoder;
- zero new production codebooks;
- exact Torch/native reconstruction parity;
- bounded GPU residency during quantization;
- an immutable, atomically restorable local baseline.

The implementation order is also a design constraint: use the cheapest exact candidate family first. Do not retain
four full trellis streams, regenerate Viterbi paths, estimate curvature, or run broad autoregressive rollout until
exact full-output replay shows that the cheaper selector-only and trellis-automorphism families are insufficient.
Every proposal tier remains subordinate to exact propagated replay; efficiency must not restore local-error authority.
Every planned step also carries a proof obligation before optimized implementation: algebra for exactness and resource
claims, finite-precision bounds for numerical claims, and a frozen paired statistical target for end-to-end accuracy.
The proof ledger below records what is already established and what remains unproven so implementation effort cannot
turn an attractive hypothesis into an undocumented default.

## Empirical premise and authority boundary

QVQ experiments show a rate-dependent change in how useful local metrics are:

- at W4 and above, local reconstruction improvements often correlate with full-model recovery;
- near W3, the correlation becomes weaker and configuration-dependent;
- below W3, lower local MSE, local or layer KLD, and local top-1/top-5 error have repeatedly accompanied worse
  end-to-end behavior.

The four-bank V4 design exists to exploit this low-rate regime. A fast local selector is still necessary so every run
can finish with a valid checkpoint even when propagation replay is disabled or fails. Its locally selected bank map is
a fallback and A/B control, not evidence of global or end-to-end optimality. With replay enabled, the banks provide
legal residual directions so quantization can choose a direction that causes less downstream damage or cancels error
already present in the live quantized model.

Use this authority ladder for sub-W3 bank optimization:

1. Local MSE or a local Hessian/YAQA proxy deterministically chooses the first legal path and bank for every tile. It
   may finalize a `local` artifact when replay is disabled. When replay is enabled, this decision is the immutable
   fallback and may not prune, accept, or veto an alternate bank.
2. Exact completed-layer replay may conditionally replace the local baseline only in explicit `layer_replay` mode.
   The result must be labelled layer-refined; local or partial-layer KLD/top-k alone is not end-to-end evidence.
3. A transform-correct full-output derivative signal from the live quantized model may generate and rank a bounded
   set of bank maps through an exact workspace or explicitly approximate proposal-only sketch. It may not accept a map
   without exact replay.
4. Exact final-output replay on prompt-disjoint replay-selection rows may conditionally accept a candidate generated
   from the proposal rows. It may not promote the policy without independent confirmation.
5. One prompt-disjoint confirmation decision may promote the fully frozen refined artifact over the immutable local
   root. Frozen external evaluation provides default-policy evidence and must not become another tuning population.

Non-finite tensors, malformed serialization, and reconstruction mismatch are correctness failures rather than local
quality judgments; they always reject a candidate.

## Current implementation gap

The merged implementation is a correct baseline and kernel format, but it does not yet satisfy this sub-W3 bank
authority contract:

- `block_ldlq_inner_banked` assigns tile banks from local candidate error;
- the automatic processor propagation gate compares outputs at the current module boundary and obtains its automatic
  gate by withholding token rows from ordinary calibration batches, not from a prompt-disjoint propagation split;
- `qvq_v4_candidates.select_v4_candidate` prunes offline V4 mapping proposals with local MSE and p95 error;
- `QVQPropagationRefiner` can evaluate serialized full-model candidates, but it is not integrated into the production
  quantization lifecycle or supplied with full-output-derived bank maps.

Treat the local-selection paths as the intended fast-baseline implementation. They must not be described as the V4
sub-W3 propagation-recovery mechanism until the replay mode, provenance, final-output generator, and exact lifecycle
refiner in this document are wired. The optimized inference kernels remain valid because selector optimization changes
only serialized bank IDs and trellis states, not decoder operations.

The existing `QVQConfig.propagated_bank_selection` field names the module-boundary gate above. It must not be reused
for the new replay lifecycle: doing so would make `propagation enabled` ambiguous and could silently run the local
token-row gate when the user requested final-output replay. The implementation should replace that field with the
nested `PropagationConfig` in this document. If the old behavior remains useful as an experiment, expose it under an
explicit `local_output_gate` control that is default-off and labelled local-only. Do not run it before, during, or as
a fallback for `replay_mode="full_model"`. The new lifecycle supports either Block-LDLQ or YAQA for the local root;
the current Block-LDLQ-only restriction belongs only to the legacy module gate.

## Bank-selection granularity

The four banks are not selected once per model, layer, or module. They are selected per 16x16 tile. Brute-force
end-to-end replay of every possible map is impossible:

```text
number of complete maps = 4 ** tile_count
```

QVQ therefore separates the unconditional local fallback from optional candidate generation and acceptance:

```text
fast local bank/path argmin
  ├─ replay disabled
  │    └─ serialize local baseline
  │
  └─ replay enabled
       └─ transform-correct final-output proposal sweep on proposal prompts
            └─ exact dense-gradient or direct-action-score workspace
                 ├─ same-state selector alternatives
                 └─ exact edge-XOR trellis-orbit alternatives
                      └─ bounded non-local immutable maps
                           └─ exact final-output replay on disjoint selection prompts
                                ├─ measured gain: freeze finalist/alignment decisions
                                └─ no gain: optionally escalate to sibling/tilted paths
                                     └─ one paired adaptive confirmation procedure
                                          ├─ promote refined artifact
                                          └─ retain exact local root
```

A replay evaluates a complete candidate map for a semantic coordinate. It does not run one full-model forward for
each individual tile.

Block-LDLQ, Viterbi, and YAQA may generate paths conditional on `bank=b`. Their losses answer, "What is a strong legal
encoding on this bank manifold?" A local argmin over those results supplies the mandatory fallback, but it does not
prove either the best intra-bank path or the best bank for end-to-end behavior. Reusing that local ordering as the
replay acceptance rule would reduce V4 back to a local codebook search and defeat its low-rate purpose.

## Four-bank set design

Selector optimization cannot recover if all four fixed banks produce nearly the same residual direction. The
rate-keyed bank definitions therefore need a separate offline design gate.

For tile `t`, bank `b`, live input `X_t`, effective dense decoded weight `Q_eff[t, b]`, and final-output Jacobian
`J_t`, define

```text
E_eff[t, b] = W[t] - Q_eff[t, b]
P[t, b] = J_t @ (X_t @ E_eff[t, b].T).
```

This effective-dense expression is a diagnostic definition. A PGC16 tile lives in QVQ's Hadamard-transformed inner
weight, so propagation-aware Viterbi must use the exact transform-domain adjoint derived in Stage 2 rather than
substituting an inner tile directly for `E_eff`. `E_eff` is useful for diagnostics, but diversity in `P` is the
relevant capacity. A useful four-bank set should provide:

- low pairwise cosine between propagated residuals where the banks differ;
- useful spread in propagated residual norm and direction;
- non-collapsed bank occupancy after full-output-derived selection;
- stable selector choices across prompt shards and seeds;
- measurable exact final-output recovery over the one-bank V4 control.

Optimize the fixed bank XOR masks and the Tier-B edge-XOR orbit palette **jointly per rate** with a nested development
protocol. They are one combined action family at quantization time: for orbit state mask `c` and bank mask `m_b`, the
four decoded values come from `mix(s XOR c)` and `mix(s XOR c XOR m_b)`. Optimizing the two palettes independently can
produce duplicate or highly correlated actions even when each palette looks diverse in isolation. Reject a candidate
palette with duplicate combined actions, collapsed propagated directions, or poor live occupancy before freezing a
codec version.

The joint scorer should reuse exact decoder subexpressions. Four orbit masks require only four first-pair mixes, while
the second pair requires at most the sixteen distinct `c XOR m_b` mixes. That is at most twenty mixed states, or forty
scalar table outputs, instead of independently decoding sixty-four scalar outputs for sixteen four-value actions. This
37.5% reduction changes neither the candidate vectors nor their scores. Deduplicate repeated `c XOR m_b` values before
launch without changing deterministic action order or tie breaking.

For each development model, generate its own bank/orbit map on that model's design-search split, then compare frozen
joint palettes on its disjoint design-validation prompts by

```text
min_bank_orbit_set E_development_model,validation_prompt[
    L_final(model, validation_prompt, map_model_search(model, bank_orbit_set); bank_orbit_set)
]
```

subject to bijection, decoder cost, zero serialized codebook tensors, and Torch/CUDA/MPS/MLX parity. Do not optimize
the joint palette and its maps on the same prompts. Local covering radius, quadrant balance, residual cosine, and
occupancy may generate diverse proposals and diagnose collapse, but they cannot eliminate a valid palette on quality
grounds. Exact propagated recovery on the design-validation split selects the palette. After selection, run one outer
test on untouched model families and prompts; that outer test supplies generalization evidence and may not select
another mask. A palette that fails the outer gate is rejected rather than retuned on those rows. Changing frozen bank
or orbit masks requires a new codec version and synchronized kernel constants.

## Data contract

When propagation replay is enabled, distinguish map **proposal**, replay **selection**, and final **confirmation**.
The default accuracy contract makes all three prompt-disjoint from each other and from ordinary calibration, YAQA
Fisher data, bank-set development, and final evaluation. Output-alignment and finalist-rollout tuning belong to the
selection population. Proposal data may share data with YAQA only when explicitly configured and recorded;
confirmation may not share with any influencing population.

```text
replay prompt pool
  ├─ proposal prompts P
  │    └─ gradients and frozen candidate-map generation only
  ├─ replay-selection prompts S
  │    └─ exact successive-fidelity replay, alignment, and finalist selection
  └─ confirmation prompts C
       └─ one held-out decision after every selection choice is frozen
```

| Population | Purpose |
|---|---|
| Ordinary calibration | Input Hessian, Block-LDLQ, scale statistics, and sequential quantized replay. |
| YAQA Fisher | Full-model Sketch-B factors when YAQA is enabled. |
| Propagation proposal | Full-output-gradient candidate generation. It cannot rank candidates by replay quality or accept one. |
| Replay selection | Exact conditional final-output decisions, output alignment, and finalist rollout. |
| Propagation confirmation | One independent check after all replay-selection and alignment decisions. |
| Final evaluation | Benchmarks that are not used to select banks or tune thresholds. |

If one unsplit `replay_dataset` is supplied, split complete prompts into `P`, `S`, and `C` before tokenization and
remove them from every other calibration/tuning population. Do not reserve different token rows from the same prompt.
If neither explicit proposal/selection/confirmation datasets nor an unsplit replay dataset is supplied, requested
replay fails closed; it must not silently consume ordinary calibration rows. Selection rows may be reused across the
bounded candidate decisions and alignment sweep; confirmation remains hidden until all selection decisions are
frozen. Track and serialize dataset revision, prompt hashes, token hashes, valid token counts, length distribution,
rendering contract, and random seeds for every population. Assert every forbidden prompt/token-hash intersection is
empty before quantization begins.

An explicit diagnostic fast mode may set `P == S` to save one dense-teacher capture. It must record the overlap, may
not claim P6 false-elimination coverage, and may not use a confidence sequence whose candidates were constructed from
future rows in that sequence. It either evaluates all frozen maps at full selection fidelity or labels any heuristic
pruning as unvalidated. The sealed `C` population remains independent, so it may still reject or promote one frozen
artifact under P10; proposal/selection reuse is an efficiency compromise, not strict-split evidence.

The independence requirement is mathematical, not merely procedural. Let the frozen candidate map be `M = f(P)` and
the paired selection difference on sequence `s` be `d_s(M)`. With `P` independent of `S`, conditioning on `M` leaves
the declared sequence-level bound applicable to `{d_s(M): s in S}`. If instead `M = f(S_1, ..., S_n)`, the candidate
depends on future observations in the same confidence sequence; its increments are not predictable under the bound's
filtration, so the advertised false-elimination probability does not follow. Sealed `C` still provides a valid final
test, but it cannot recover the compute spent selecting an overfit finalist or authorize the earlier prune.

The initial proposal and replay-selection caps are each 65,536 valid tokens and 128 independent sequences; their
budgets are resolved separately because gradient-map stability and paired replay-selection power need not converge at
the same sample count. Confirmation starts with separately sealed 16,384-token/32-sequence and
65,536-token/128-sequence populations, but its predeclared reserve may contain up
to 512 complete prompts when the power calculation requires it. Sequence and token limits are a joint contract: do
not claim support for 512 sequences while retaining a 65,536-token maximum unless the frozen population actually has
at most 128 valid tokens per sequence. Before dense-teacher capture, enumerate the complete maximum confirmation
reserve, record its exact token total and length distribution, and prove it fits the teacher/spool budget. After
selection-only variance resolves the required maximum look, open only the corresponding sealed prefix. Never truncate
individual prompts merely to meet a token cap. These are caps, not mandatory first-stage costs or universal
sample-complexity claims. Selection starts on a frozen stratified 8,192-token/32-sequence subset and expands only for
surviving maps. Confirmation may use a predeclared paired group-sequential procedure, but its maximum reserve, stop
boundaries, and alpha spending are frozen before proposal generation. Fail closed when replay is requested and the
configured minimum or power-resolved reserve cannot be met. `replay_mode="none"` requires neither propagation
population nor dense-teacher capture.

## Compact dense-teacher targets

Full vocabulary logits are too large to retain for large models. During a dense teacher pass, store the following per
valid token:

- top-k teacher token IDs and logits, initially k=32;
- teacher log-sum-exp at every configured evaluation temperature;
- the natural next-token label when available.

Categorical teacher samples and extra temperatures are optional ablations, not initial defaults. The deterministic
default uses one temperature and top-k-plus-rest targets, natural NLL, and a smooth margin. The smooth competitor term
`logsumexp(z_without_a)` can be recovered stably from the already computed `logsumexp(z)` and `z[a]`; it must not
trigger a second vocabulary scan. Teacher top-1/top-5 identities and the top-1 margin are views of the sorted top-k
sketch and are not serialized again.

For the predeclared bounded self-conditioned rollout subset, also store the dense rollout token trajectory, stopping
condition, step mask, and compact per-step teacher targets. Capture these before proposal generation; a rollout contract added
after candidate results are visible is invalid.

Teacher-forced and self-conditioned targets are different contracts. Compact logits captured on the dense prefix are
valid only while the candidate is evaluated on that same prefix. After a candidate emits a different token, its next
prefix is different and the stored dense logits are not a teacher distribution for that state. The initial exact
rollout acceptance metric is therefore deterministic greedy-trajectory disagreement:

```text
L_rollout(sequence)
    = levenshtein(candidate_generated_tokens, dense_generated_tokens)
      / max(1, candidate_length, dense_length).
```

Use the same rendering, maximum generation length, stop-token set, and deterministic tie contract for every artifact.
Track each candidate's EOS mask independently when variants are batched. The compact per-step logits remain
teacher-forced diagnostics before divergence. A future mode may query the dense teacher on candidate-generated
prefixes, but it must declare and charge those extra teacher forwards and evaluate local root and candidate under the
same prefix policy; it may not reuse dense-trajectory logits after divergence.

For a sample `y ~ p_dense`, the quantity

```text
-log p_quant(y)
```

is an unbiased Monte Carlo estimator of the teacher-to-quantized cross entropy. Teacher entropy is constant across
candidates, so paired candidate differences estimate forward-KL differences without retaining the full vocabulary
tensor. Use common random samples, token weights, and accumulation order across candidates to reduce variance. Never
resample a teacher label per candidate. This estimator may generate gradients and proposals, but it cannot be the sole
selection or confirmation authority because final-output KL has not reliably predicted downstream task recovery.
Small-model tests must compare this compact score with exact vocabulary KL, candidate deltas, and ranking agreement.

Top-k-plus-rest KL is a deterministic coarsened divergence: the retained top-k tokens remain distinct and all omitted
tokens become one bucket. It is not exact vocabulary KL. A temperature-specific omitted mass requires the teacher
log-sum-exp at that same temperature; temperature-1 metadata cannot reconstruct it at another temperature.

Layer-mode targets are a different contract. Hidden states are not categorical distributions, so do not apply
vocabulary KL or top-k metrics to them. Use FP32 standardized hidden-state error, cosine, norm ratio, and non-finite
guardrails at a completed semantic boundary. Record that these are truncated layer metrics, not logits.

## Stage-wise lifecycle

The efficient default data and fallback flow is:

```text
Dense model
  ├─ capture proposal-teacher targets P
  │    └─ optional fast mode reuses YAQA only when every contract matches exactly
  ├─ capture replay-selection targets S from disjoint prompts
  └─ capture sealed confirmation targets C from disjoint prompts

Ordinary calibration data
  └─ quantize every module
       └─ fast local bank winner per tile
            └─ save exact local-baseline artifact

If replay is disabled
  └─ finalize that local baseline

If replay is enabled
  ├─ resolve compute/transfer/I/O/headroom-safe workspace before the sweep
  │    ├─ exact dense/direct aggregate or simultaneous shards
  │    └─ if shard state does not fit: sequential disjoint-shard maps
  └─ one live-quantized final-output forward/backward sweep on P
       └─ score bounded actions
            ├─ selector-only alternatives
            └─ exact edge-XOR trellis-orbit alternatives
                 └─ at most three rate-aware/diverse model-wide maps
                      └─ exact small-fidelity replay on S of all three frozen maps
                           └─ eliminate a map only under a simultaneous paired bound
                                └─ unresolved W1--W2.5 maps advance on new medium rows
                                     └─ advance 1, 2, or all 3 unresolved maps on new full-fidelity rows
                                          └─ choose finalist, optional alignment, then isolated rollout
                                               └─ aligned rejection: conditionally test unaligned finalist

No measured gain or insufficient candidate diversity
  └─ generate baseline-history sibling paths only for full-output-ranked coordinates
       └─ if still insufficient, evaluate propagation-tilted Viterbi as a research escalation

Independent confirmation prompts
  └─ confirm the selected map
       ├─ pass: serialize the refined winner
       └─ fail, non-finite, or error: retain the exact local root
```

The dense pass supplies immutable targets; it does not choose banks. Replay gradients and candidate decisions are
computed after quantization against the live quantized baseline, where accumulated quantization error is present.

### Stage 0: teacher capture

When `replay_mode` is `layer` or `full_model`, capture immutable dense-teacher targets on proposal `P`, replay-
selection `S`, and confirmation `C` prompts before replacing the only available dense model. Accuracy-first mode uses
`P` disjoint from YAQA and `S` disjoint from both. Explicit fast mode may capture compact `P` targets during the YAQA
Fisher pass only when prompts, rendering, masks, teacher seed, temperature, target bytes, and evaluation mode match
exactly. A second explicit diagnostic mode may reuse `P` as `S` under the restricted, non-P6 contract above. These are
recorded pass reuses, not strict-split evidence. Confirmation always uses its own disjoint dense capture. Capture and
seal the complete pre-enumerated maximum confirmation reserve—including its exact full-prompt token total—while the
dense model is available; the later power decision may expose only a prefix and cannot request uncaptured rows. If any
contract differs, run a separate teacher pass and record why.

Full-model mode stores the compact final-output targets above. Layer mode is a deferred, lower-authority budget tier;
if enabled, it must not retain every full boundary tensor in accelerator or host memory. It uses one of two explicit
policies:

- `spool`: stream FP16/BF16 boundary targets to NVMe with a hard byte cap and per-chunk hashes;
- `recompute`: retain only prompt metadata and regenerate one layer's dense targets from the immutable source model
  when that layer becomes active.

Choose by measured disk capacity and dense-prefix cost; never silently switch policies. Keep only one active layer's
target microbatches in memory. Store confirmation targets behind a separate sealed interface that exposes only the
final paired confirmation operation after the refined artifact is frozen; proposal/selection code receives no handle
to them. Skip teacher capture entirely when `replay_mode="none"`.

### Stage 1: unconditional fast local baseline

Use Ultra's existing hooked-linear capture and early-stop exception path:

```text
forward to target hook
        |
capture live input and required output
        |
raise the capture exception
        |
quantize the target subset
        |
install/replay the accepted quantized result
        |
advance to the next subset
```

Ordinary capture does not require a complete layer or model forward for every projection. Preserve sequential live
inputs so later subsets observe errors introduced by accepted earlier subsets.

Run the existing sequential banked solver as one module-level transaction. Do not model the baseline as independent
per-tile argmins:

```text
Block-LDLQ
  └─ visit input blocks in feedback order
       ├─ construct the corrected block from already committed error
       ├─ evaluate all four banks on that same corrected block
       ├─ choose bank 0 on exact ties
       └─ commit the winning block before constructing the next block
            └─ compare the complete mixed proxy with canonical bank 0
                 └─ non-finite or non-improving: restore canonical bank 0

YAQA
  └─ visit anti-diagonals in two-sided feedback order
       ├─ construct corrected tiles from already committed error
       ├─ evaluate all four banks on each independent anti-diagonal tile
       ├─ choose bank 0 on exact ties
       └─ commit the anti-diagonal before constructing the next one
            └─ compare the complete Kronecker proxy with canonical bank 0
                 └─ non-finite or non-improving: restore canonical bank 0
```

This preserves the current Block-LDLQ and YAQA mathematics. Candidate rank is its deterministic generation order;
there is no invented `path.id`. Cross-device bitwise reproducibility is not presumed: record device/backend and test
selector parity separately. Within one backend and fixed seed, stable bank order and candidate order must make the
baseline repeatable.

Pack the winning trellis and bank selectors immediately, reconstruct them through the checkpoint decoder, and verify
shape, dtype, finiteness, and dense-reference parity. Save the resulting module artifact to a durable CPU/NVMe
transaction journal before advancing. After all modules are installed, hash and freeze the complete local-baseline
manifest.

This local selector is deliberately fast and unambitious. It ensures that every quantization produces a valid,
loadable checkpoint. It does not demonstrate that its selectors minimize layer or end-to-end loss below W3.

When replay is disabled, discard losing bank paths as soon as each tile or bounded tile batch is resolved and finalize
the local baseline. No dense-teacher capture, alternate-path spool, backward pass, or replay is required.

Replay candidate generation uses progressive exact tiers. A higher-cost tier is entered only when exact full-output
selection shows no useful gain or inadequate candidate diversity from all cheaper enabled tiers:

| Tier | Candidate family | Extra quantization work | Temporary representation |
|---|---|---:|---|
| A | Same-state bank selector | Decode/score only | Two bank bits per tile per map. |
| B | Exact edge-XOR trellis orbit plus bank | No Viterbi | Rate-keyed orbit ID and bank ID per tile. |
| C | Baseline-history sibling bank paths | Existing solver output or on-demand Viterbi | Active-module streams only. |
| D | Propagation-tilted Viterbi | New Viterbi search | Active-module streams only. |

Tier A keeps the baseline trellis states and changes only the bank selector. This is always a legal artifact because
the state recurrence is independent of the bank decoder. Under the current codec, a same-state bank remap changes
only the bank-dependent decoded pair, so it is deliberately restricted but exceptionally cheap.

Tier B obtains new legal paths without Viterbi. For V4 transition width `E = 4R`, choose an `E`-bit mask `a` and the
unique 16-bit periodic mask `c` satisfying

```text
c = ((c << E) & 65535) | a.
```

For every state and emitted edge in a tail-biting tile, apply

```text
s'_t = s_t XOR c
e'_t = e_t XOR a.
```

Then the transformed path obeys the exact recurrence:

```text
s'_(t+1)
    = ((s'_t << E) & 65535) | e'_t
    = s_(t+1) XOR c.
```

Tail biting is preserved because the same bijection is applied at every step. In planar storage this is an XOR of
the selected edge planes with all-ones words. The final winner uses the existing trellis payload and bank selector;
there is no new inference field or operation. Freeze a small rate-keyed orbit palette offline, initially baseline plus
three alternatives, and validate every supported `E` independently. There are `2**E` legal masks, but searching all
of them per tile is unnecessary and would merely move an offline design problem into every quantization run.

The compact map cost is tiny relative to full streams. A selector-only map uses `2 / 256 = 0.0078125` bits per weight.
A bank-plus-four-orbit action uses four bits per tile. The three bounded maps share that one action map; A and B add one
inclusion bit per tile, while C applies the complete action map. A and B need not be nested when B is the predeclared
shard-consensus mask. Tiles outside C's largest trust fraction store their local-root action in the shared map, so C
does not need a third mask even when its rate-specific maximum is below 100%. The intended total is six bits per tile,
or
`6 / 256 = 0.0234375` temporary bits per weight: about 5.86 GB for two trillion weights. Three independent four-bit
maps would be the simpler 11.72 GB upper bound. In contrast, four W2 trellis streams consume eight temporary bits per
weight, or 2 TB decimal, before the live baseline. The shared compact form is about 341 times smaller.

Tier C exploits work already performed by the banked Block-LDLQ or YAQA solver: both evaluate all four banks at each
corrected block/tile. On small models, optionally retain the losing baseline-history siblings while they are resident.
On large models, discard them initially and regenerate siblings only for modules or tiles ranked by the full-output
gradient. Never duplicate the winning baseline stream in the sibling store.

Tier D is research escalation. It must reload the immutable dense source weight and reproduce the exact scale, RHT
signs, calibration Hessian/factors, YAQA factors, solver order, and corrected-sequence inputs. Packed local streams
alone are insufficient. Write any Tier C/D streams directly to bounded CPU/NVMe storage, retain only an active-module
scratch on the accelerator, and discard rejected material immediately after synchronization.

When replay is disabled, only the winning `R`-bit stream and normal selector payload survive. No orbit maps, sibling
paths, source/factor regeneration metadata, or replay workspace are created.

### Stage 2: non-local bank-map generation

Skip Stages 2--5 when replay is disabled. Otherwise use an explicit three-population lifecycle:

```text
pass A: baseline
  └─ capture dense teacher targets
       └─ sequentially quantize and freeze the complete local model

pass B: propagation proposal
  └─ run P microbatches through the live quantized baseline
       ├─ accumulate exact dense gradients or direct action scores under the resolved cap
       └─ generate and freeze selector-only and exact edge-XOR maps without Viterbi

pass C: replay selection
  └─ run the frozen maps on prompt-disjoint S microbatches
       ├─ select by exact successive-fidelity final-output replay
       └─ only after exact replay rejects them and an escalation is enabled
            ├─ reload one immutable dense source module
            ├─ reconstruct or recompute its exact quantization factors
            ├─ generate bounded sibling or tilted paths
            └─ release dense weight, factors, gradients, and decoded workspace
```

The source provider must verify model revision, tensor fingerprint, calibration hashes, RHT seed/signs, scale, solver
configuration, and factor fingerprints before regeneration. If any dependency cannot be reproduced, fail closed to
selector/orbit selection or the saved local baseline; never regenerate from approximate state silently.

In `full_model` mode, construct the loss at final outputs. In `layer` mode, construct it at the declared completed-
layer boundary; this cheaper proxy must be recorded as layer-refined rather than end-to-end-refined. Keep the model in
evaluation mode. The initial reference implementation uses an explicitly differentiable QVQLinear reference/VJP path
without enabling dropout or training-time model behavior. Optimized native input-gradient kernels are a later
performance step and must match the reference gradients.

Do not pass the authoritative acceptance utility directly to autograd. Its task, greedy-rollout, top-k, and discrete
margin components are not generally differentiable. Define two API types with no implicit conversion:

```text
ProposalSurrogate
    differentiable, teacher-forced, final-output-derived
    -> gradients and bounded map proposals only

ReplayScore
    exact structured sequence metrics from the serialized runtime payload
    -> selection acceptance, confirmation, and fallback only
```

The initial full-model proposal surrogate is a deterministic frozen weighted sum of standardized top-k-plus-rest
teacher KL, natural-token NLL, and a smooth teacher-top-1 margin loss. For candidate logits `z`, fixed teacher-top-1
token `a`, and temperature `tau_margin`, one valid reference definition is

```text
L_proposal(sequence)
    = a_kl * mean_valid(coarsened_KL_topk_plus_rest(z)) / s_kl
      + a_nll * mean_valid(-log_softmax(z)[natural_label]) / s_nll
      + a_margin * mean_valid(
            softplus((logsumexp(z_without_a) - z[a]) / tau_margin)
        ) / s_margin.
```

Compute `logsumexp(z_without_a)` by stable log-subtraction from the already available `logsumexp(z)` and `z[a]`.
All coefficients, temperatures, masks, sequence/token reductions, and positive finite scales `s_*` are frozen from
dense proposal-only references before candidate generation. A component with no declared target has exactly zero
coefficient; coefficients are never silently renormalized. Sampled teacher cross entropy may be an explicitly enabled
ablation, but it is not required by the default. This surrogate may disagree with `ReplayScore`; exact replay
intentionally has final authority.

For module output `y`, candidate change `delta_y`, and live downstream gradient `g_y`, the generic first-order score is

```text
delta_L_first = <g_y, delta_y>.
```

QVQ Viterbi does not operate on an ordinary dense `delta_W`. It chooses a tile in the transformed inner matrix `Q`.
For the exact runtime operators

```text
x_h = T_in(x * SU)
z   = x_h @ Q
y   = T_out(z) * SV + bias,
```

the inner-output adjoint and first-order candidate score are

```text
g_h = T_out^*(g_y * SV)
delta_z = x_h @ delta_Q
delta_L_inner = <g_h, delta_z>.
```

`T_out^*` is the exact adjoint of the configured Hadamard operator; do not assume the implementation is symmetric or
drop `SU`, `SV`, normalization, composite-width, or transpose semantics. Hold the current `SU` and `SV` fixed during
bank/trellis proposal generation. The reference path should obtain `g_h` through autograd first, then validate any
explicit adjoint against the complete `QVQLinear` forward with FP64 finite differences. For a semantic group, sum the
first-order terms at every affected linear output before constructing a map.

This requires a microbatched proposal-dataset sweep, not one monolithic forward/backward call. The efficient mean
first-order ranker does not store a tensor for every sequence, tile, and bank. For a fixed tile alternative
`delta_Q[t, b]`, apply the configured sequence weight before accumulation and form

```text
G_Q = sum_sequence w_s * x_h,s.T @ g_h,s

sum_sequence w_s * <g_h,s, x_h,s @ delta_Q[t, b]>
    = <G_Q, delta_Q[t, b]>.
```

This contraction is exactly the same mean first-order score, up to the declared floating-point accumulation contract.
However, one data-major microbatch sweep cannot both accumulate the complete proposal population and keep only one
module-sized `G_Q`: every module's partial accumulator would have to survive until the last microbatch. The workspace
policy must therefore be explicit rather than claiming active-module memory for every model size:

| Policy | Exactness | Persistent proposal state | Intended scope |
|---|---|---:|---|
| `dense_gradient` | Exact within the accumulation contract. | One FP32 value per weight per active shard. | Small models or modules whose complete `G_Q` set fits the declared cap. |
| `direct_action_scores` | Exact within the accumulation contract. | Sixteen FP32 scalar scores per 256-weight tile per active shard. | Large dense modules and covered MoE experts. |
| `two_sketch_generator` | Approximate proposal generator only. | Two independent bounded gradient sketches. | Explicit research fallback beyond the exact host/NVMe cap. |

`direct_action_scores` applies linearity before storage:

```text
score[t, a] += w_s * <g_h,s, x_h,s @ delta_Q[t, a]>
```

The equality requires each Tier A/B `delta_Q[t, a]` to be frozen from the immutable local root and independent of the
proposal row; stateful Tier C/D regeneration is not folded into this accumulator. Under that condition it never
persists the complete cross-microbatch `G_Q` and remains algebraically identical to `<G_Q, delta_Q[t, a]>`. It may
materialize a bounded microbatch/tile gradient as explicitly charged below. Decode and contract the joint
bank/orbit actions in bounded tile chunks and retain deterministic FP32 accumulation and tie order. A score chunk is
not complete until it has consumed every assigned microbatch. Therefore "asynchronous persistence" must name one of
these schedules rather than implying that unfinished chunks can be written once:

| Direct-score schedule | Persistent state | Transfer/I/O consequence |
|---|---:|---|
| `host_resident` | Complete action-score accumulator in host RAM. | Transfer every partial score once, then write the final accumulator at most once. |
| `host_xg_scoring` | Complete action-score accumulator in NUMA-local host RAM. | Transfer transformed activations/adjoints and perform deterministic FP32 scoring on the CPU. |
| `nvme_read_modify_write` | Complete accumulator on NVMe. | Read and rewrite each score chunk for every contributing microbatch; never select this merely because capacity fits. |
| `module_or_stripe_major` | One module/stripe accumulator. | Recompute or spool the matching activations/adjoints so one chunk consumes all rows before one final write. |

A module/stripe-major exact alternative may release one completed accumulator at a time, but its activation spool or
extra model work must be charged. The one-sweep data-major path may use `host_resident` only when the complete score
state and its reserved headroom fit host RAM; NVMe capacity by itself does not make it efficient.

The auto policy must price compute as well as bytes. For a module with `K` input features, `N` output features, `S`
total valid activation rows, `B` microbatches, and `A=16` combined actions per tile, the leading FP32 work is

```text
C_dense  ~= 2*S*K*N + 2*A*K*N
C_direct ~= 2*S*K*N + 2*A*B*K*N

M_dense  = 4*K*N bytes per accumulator
M_direct = 4*A*K*N/256 = K*N/4 bytes per accumulator.

D_D2H_data_major_device_scores >= B*M_direct
D_D2H_host_xg = b_xg*S*(K + N)
D_D2H_module_major_final_scores >= M_direct
D_NVMe_host_resident_scores in {0, M_direct}  # compact in RAM or persist once
D_NVMe_read_modify_write >= (2*B - 1)*M_direct.

host_xg transfer wins on bytes when
    b_xg*S_b*(K + N) < K*N/4,  where S_b = S/B.

For K=N=H and FP16/BF16 x/g (b_xg=2):
    S_b < H/16.
```

Thus direct scoring cuts persistent proposal state by 16x at the price of repeating the small action-dot reduction for
each microbatch, `2*A*(B-1)*K*N` extra operations. Dense `G_Q` is preferred when it fits without harmful transfers;
direct scores are an exact large-model memory trade only when their transfer schedule also fits. `host_xg_scoring`
is a measured option, not a CPU preference: it is legal only when deterministic CPU/reference winner parity passes,
its NUMA-local accumulator fits, and its lower transfer cost outweighs lower host FLOP/s. The `C_dense`/`C_direct`
expressions price the contraction only. The complete proposal model is

```text
C_proposal_total = C_live_forward + C_live_backward_or_vjp
                   + C_action_device + C_action_host + C_recompute

T_proposal ~= (C_live_forward + C_live_backward_or_vjp + C_action_device)/F_device
             + C_action_host/F_host
             + D_D2H/BW_D2H + D_host/BW_host + D_NVMe/BW_NVMe.
```

The action formulas assume a tiled microbatch `X.T @ G` intermediate. Charge its bounded device or host tile, or the
full `4*K*N` FP32 temporary when the implementation materializes it, plus activation/adjoint spool and every repeated
forward/backward required by module/stripe-major scheduling. Require
`M_scores + M_retained + M_maps + M_temporary <= M_spool_cap - M_reserved`; the reserve is explicit and nonzero.
Measure fused/chunked launch, NUMA placement, reduction parity, and transfer costs before changing the crossover,
because FLOP and byte counts alone do not establish wall time.

The sketched path may only rank or form maps. A Johnson--Lindenstrauss dimension bound alone does not prove that the
best action is preserved. If every approximate action score has error at most `epsilon_score` and the exact gap between
the best and second-best actions is `gamma`, the winner is certified only when

```text
gamma > 2*epsilon_score.
```

Agreement from two independently seeded sketches is useful evidence but not a certificate when that score-gap
condition is unresolved. Keep the baseline on disagreement or near ties. The scalable research fallback is a first
sketch pass that retains the union of the top-`m`/near-tie actions from both sketches, followed by a second live
gradient pass that exactly rescores only that fixed shortlist. Its exact persistent score state is

```text
M_shortlist = 4*m*K*N/256 = m*K*N/64 bytes.
```

For `m=2`, that is 0.03125 bytes/weight, or 62.5 GB for two trillion weights instead of 500 GB. This is exact only
within the shortlist, so record shortlist recall and retain exact serialized-payload replay and sealed confirmation as
the sole publication authority. No sketch can publish an artifact directly.

Score Tier A/B one 16x16 tile at a time and never materialize a dense module-sized `delta_Q` for every action. The
scalable replay path is a quantization-only compact-overlay decoder: read the immutable parent trellis and compact
action map, apply the selected state XOR and bank ID while decoding, and never write a transformed trellis. The
confirmed winner is baked into the normal trellis and selector tensors once before publication, so production
inference has no orbit field or extra operation. Prove this overlay decoder bit-exact with materializing and installing
the same serialized payload.

For `N_tile` 16x16 tiles, `N_changed` non-baseline orbit tiles, and a packed `b_map`-bit action map, where `b_map` is
six for the shared-map path and eleven for the sequential-shard path, the logical payload and straightforward
reference arithmetic are

```text
B_overlay_map = N_tile * b_map / 8 bytes per map
C_overlay_reference = N_changed * (256 / V) state-XOR operations per module invocation
```

plus one bank/orbit-map extraction per tile. Cache residence, alignment, cache-line rounding, branchless treatment of
baseline actions, and kernel-launch effects determine physical traffic and are measured separately. At V4, `V=4`, so
the reference arithmetic increment is 64 state XORs per changed tile while the six-bit map is only 0.75 byte per tile.
This is the explicit cost paid to avoid writing a transformed trellis.

Materializing one active-module edge-XOR trellis scratch remains the simple reference fallback. If it is released after
each module invocation, a `B_replay`-microbatch full-model replay incurs at least

```text
D_orbit_scratch >= B_replay * sum_changed_orbit_module(
    B_parent_trellis_read + B_candidate_trellis_write
).
```

Caching transformed modules instead requires their complete residency or a declared module-major execution schedule.
The replay resolver must charge these bytes and choose compact-overlay decode, cached materialization, or reference
rematerialization by predicted and measured time; the compact six-/eleven-bit map size alone is not its execution cost.

Two or four fixed proposal-prompt-shard aggregates may diagnose sign instability without allocating per-sequence tile
tensors. When simultaneous exact shard accumulators exceed the cap, do not replace prompt-direction evidence
with two nested fractions of the same aggregate. Instead, process two disjoint prompt shards sequentially, compact
and release the first shard's action map before reusing one accumulator for the second, and build at most three exact-
replay proposals from shard 1, shard 2, and their agreement mask. Because the shards partition the rows,
`sum_j S_j = S`: leading forward/backward token work is unchanged and peak score state remains one accumulator. Total
peak is still

```text
M_sequential_peak = M_scores + M_prior_compact_shard_maps + M_temporary,
```

so the resolver must charge already compacted shard maps. There is no exact all-row aggregate unless separately paid
for. This fallback may persist two four-bit action streams plus three one-bit inclusion masks, `11 bits/tile` or
10.74 GB at two trillion weights, rather than pretending it still meets the six-bit shared-action-map contract. Shard
disagreement can trigger a predeclared escalation or smaller trust map, but it is not an acceptance test. Experts that
receive no proposal tokens retain their exact local-root action.

Under the strict `P != S` contract, the proposal sweep cannot supply the local-root `ReplayScore`; compute the baseline
once on each newly opened `S` fidelity and reuse it for all same-shape candidates. The diagnostic `P == S` mode may
reuse a score only when its forward path, batch shape, decoder, kernel, dtype, masks, and accumulation order exactly
match the serial replay evaluator. A differentiable proposal forward is not silently interchangeable with a native
inference forward merely because their outputs are close.

Curvature is optional and default-off. Even diagonal output curvature is not independently tile-additive. If
`delta_z = sum_t delta_z_t`, then

```text
delta_z.T @ D @ delta_z
    = sum_t delta_z_t.T @ D @ delta_z_t
      + 2 * sum_(t < u) delta_z_t.T @ D @ delta_z_u.
```

A per-tile emission that keeps only the first sum changes the intended objective. The Hadamard/SV epilogue can also
turn diagonal curvature in output space into a dense operator in inner space. Therefore the initial implementation
may use curvature only to rescore an already assembled complete map through the exact QVQ transform. It must not put
an isolated quadratic term into Viterbi emissions. A later incremental formulation must carry the accumulated
projected residual or prove an equivalent decomposition before curvature-aware Viterbi is enabled.

One reduced loss gradient does not estimate Fisher/Gauss--Newton curvature. Empirical Fisher must accumulate squared
per-sequence or explicitly defined per-token score gradients before reduction. A Hutchinson/Gauss--Newton estimator
must declare its probe count and HVP/VJP passes. Never square the mean gradient and call it Fisher. Record estimator
type, sample/probe count, normalization, dtype, and additional forward/backward-equivalent cost.

This proxy can prefer a candidate with worse local MSE when it cancels accumulated downstream error. Recompute live
sensitivity after an accepted sweep; do not reuse dense-model gradients as if they described the quantized model.

For `full_model` mode, the differentiated proposal surrogate must be constructed at the final model output. Module
MSE, module KLD, layer KLD, or a truncated layer output cannot be the sole generator, ranker, or acceptor of a
full-model candidate below W3. A normalized local term may regularize a final-output-derived path proposal but has no
acceptance authority.
Teacher-forced final logits, sequence loss, top-logit margins, and sampled teacher cross entropy may contribute to the
full-output surrogate. `layer` mode is an explicit lower-authority cost tier and must not be silently substituted when
full-model replay was requested.

Selector-only and exact edge-XOR orbit candidates are the first proposal implementations. They are legal serialized
artifacts and avoid independent-history path splicing. Baseline-history sibling streams are the first escalation. A
mixed sibling map splices paths generated under different Block-LDLQ/YAQA error histories, so it is not the optimum
for the assembled mixed history. Record that arm as `baseline_history_sibling_proposal`, not
`exact_mixed_quantization`.

Propagation-tilted path regeneration is the target implementation after the two-pass source/factor contract above is
validated. Full-model mode feeds the transform-correct final-output-derived linear term back into Viterbi emissions;
layer mode uses the analogous completed-layer term with its lower-authority provenance. For a path edge reconstructing
inner vector `q`, define a normalized additive objective

```text
c_lambda(q)
    = c_local(q) / s_local
      + lambda * <g_h, x_h @ q> / s_propagation.
```

`s_local` and `s_propagation` are positive finite module/rate scales frozen from proposal-only baseline emissions before
the sweep. Their exact estimator, epsilon floor, clipping, sequence normalization, and accumulation dtype are part of
the artifact provenance. Multiplying the local objective by the positive constant `1 / s_local` preserves its argmin,
so `lambda=0` must reproduce the declared bank-specific local path bit-for-bit on the same backend. Positive `lambda`
values tilt that path using downstream sensitivity. Always retain the separately serialized local path even if
regeneration is enabled. An optional hard trust radius is a proposal-only experiment and must be named, normalized,
and swept
separately; it cannot be hidden inside `lambda` or used by replay acceptance. Exact replay at the configured horizon
chooses among the resulting paths. This expands propagation-aware choice inside each bank without changing inference
storage or operations.

Use hooks to consume activations and gradients as the backward traversal reaches a subset. Generate Tier A/B scores
through the selected workspace policy, then free the activation, gradient, and decoded workspace. Persist compact
action maps, retained best-action gains, optional shard summaries, workspace provenance, and immutable hashes. Do not
persist every rejected action after the deterministic ranking thresholds are resolved. Only Tier C/D persists packed
candidate state streams, and only for the active full-output-ranked coordinate.

One final-output proposal sweep produces ordered tile actions and fixed shard summaries. Do not create redundant
"risk-adjusted" and "first-order" maps from the same signal. The baseline remains an available action at every tile.
Construct at most three complete maps, but make their trust region rate-aware because a full-direction map is much
more disruptive at W1/W1.5 than at W2.5/W3. Fractions are maximum inclusion caps, not quotas: never fill a map with a
non-improving, sketch-uncertified, or numerically unresolved action merely to hit the displayed percentage. The
initial development caps are hypotheses to freeze before teacher capture and validate by rate:

```text
rate       map A   map B   map C
W1/W1.5      5%     15%     35%
W2          10%     30%     60%
W2.5/W3     25%     50%    100%
```

Exact replay chooses the step size. When simultaneous aggregate and shard scores fit, the frozen shard-churn rule may
replace the middle correlated step with a shard-consensus inclusion mask over the same primary action IDs; do not add
a fourth candidate. This preserves the six-bit shared-action-map contract and candidate cap. When only the sequential
disjoint-shard fallback fits, use the separately charged 11-bit temporary layout above and construct at most one
shard-1 direction, one shard-2 direction, and one agreement direction. This is not the rejected same-signal second-
best-action arm: the two action streams come from disjoint final-output populations and exist specifically to avoid
aggregate cancellation. A generic second-best-action map remains a research arm because it adds another action-ID
stream without adding independent propagation evidence. A proposed action must beat the baseline action under the
configured generator and satisfy any sketch/shard agreement rule before it can enter a map.

Per-sequence paired statistics are computed from exact replay outputs of these complete maps, where cross-tile and
cross-module terms are present. Selection ranking itself needs no nominal confidence claim; independent sealed
confirmation supplies the inferential gate. Record rate-specific action count, selector churn, shard agreement, and
the fraction of tiles that retained the baseline so later evidence can revise the trust schedule rather than silently
changing it.

### Stage 3: layer replay or partial-boundary instrumentation

This stage is not part of the initial full-model MVP. It is an optional lower-authority budget fallback and diagnostic
mode after the minimal full-output path works. Reuse the existing output-alignment clean-replay/hook infrastructure;
do not build a second boundary-spool lifecycle merely for bank search. When explicitly enabled, use the existing
hook/exception mechanism to stop at the nearest meaningful nonlinear boundary:

| Candidate coordinate | First required replay boundary |
|---|---|
| QKV | Completed attention output. |
| Attention O | Attention residual output. |
| Dense MLP gate/up | Completed activation-and-product output. |
| Dense MLP down | MLP residual output. |
| Expert gate/up/down | Aggregated MoE output. |
| Router | Complete rerouted MoE output. |

Score every bounded candidate map from the same captured input; do not rerun prefix capture once per bank or map. In
`layer` mode, evaluate each serialized candidate as an immutable overlay from the same current manifest, replay through
the completed owning-layer boundary, and compare it with the saved local/current baseline on replay-selection
prompts. Accept only a paired improvement larger than measured replay noise with no numerical guardrail regression.
Discard the overlay after every losing, tied, non-finite, or failed candidate. Do not expose confirmation rows while
selecting individual layers.

The layer-mode primary selection loss is FP32 normalized hidden-state error:

```text
reference_floor = epsilon_relative * mean_all_valid(Y_dense^2)
relative_token_error =
    mean_feature((Y_quant - Y_dense)^2)
    / max(mean_feature(Y_dense^2), reference_floor)

L_hidden = mean_sequence(mean_valid_token(relative_token_error))
```

Also record cosine, norm ratio, p95/p99 token error, and non-finite counts. These are guardrails and diagnostics, not
categorical KL or top-k. The dataset-relative floor prevents nearly zero-energy tokens from dominating the ratio.
Use identical masks, per-sequence weighting, and accumulation order for baseline and candidate.

When a layer-mode artifact changes more than one boundary, its selection and final confirmation primary is not whichever
layer was processed last. Aggregate all predeclared changed boundaries with fixed positive weights:

```text
U_layer(sequence)
    = sum_boundary w_boundary
      * standardized_normalized_hidden_error(boundary, sequence).
```

Freeze boundary weights and dense/proposal normalization constants before refinement, require the same primary sequence
IDs at every included boundary, and evaluate the complete local root and complete refined artifact across the same
boundary set at confirmation. Spool or recompute those targets under the existing byte cap. Missing boundary targets
reject the refined artifact rather than silently changing the aggregate.

Layer replay is more propagation-aware than tile or module loss because it includes the complete attention, MLP, or
MoE interaction at that boundary. It remains a truncated heuristic: later layers may amplify, rotate, or cancel the
observed error. A checkpoint selected this way must record `bank_selection_provenance="layer_replay"` and may not be
reported as full-model recovery evidence.

In `full_model` mode, partial-boundary measurements identify amplification, routing changes, and the earliest
divergence. They may schedule prefetch or diagnostic work, but they must not choose the stateful candidate order,
decide which semantic coordinate to split, or eliminate a finite, valid candidate below W3. Greedy conditional
selection is order-dependent, so a local ordering heuristic would still give local metrics indirect control over the
final bank map.

A two-to-four-layer propagation window can collect diagnostics when requested, but is default-off. In `full_model`
mode it is not a ranking or acceptance gate. Use a fixed canonical order or a full-output-derived ordering for
stateful refinement. The bounded candidate count from Stage 2, not a local or short-window cutoff, controls
final-output replay cost.

### Stage 4: exact full-output replay selection

Run this stage only in `full_model` mode. Every distinct candidate map emitted by the full-output-derived generator
runs through the configured final-output selection comparator. Final-logit KL or sampled teacher cross entropy may be a
proposal signal or guardrail, but cannot accept a candidate by itself. The prefix before the candidate must not be
rerun when an exact cached boundary input is available:

```text
cached accepted state entering the candidate coordinate
        |
candidate serialized bank map
        |
owning compound operation
        |
remaining layer/model suffix
        |
final norm and LM head
        |
final-output sequence score and guardrails
```

Selection candidates use an immutable overlay, not repeated mutation of registered module tensors. The overlay manifest
identifies the local/current parent hash plus alternate packed trellis, bank selector, SU, and SV chunks. `QVQLinear`
executes `forward_with_payload` through the same decoder, transform, kernel, and epilogue contract as an installed
artifact while the registered baseline tensors remain untouched. Never score an unpublished dense staging matrix.
Before trusting this path, prove overlay output is bit-exact with installing the same serialized payload on every
supported target backend. Install and publish only the final confirmed winner.

Do not begin with three suffix replays for every decoder layer. Assemble the first-order changes across the entire
model into at most three rate-aware model-wide maps and evaluate those first:

```text
local/current immutable manifest
  ├─ global map A: smallest rate-keyed trust fraction
  ├─ global map B: medium trust fraction or shard-consensus mask
  └─ global map C: largest rate-keyed trust fraction
       └─ exact full-output replay under successive fidelity
            ├─ commit at most one global winner
            └─ split into layer/semantic coordinates only if
                 the global maps are rejected or gradient votes conflict
                 and the frozen forward-equivalent budget remains
```

This changes the first exact cost from `K` suffixes for every layer to at most `K` complete-model candidates. After a
global winner, optional coordinate refinement is conditional on that winner. Splitting order may use only the fixed
canonical order or final-output-derived disagreement/effect information; local or short-window quality cannot decide
which coordinate receives the remaining replay budget.

Selections are conditional. If `A` is the current accepted set and `c` is a candidate:

```text
delta(c | A) = L(A union {c}) - L(A).
```

Evaluate one immutable candidate overlay against the current accepted manifest and either advance the manifest pointer
or discard the overlay. Do not score every module independently against the original baseline because this discards
cross-module effects. The live baseline tensors are never mutated during proposal generation or replay selection.

Within one semantic coordinate, evaluate every alternative map from the same current model snapshot, select the best
exact full-output result, and commit at most one map. Across coordinates, the accepted model becomes the baseline for
the next coordinate. Use a fixed canonical coordinate order or one derived from full-output gradients. A second
sweep may run only when its predeclared selection-set stopping rule is satisfied. This prevents evaluation order among
mutually exclusive maps within one coordinate from changing the result. Cross-coordinate greedy order remains part
of the declared algorithm; another sweep may reduce, but cannot prove the absence of, order dependence.

Use a predeclared selection-only successive-fidelity schedule to bound exact replay cost without restoring local-metric
authority:

```text
all non-local proposals
  └─ exact replay on the small S shard after every map is frozen from P
       ├─ eliminate only maps proven materially inferior by a simultaneous paired bound
       └─ advance every unresolved map on new rows to medium fidelity
            ├─ one survivor: advance it
            ├─ two survivors: advance both
            └─ three unresolved survivors: advance all three
                 └─ full selection fidelity on new rows
```

The shard order, independent-sequence minima, token budgets, paired sampling-bound survivor rule, and stopping rule are
frozen before selection. Selection is adaptive tuning, so its intervals are algorithmic controls and receive no
confirmatory performance claim.
The sealed confirmation procedure is the only inferential gate. A token budget alone is insufficient because a few
long prompts do not provide the declared sequence-level evidence. Hash-stratify the first fidelity population by
domain and length before selection; never use a convenient dataset prefix whose composition was not frozen. The initial
schedule is 8,192/32, 32,768/64, and 65,536/128 valid tokens/independent sequences. These are cumulative populations:
cache structured per-sequence outputs and evaluate only newly opened rows at the next fidelity. Never rerun the first
shard merely to recompute the same aggregate. The baseline is likewise evaluated once per row and reused for every
same-shape candidate.

At those token budgets, exhaustive evaluation of three candidates costs `3*65,536 = 196,608` candidate-token lanes.
The accuracy-protecting low-rate path carries all three maps through medium unless a simultaneous bound eliminates
one. If one map then reaches full fidelity, candidate work is

```text
3*32,768 + 1*(65,536-32,768) = 131,072 candidate-token lanes.
```

Including the one cumulative baseline lane, total replay-token lanes are 196,608, still 25% below exhaustive replay's
262,144. If two maps remain unresolved after medium fidelity, carrying both through full fidelity costs

```text
3*32,768 + 2*(65,536-32,768) = 163,840 candidate-token lanes,
```

or 229,376 lanes including the baseline: still 12.5% below exhaustive replay. If all three remain unresolved, all
three reach full fidelity and the schedule intentionally becomes exhaustive rather than making an unsupported low-rate
prune. Earlier simultaneous eliminations reduce these costs. These are workload-count proofs, not wall-time claims;
measured suffix length, routing, transfer, and kernel utilization must validate realized speed.

Serial fixed-shape overlay replay is the reference and initial default. Evaluate the unchanged baseline once per
fidelity and reuse that exact score for all serial candidates under the same prompts, masks, kernel policy, and
accumulation contract. Candidate batching is optional only after a benchmark proves it faster and parity tests prove
the same winner. A batched implementation includes an unchanged same-shape baseline lane because batching may change
kernel selection or numerical order. Local MSE/KLD/top-k cannot enter this funnel. Numerical replay noise and
population-sampling uncertainty are separate: a bit-identical baseline gives a zero numerical-noise floor but does
not make a 32-sequence ordering reliable. Maps are frozen from `P` before `S` is opened. For every ordered candidate
pair, define `d_cj(s) = U_c(s) - U_j(s)`, so positive values mean candidate `c` is worse. Eliminate `c` only when a
frozen simultaneous paired confidence sequence or empirical-Bernstein lower bound establishes

```text
LCB_selection(mean(d_cj)) > Delta_select
```

for at least one retained `j`. `Delta_select`, the selection false-elimination target, and family-wise multiplicity
over every pair and interim look are frozen before selection. An empirical-Bernstein bound is legal only when the
paired utility difference has a predeclared finite range or clipping contract; otherwise use a paired fixed-look/
bootstrap or other selection bound whose assumptions are validated by P6. These are algorithmic selection bounds, not
confirmatory performance claims. At W1--W2.5, an unresolved third map advances rather than being discarded by rank.
W3+ may use the same bound for a faster path only after P6 evidence supports it. The unchanged baseline remains an
exact fallback. Confirmation remains sealed and is not a selection-fidelity stage.

The P6 false-elimination guarantee applies only to a candidate family whose exact maps and pair/look multiplicity are
frozen from `P` before any `S` result is visible. A coordinate, sibling path, tilted path, or alignment candidate made
after inspecting `S` is selection-adaptive even when its underlying action scores came from `P`. It may use an
independent, pre-reserved unopened `S` extension under a separately budgeted simultaneous bound, or it may reuse `S`
only as explicitly non-inferential tuning and rely on sealed `C` for the one publication decision. The MVP chooses the
second, cheaper policy for alignment and evidence-gated escalation; it cannot claim P6 exhaustive-winner parity for
those adaptive candidates, and `C` may reject the one frozen finalist but never choose a runner-up.

Cached per-sequence rows are reusable only under a fixed microbatch partition, padding policy, candidate lane shape,
kernel dispatch, dtype, routing policy, and reduction contract. If any of these changes between fidelities, either
prove per-sequence batch-partition invariance at the declared tolerance or recompute the affected prefix and charge it;
never merge numerically different execution populations silently.

Broad selection replay is teacher-forced because it supplies exact final-output NLL, coarsened distribution, top-k,
and margin metrics cheaply in one logits pass. Self-conditioned rollout is not equivalent after trajectories diverge, so it is
retained for the local root and at most one or two finalists rather than run for every preliminary map. Each rollout
candidate gets isolated KV, routing, and recurrent state and replays from the prompt. Never reuse baseline KV entries
produced by a changed module. Track EOS/stopping independently and require batched rollout to match fresh serial
rollout. The default policy is `finalists_only`; broad-map rollout is an explicit, charged ablation.

The exact replay evaluator must not receive local metrics. This separation should be enforced in the API: candidate
metadata may contain diagnostics, but the acceptance callback receives only full-output selection metrics from the
exact serialized-payload runtime.

### Stage 5: finalist alignment, rollout, and confirmation

Confirmation tests one frozen hypothesis. Complete the single default bank-selection sweep using selection prompts only.
Then generate and choose fixed-trellis output-alignment candidates using teacher-forced selection replay. Run the isolated
self-conditioned rollout, when configured, on the resulting complete finalist rather than on the pre-alignment map.
If alignment causes a rollout rejection, evaluate the already selection-valid unaligned finalist through one
predeclared conditional third rollout lane before spending on Tier C/D regeneration. Reuse the sealed local-root
rollout result; only the unaligned candidate needs the extra lane. If it also fails, restore the local root. The
expected rollout cost is `C_local + C_aligned + p_alignment_reject*C_unaligned`, while avoiding an escalation costing
`p_alignment_reject*C_escalation` whenever the unaligned candidate passes. Freeze all selectors/trellis/SU/SV
decisions before one paired confirmation decision. A predeclared two-stage confirmation may stop early for a statistical pass or
hard correctness failure, or expand an uncertain result to its maximum budget; its sequence partition, cumulative
alpha spending, numerical guardrails, and stop boundaries are fixed before proposal generation:

```text
selection-only sweeps
  └─ selection-only output-alignment decision
       └─ isolated local-root/finalist rollout
            └─ freeze complete candidate artifact
                 └─ one paired adaptive confirmation procedure
                      ├─ initial disjoint confirmation shard
                      │    ├─ alpha-spent pass or hard guardrail/non-finite failure: stop
                      │    └─ uncertain: open the predeclared expansion shard
                      ├─ pass: retain refined artifact
                      └─ fail, non-finite, or guardrail violation: retain exact local root
```

Full-model mode confirms final outputs; layer mode confirms the predeclared completed-layer boundaries. Do not use a
confirmation result to run another sweep, retune a threshold, retry alignment, choose a candidate, or change the
predeclared expansion rule. Such behavior converts confirmation into selection.

If the frozen refined artifact hash equals the local-root hash, finalize the local root without opening confirmation.
There is no changed hypothesis to test, and preserving the sealed rows is more useful than performing a no-op replay.

Run fixed-trellis output alignment only after selection-set bank/trellis selection. Treat it as another selection candidate:
its local closed-form loss cannot accept it below W3. Install it transactionally and apply the configured replay
horizon on selection prompts. The one final confirmation covers both bank selection and alignment because the optimal
SU/SV values depend on the selected decoded matrix.

### Normative selector algorithm

```text
if replay_mode != "none":
    assert_prompt_disjoint(proposal=P, selection=S, confirmation=C)
    capture_and_hash_dense_teacher_targets(P, S, sealed_C)
    reserve_complete_prompt_confirmation_population_and_budgets()

for each quantized module m:
    baseline_module = sequential_local_banked_solver(
        solver=block_ldlq_or_yaqa,
        ties="bank0_then_generation_order",
        whole_proxy_rollback="canonical_bank0",
    )
    baseline_module = pack_and_reconstruct_through_checkpoint_contract(baseline_module)
    validate_exact_module_artifact(baseline_module)
    append_copy_on_write_local_journal(baseline_module)
    install(baseline_module)

local_root = hash_and_freeze_complete_baseline_manifest()

if replay_mode == "none":
    serialize(local_root, bank_selection_provenance="local")
    return

reserve_budget_for_alignment_and_conditional_unaligned_rollout()

proposal_state = stream_one_live_quantized_forward_backward_sweep(
    dataset=P,
    proposal_surrogate=differentiable_final_output_proposal_surrogate,
    workspace=resolve_workspace_policy_before_sweep(
        include_compute_transfer_io_and_reserved_headroom=True,
    ),
    exact_reductions=("dense_gradient", "direct_action_scores", "host_xg_scoring"),
    approximate_reduction="two_sketch_generator_only_when_explicitly_enabled",
    optional_diagnostic_shards=diagnostic_shards,
    shard_overflow_policy="sequential_disjoint_shard_maps",
)

ranked_actions = []
for each module m in backward-consumption order:
    selector_actions = same_state_bank_alternatives(local_root[m])
    orbit_actions = exact_edge_xor_orbit_alternatives(
        local_root[m],
        rate_keyed_orbit_palette,
    )
    ranked_actions += finalize_transform_domain_action_scores(
        selector_actions + orbit_actions,
        proposal_state.consume(m),
    )

proposals = build_bounded_immutable_overlay_maps(
    ranked_actions,
    maximum_fractions=rate_aware_trust_fraction_caps,
    replace_middle_with_consensus_map_when=predeclared_shard_churn_rule,
    preserve_disjoint_shard_directions_when=proposal_state.used_sequential_shards,
    parent=local_root,
)

survivor = replay_successive_fidelity(
    proposals,
    dataset=S,
    baseline=local_root,
    cumulative_fidelities=(selection_initial_fidelity, selection_medium_fidelity, selection_max_fidelity),
    evaluate_new_rows_only=True,
    low_rate_survivors="eliminate_only_by_simultaneous_bound_else_advance_all",
    clear_winner_rule=predeclared_simultaneous_paired_sampling_bound,
    on_budget_exhaustion="last_fully_selection_validated_or_local_root",
)

unaligned_survivor = survivor
if survivor is not local_root and output_alignment_is_configured:
    aligned_overlay = generate_fixed_trellis_output_alignment_overlay(survivor)
    survivor = choose_unaligned_or_aligned_by_frozen_selection_comparator(
        survivor,
        aligned_overlay,
    )

if survivor is not local_root and selection_rollout_policy == "finalists_only":
    survivor = compare_local_and_complete_finalist_with_isolated_rollout(
        survivor,
        local_root,
        conditional_budgeted_unaligned_fallback=unaligned_survivor,
    )

ranked_coordinates = (
    choose_only_from_final_output_proposal_scores(ranked_actions)
    if any_escalation_tier_is_enabled()
    else ()
)
for escalation_tier in ("baseline_history_siblings", "propagation_tilted_viterbi"):
    if survivor is not local_root or not escalation_budget_allows(escalation_tier):
        continue
    escalation_actions = generate_verified_actions_on_demand(
        escalation_tier,
        ranked_coordinates,
    )
    escalation_candidate = repeat_bounded_successive_fidelity_selection(
        escalation_actions,
        parent=local_root,
        on_budget_exhaustion="last_fully_selection_validated_or_local_root",
    )
    escalation_candidate = choose_configured_alignment_on_selection(
        escalation_candidate,
        parent=local_root,
    )
    survivor = compare_local_and_complete_finalist_with_isolated_rollout(
        escalation_candidate,
        local_root,
    )

refined_artifact = freeze_overlay_manifest(survivor)

if refined_artifact.hash == local_root.hash:
    final_artifact = local_root
else:
    confirmation = run_predeclared_paired_adaptive_confirmation(
        local_root,
        refined_artifact,
    )
    if confirmation.passes:
        final_artifact = install_validate_and_publish_once(refined_artifact)
    else:
        final_artifact = local_root

if final_artifact is local_root:
    final_provenance = "local"
elif replay_mode == "layer":
    final_provenance = "layer_replay"
elif task_recovery_target_is_configured:
    final_provenance = "full_model_task_replay"
else:
    final_provenance = "full_model_distribution_replay"

serialize(final_artifact, bank_selection_provenance=final_provenance, local_baseline_hash=local_root.hash)
```

The second live-gradient sweep is disabled in the initial policy. A future sweep may be enabled only by a frozen
selection-only trigger after a meaningful accepted gain or selector-churn signal, and it repeats the same cheapest-tier
ordering. Curvature, layer boundaries, sibling streams, and tilted Viterbi are not silently inserted into the MVP.

The mode outcome is deliberately simple:

```text
saved local baseline
  ├─ replay_mode="none"
  │    └─ final artifact: local
  │
  ├─ replay_mode="layer"
  │    ├─ strict confirmed boundary improvement: layer-refined winner
  │    └─ otherwise: exact saved local baseline
  │
  └─ replay_mode="full_model"
       ├─ strict confirmed task/deployment-utility improvement: full-model-task-refined winner
       ├─ strict confirmed generic trajectory improvement: full-model-distribution-refined winner
       └─ otherwise or no changed artifact: exact saved local baseline
```

The sequential local solver can be Block-LDLQ or YAQA. Its score has authority only for constructing the saved local
fallback. It may not enter propagation map generation or a replay acceptance comparator except through the explicitly
configured normalized local path term used by propagation-tilted Viterbi. A candidate tie is not an improvement;
stable bank/candidate generation order and the installed baseline win all ties.

## Hierarchical replay coordinates

In full-model mode, begin with model-wide bounded maps and split only after exact final-output rejection or conflicting
full-output-gradient votes, while the frozen coordinate and forward-equivalent budgets remain:

```text
whole-model bounded maps
        |
        +-- accepted
        |     └─ optional budgeted residual refinement
        |
        +-- rejected
              |
              +-- whole decoder layer
                    |
                    +-- QKV + attention O
                    |
                    +-- MLP or MoE
                          |
                          +-- shared MLP
                          +-- router
                          +-- routed expert clusters
```

Attention Q/K/V/O are initially coupled because changing Q or K alters the attention distribution that acts on V.
If the combined candidate fails exact final-output replay, split QKV from O.

Dense gate/up remain coupled through the multiplicative activation. Test `gate + up + down` first, then split
`gate + up` from `down`. Do not promote gate and up independently from their linear outputs, and do not use their
local scores to decide whether the split is worthwhile.

## Large-MoE execution

A large MoE layer does not need to reside on one GPU. Use the model's actual device map and stream only active expert
clusters:

```text
cached layer or pre-MoE state
        |
router
        |
active expert cluster 0 -> owning device
active expert cluster 1 -> owning device
...
        |
aggregate routed contributions
        |
remaining model suffix
```

For an expert-weight-only candidate, routing is unchanged because the router saw the same input. Cache routed token
indices, routing weights, and the aggregate contribution from unchanged experts. Recompute only the candidate expert
contribution:

```text
Y_moe_candidate = Y_moe_baseline
                + route_weight * (expert_candidate(X_e) - expert_baseline(X_e)).
```

This replacement is exact for fixed routing. If the router or any earlier computation changes, discard the cached
assignments and reroute. Do not refine experts that receive no proposal tokens or no replay-selection tokens unless a
separately declared routing-development population supplies the missing role; retain the baseline and report missing
coverage. Confirmation routing remains sealed until
the artifact is frozen. It may reject the complete artifact for a predeclared coverage failure, but it may not change
which experts are refined. Use additional routing-coverage development prompts before making a production default.

Do not fuse individual expert gate/up weights merely to make replay convenient; that duplicates packed tensors and
can cause OOM. Group only compatible active experts for dispatch.

## Memory hierarchy

The intended peak GPU residency is:

```text
live quantized model under its normal device map
+ one activation microbatch
+ one bounded proposal-scoring chunk for the selected workspace policy
+ compact active-module action map
+ optional materialized active-module candidate scratch only on the reference path
+ required suffix workspace
```

Use the following placement:

| Object | Preferred placement |
|---|---|
| Live baseline quantized model | Normal sharding/offload device map. |
| Dense `G_Q` accumulators | Accelerator then CPU only when their complete cross-microbatch set fits the declared cap. |
| Direct action-score accumulators | Bounded accelerator chunks with asynchronous CPU/NVMe persistence. |
| Host x/g scorer | Transformed activations/adjoints transferred into NUMA-local CPU scoring and accumulation. |
| Independent proposal sketches | Accelerator while updating, then CPU; explicit approximate-generator mode only. |
| Compact selector/orbit maps | CPU memory or NVMe, content-hashed once. |
| Compact-overlay decoder map | Owning accelerator while its module executes; no transformed-trellis write. |
| Reference candidate trellis scratch | Owning accelerator only; charge every rematerialization. |
| Tier C/D sibling streams | Active module only; bounded CPU/NVMe if retained. |
| Optional layer boundaries | Reuse output-alignment spool/recompute support. |
| Teacher sketches | CPU or NVMe. |
| Regeneration source/factors | One verified dense module and its reproducible factors at a time. |

Report map storage and generator peak separately. For a two-trillion-weight model:

| Proposal object | Exact density | Approximate decimal storage |
|---|---:|---:|
| FP32 dense `G_Q` | 4 bytes/weight | 8 TB |
| Sixteen FP32 action scores per 256-weight tile and accumulator | 0.25 bytes/weight | 500 GB |
| Retained FP32 gain plus four-bit action per tile | 4.5 bytes/tile | 35.16 GB |
| One uint64 global tile index | 8 bytes/tile | 62.5 GB |
| Final shared action plus two inclusion masks | 6 bits/tile | 5.86 GB |
| Sequential two-shard actions plus three inclusion masks | 11 bits/tile | 10.74 GB |

The last two rows are alternative compact-map costs, not additive production payloads or generator peaks. The 11-bit
layout is permitted only for the memory-bounded sequential-shard proposal fallback and remains temporary; the
confirmed artifact still serializes the ordinary two-bit bank selector. Do not retain a global uint64 sort index by default.
Resolve rate/role-specific thresholds with a deterministic streamed histogram/quantile pass, or perform a second
bounded score read, and specify tie handling. Exact replay tolerates an approximate ranking threshold; it does not
tolerate an action score or selector being silently changed after the manifest is hashed.

Every simultaneously live diagnostic shard multiplies the dense-gradient or direct-score accumulator term. Do not
retain a separate aggregate when shard accumulators exist; derive the aggregate by a chunked deterministic sum after
the shards close. Resolve shard count under the same cap. If two exact shard accumulators do not fit, use the
sequential disjoint-shard policy described above: one accumulator is reused, each shard is compacted before release,
and its independently derived action direction remains available to exact replay. Do not claim an all-row aggregate
in that mode, and never substitute an unrecorded lower-precision shard vote merely to eliminate a candidate early.

For a two-trillion-weight model, one 500 GB direct-score accumulator plus the 35.16 GB retained-action stream and the
5.86 GB shared-map form already consume about 541.02 GB decimal. A nominal 512 GiB cap is about 549.76 GB, leaving
only 8.74 GB before temporary chunks, hashes, manifests, filesystem reserve, or a second action stream. Therefore the
auto resolver subtracts an explicit reserve before admission and rejects any policy whose complete live-and-
persistent set does not fit. It must also evaluate the transfer equations above; free disk capacity is not a speed or
feasibility proof.

The local root and every selection proposal are immutable manifests. A proposal contains the parent hash and compact
per-module selector/orbit or trellis deltas; it is not another resident checkpoint. Resolve and validate each immutable
chunk once and cache by content hash. Stream the compact map to the quantization-only overlay decoder, or explicitly charge
the reference path that resolves it into active-module scratch. The ordinary registered model state remains the local
root until confirmation passes, eliminating per-candidate undo journals, cache invalidation, and mutation locks.

Tier C/D materialization is charged against a global scratch cap. If a full candidate cannot be evaluated by streaming
one active module at a time within that cap, reject it before replay. Prefetch may overlap checked immutable chunks,
but correctness cannot depend on an unchecked transfer. One model-level exclusive lock is required only for the final
install, validation, and publication. This is the complete initial free-threaded/GIL-free safety model; a generic
multi-version mutable DAG and per-module lock hierarchy are unnecessary for MVP selection.

## Compute budget

The user-visible modes have intentionally different costs and evidence strength:

| Replay mode | Dense teacher capture | Default candidates | Replay horizon | Artifact provenance |
|---|---:|---:|---|---|
| `none` | No | No | None | `local` |
| `layer` | Yes | Compact maps | Completed owning layer | `layer_replay` |
| `full_model` | Disjoint P/S/C; YAQA-P and P=S reuse are explicit fast modes | Compact maps | Final outputs | Distribution- or task-replay provenance. |

All three modes pay for the sequential banked local solver needed to construct the fallback. Call it local, not free:
it evaluates four banks and a canonical rollback oracle, although native batching can amortize launches. Only replay
modes pay for teacher targets, backward sensitivity, compact candidates, exact replay, and confirmation. Full trellis
spooling and Viterbi regeneration are escalation costs, not base replay costs.

If every module candidate were replayed through the full suffix, a model with `L` layers, `G` groups per layer, and
`K` candidates per group would cost approximately

```text
K * G * (L + 1) / 2
```

full-forward equivalents per sweep. This is not the production plan.

The accuracy-first default keeps proposal `P` disjoint from YAQA and replay-selection `S` disjoint from both. An
explicit low-cost mode may fuse YAQA and `P` teacher capture only when prompt IDs, rendering, masks, seeds,
temperatures, evaluation mode, and target bytes match exactly. A separate diagnostic mode may set `P == S` under the
restricted contract above. Each reuse records the overlap and cannot claim the displaced strict-split evidence. The
staged plan spends:

1. dense targets over disjoint `P`, `S`, and sealed `C` populations, streamed during one dense-model residency when
   possible; strict `P != S` adds only the `S` dense-target work relative to row reuse;
2. existing partial-hook baseline captures;
3. one microbatched live-quantized forward/backward sweep on `P` using the pre-resolved proposal workspace;
4. selector and edge-XOR action scoring without Viterbi;
5. one baseline plus three serial candidate overlays at small `S` fidelity;
6. every unresolved candidate on only the new medium/full `S` rows under the frozen simultaneous paired bound;
7. optional alignment followed by isolated rollout for the local root and complete aligned finalist, plus one
   conditional unaligned lane only after aligned rejection;
8. one predeclared paired adaptive confirmation procedure after all selection decisions are frozen.

Only after those steps show no useful gain may the run spend its frozen escalation budget on active-coordinate sibling
paths, propagation-tilted Viterbi, a second live-gradient sweep, curvature, or layer-boundary replay. Each adds its
declared Viterbi, HVP/VJP, teacher, or suffix cost. Report compute and storage separately; no escalation is silently
substituted for the default.

Cache exact per-sequence baseline scores on `S` and extend them with new rows at each fidelity rather than recomputing an
overlapping prefix, subject to the fixed-execution or batch-partition-invariance contract. The target is three rate-
aware model-wide maps from one proposal sweep, optionally substituting disjoint-shard directions, followed by every
still-unresolved full-fidelity survivor according to the paired bound. Exact rejection or conflicting full-output-gradient shard votes,
rather than local reconstruction behavior, may trigger finer attention/MLP/expert work.
Record actual forward-layer equivalents, backward-layer equivalents, HVP/VJP passes, tokens, active experts, bytes
spooled/regenerated/transferred, and wall time instead of reporting only a model-level duration.

Reserve the complete sealed-confirmation budget before proposal generation. Then cap proposal forward/backward-layer
equivalents, selection replay tokens/forward-layer equivalents, rollout work, alignment work, orbit materialization
traffic, scratch/spool bytes, and refined coordinates independently. Exhausting a selection cap returns the **last
fully selection-validated artifact**, or the exact
local root when none exists; it never confirms an in-progress small-fidelity candidate. Confirmation has its own
untouchable reserve and cannot be consumed by proposal or selection escalation.

For example, `L=80, K=3` layerwise suffix replay is about `121.5` full-forward equivalents per sweep even before
backward work, so candidate batching alone does not make the unbounded layerwise design viable. Count every baseline
and candidate lane, newly opened fidelity fraction, teacher query, rerun after rollout divergence, and alignment
evaluation; batching changes elapsed time but not charged FLOPs. Raw replay-call count remains a safety check, not the
primary budget, because an 8K replay and a 65K replay do not cost the same. `maximum_sweeps=1` is the initial default.

The initial safety counts follow directly from the schedule. Three incremental baseline lanes plus `3 + 3 + 1`
candidate lanes reserve ten exact selection replay calls when one map reaches full fidelity. Two full-fidelity maps
reserve eleven calls; all three reserve twelve rather than forcing an unsupported prune. The cached unaligned finalist leaves one alignment lane;
local-root plus complete aligned finalist use two rollout lanes, while one conditional unaligned fallback reserves a
third without consuming it when alignment passes. Two artifacts over at most two confirmation looks reserve four
lanes. These counts supplement, but never replace, token and layer-equivalent accounting.

The differentiable Python QVQ reference is an initial small-model correctness oracle, not a scalable 2T-model replay
engine. Full-model replay on large models requires a bounded input-VJP path for every installed QVQLinear (or verified
activation recomputation with one decoded module at a time), plus checkpointing/offload accounting. Native VJP or an
equivalent bounded reference implementation must precede large-model full-output replay, even if forward inference
kernels are already optimized.

This is an intentional compute trade: the generator compresses `4 ** tile_count` possibilities into a bounded map
set using a final-output-derived gradient. Replacing exact replay with local pruning would be cheaper, but it would
remove the primary accuracy mechanism V4 banks were added to provide.

## Acceptance metrics

Maintain two metric namespaces with a hard authority boundary.

Diagnostic-only local metrics:

- weight MSE, relative L2, SQNR, and Hessian/YAQA proxy;
- module and layer activation MSE/cosine;
- module or partial-layer KLD;
- module or partial-layer top-1/top-5 agreement;
- bank occupancy, selector churn, and residual-direction diversity.

These metrics explain behavior and validate arithmetic. Local weighted error may choose only the mandatory fallback.
Below W3, no local metric may enter a replay-refinement comparator, including as a non-inferiority guardrail. Exact
completed-layer metrics may enter only the explicitly labelled `layer` replay comparator.

Return a structured full-output score rather than treating mean final KLD as sufficient. Separate differentiable
proposal signals from exact acceptance authority:

- deterministic coarsened top-k-plus-rest KL for gradients, proposal ranking, and replay diagnostics;
- coarsened top-k-plus-rest JSD;
- teacher top-1 margin loss;
- top-1 disagreement;
- top-5 overlap loss;
- non-finite count;
- natural next-token NLL and a predeclared **teacher-forced** task/deployment loss where available;
- finalist-only self-conditioned generated-task loss and rollout disagreement on frozen selection and confirmation subsets;
- final hidden-state and final-logit range/non-finite checks.

Compute coarsened KL, coarsened JSD, natural NLL, top-1/top-5, critical margin, range, and non-finite counts from one
candidate logits pass. These are metric reductions over the same outputs, not separate model replays.

Freeze the comparator before teacher capture. Broad teacher-forced selection uses a lower-is-better per-sequence
utility, defined before proposal generation as

```text
U_teacher_forced_sequence
    = w_task_tf * standardized_teacher_forced_task_loss
      + w_nll * standardized_natural_token_NLL
      + w_margin * standardized_critical_margin_hinge.
```

Use these exact component contracts initially:

```text
teacher_forced_task_loss(sequence):
    predeclared lower-is-better loss available from the same forced-logits pass,
    such as multiple-choice negative log-likelihood or answer-token NLL

generated_task_loss(sequence):
    predeclared lower-is-better self-conditioned rollout loss, such as 1 - exact_match;
    parser, target, generation settings, and isolated-state contract are frozen before teacher capture;
    unavailable during broad teacher-forced selection

rollout_disagreement(sequence):
    normalized Levenshtein distance defined in Compact dense-teacher targets

natural_token_NLL(sequence):
    mean_valid_token(-log_softmax(candidate_logits)[natural_next_token])

critical_margin_hinge(sequence):
    sum_t critical_weight[t] * relu(dense_margin[t] - candidate_margin[t])
    / max(1, sum_t critical_weight[t])

candidate_margin[t]:
    candidate_logit[t, dense_top1_id[t]]
    - max_(v != dense_top1_id[t]) candidate_logit[t, v].
```

`critical_weight` is a frozen nonnegative mask/weight derived only from dense proposal metadata, such as a predeclared
dense-margin band or task-token mask. Record the band and require at least one weighted token per eligible sequence;
otherwise the margin component is unavailable for that sequence population.

For component `j`, standardize with immutable constants

```text
standardized_L_j = (L_j - center_j) / scale_j
scale_j = max(1.4826 * MAD_dense_proposal(L_j), declared_absolute_floor_j).
```

The center cancels in paired comparisons but is retained for reproducibility. A task component with a degenerate
dense reference uses an explicit positive absolute scale, normally one for losses already bounded in `[0, 1]`. Record
every center, scale, floor, reduction, and eligibility mask. Convert every component to a lower-is-better loss and
require finite, nonzero scales.

Every nonzero-weight broad-selection component must exist for the same declared primary sequence population. Missing
components have an explicit zero weight before selection; never drop rows, renormalize weights, or substitute a
population after observing results. Autoregressive exact match is never synthesized from teacher-forced logits.
Rollout is a separate `FinalistReplayScore` over a frozen subset or the complete primary population; compare the local
root and the same one or two finalists on identical sequence IDs. When ground-truth/generated-task targets exist,
that frozen task loss may rank finalists. Dense-trajectory Levenshtein disagreement is then a material catastrophic-
drift guardrail, not a primary ranker: a candidate that corrects a dense-model mistake must not lose solely because it
differs from the dense trajectory. Without a generated-task target, rollout may reject catastrophic drift or choose a
distribution-refined finalist, but it cannot claim task recovery. Rollout may not retroactively rescore preliminary
maps that never ran rollout.
Distribution metrics remain proposal signals and non-inferiority guardrails rather than the sole primary. When no
task/deployment-like component is supplied, record the result as
`full_model_distribution_replay`; it is not evidence of task recovery even though it reaches final outputs.

The comparator has one predeclared primary utility and separately declared guardrails; it is not an informal vote
across whichever metrics happen to improve:

```text
full-model primary:
    paired per-sequence U_sequence

layer primary:
    paired normalized FP32 hidden-state error at declared boundaries

selection root-survivor rule (descriptive tuning only):
    mean_sequence(candidate - baseline)
        < -Delta_selection_root

confirmation promotion rule:
    upper_confidence_bound_sequence(candidate - baseline)
        < -Delta_min

hard correctness:
    zero new non-finite values
    valid serialized payload, legal selectors, and declared finite output range

material non-inferiority guardrails:
    every predeclared final-output coarsened-KL/coarsened-JSD/top-1/top-5/rollout bound passes
    every predeclared cosine/norm/p95/p99 bound passes in layer mode
    optional deployment-required minimum independent-sequence sign agreement passes

tie, unavailable required metric, or failed guardrail:
    baseline wins
```

Keep three distinct scales with distinct authority:

```text
eta_num          = replay numerical-equivalence/noise floor
Delta_select     = material pairwise gap required to eliminate a selection candidate
Delta_min        = minimum material improvement required for confirmation promotion

eta_num <= Delta_selection_root <= Delta_min
```

`Delta_selection_root` defaults to `Delta_min`; a smaller frozen value may retain a promising finalist but cannot weaken
the confirmation effect target. `eta_num` can be zero on a bit-exact backend and never substitutes for a material
effect. The selection and confirmation bounds additionally include their sampling uncertainty.

Confirmation uncertainty is over independent sequences, not tokens. Token losses within one prompt are clustered.
Use a paired sequence bootstrap, declared Studentized interval, or predeclared paired group-sequential procedure; four
aggregate shards alone do not justify a normal confidence claim. Selection may record the same intervals as diagnostics
but cannot assign nominal coverage after adaptive map selection. Specify sequence/token weighting, masked positions,
confidence level, cumulative alpha spending, multiplicity handling, absolute floors, and every non-inferiority
tolerance in configuration/provenance.
Do not retune them after seeing confirmation. Full-model quality fields come from the complete quantized model at
final output. Layer mode uses only mathematically defined hidden-state metrics at completed boundaries and preserves
lower-authority provenance.

When explicitly required by a deployment contract, `minimum_sequence_agreement` means the fraction of primary sequence
IDs for which paired `U_candidate - U_baseline < 0`; an exact tie is not agreement. It is a stability guardrail in
addition to, not a replacement for, the paired interval. It is disabled by default: an arbitrary 75% majority can
reject a large concentrated rescue or reward many negligible changes. Do not reinterpret diagnostic shard votes as
independent sequence agreement.

An empty guardrail dictionary is valid only for a diagnostic prototype and cannot promote a production artifact.
The production rule always enforces hard correctness and the primary paired utility. Secondary final-output metrics
use predeclared **material** non-inferiority tolerances rather than six default zero-tolerance vetoes. NLL or critical
margin already present in the primary utility is not duplicated as an independent hard guardrail unless a deployment
contract explicitly requires it. Coarsened KL, coarsened JSD, top-1 disagreement, top-5 overlap loss, and rollout
disagreement remain valuable safety diagnostics, but earlier low-rate evidence shows that individually improved
distribution metrics can disagree with task recovery. Their tolerances must combine a selection-only numerical-noise
floor with a separately declared materiality floor; confirmation may not estimate either value.

When multiple secondary guardrails are inferential, use a predeclared simultaneous paired-bootstrap envelope or
otherwise spend the family-wise error budget. Do not apply six uncorrected one-sided tests and call the family a 95%
gate. Compact top-k-plus-rest targets cannot produce exact vocabulary JSD; calling this metric `jsd` without the
`coarsened_` qualifier is a contract error. A zero tolerance remains available only when explicitly justified, not as
the low-rate default. Guardrail tolerances and the required metric set are frozen before teacher capture.

Before opening confirmation, use selection-only paired variance and a predeclared minimum detectable material effect to
freeze the maximum confirmation population. Because the selected replay winner can have an optimistically small
variance estimate, use a conservative planning value: the upper confidence bound or maximum frozen variance across
eligible finalists, selection shards, and declared propagation seeds. The initial 32-sequence shard may stop for a
statistical pass or hard correctness failure; an uncertain result opens only the declared additional rows, commonly
up to 128 and up to 256 or 512 when measured variance requires it. The matching maximum-token population is the exact
sum over those complete pre-enumerated prompts, not a fixed 65,536-token cap that silently truncates them. Cache first-
stage per-sequence results and never rerun or reinterpret them.

Before selection, replay the unchanged installed local root twice through the same fixed-shape path. Two bit-identical
repeats establish the deterministic fast path. If they differ and nondeterministic replay is allowed, run at least the
configured larger repeat count on selection-only instrumentation rows and freeze a conservative noise quantile as
`eta_num`. Two unequal observations cannot estimate a noise distribution. Confirmation data may not set or update the
floor. Record all repeat deltas, backend determinism settings, and the resolved floor.

Mean final KLD remains insufficient by itself: it averages the supplied token distribution, can hide rare decisive
logits, and does not measure autoregressive trajectory stability. Retain raw paired per-token/per-sequence outputs
needed to explain candidate disagreements. A self-conditioned rollout can be a finalist-selection or confirmation
guardrail only when the dense rollout trajectory and scoring contract were captured before proposal generation and its threshold
was frozen. Never add or retune a rollout gate after inspecting confirmation.

Final ARC, GSM8K, STEM, History, MMLU, or similar tasks are evaluated only after the bank-selection policy and
thresholds are frozen. If their results decide whether replay becomes default at a rate, they become default-policy
development evidence; reserve another untouched suite/split for confirmatory performance reporting. Report
correct-to-wrong and wrong-to-correct flips plus paired uncertainty, not only net score.

## Proposed configuration

The public configuration should be nested rather than adding more unrelated fields directly to `QVQConfig`:

```python
PropagationConfig(
    replay_mode="full_model",  # "none", "layer", or "full_model"
    replay_dataset=None,  # optional unsplit P/S/C source; never inferred from ordinary calibration
    proposal_dataset=None,
    selection_dataset=None,
    confirmation_dataset=None,
    proposal_valid_tokens=65_536,
    proposal_minimum_sequences=128,
    reuse_yaqa_proposal_teacher_pass=False,  # explicit fast mode only after exact contract match
    reuse_proposal_rows_for_selection=False,  # diagnostic only; disables P6 coverage and early statistical pruning
    selection_fidelity_valid_tokens=(8_192, 32_768, 65_536),
    selection_fidelity_minimum_sequences=(32, 64, 128),
    selection_evaluate_new_rows_only=True,
    selection_survivor_rule="simultaneous_paired_bound_else_advance_all",
    selection_false_elimination_probability=None,  # required P6 family-wise value, frozen before selection
    selection_material_gap=None,  # Delta_select, required P6 value
    selection_sampling_bound_method=None,  # required P6 method and tail/boundedness contract
    confirmation_initial_valid_tokens=16_384,
    confirmation_initial_minimum_sequences=32,
    confirmation_reference_valid_tokens=65_536,  # planning reference, not a cap on a 256/512-row look
    confirmation_reference_minimum_sequences=128,
    confirmation_maximum_sequences=512,  # freeze from selection-only variance/MDE before opening confirmation
    confirmation_maximum_valid_tokens=None,  # exact complete-prompt total for the sealed maximum reserve
    confirmation_variance_planning="upper_bound_or_max_across_finalists_shards_and_seeds",
    confirmation_repeated_look_method="paired_alpha_spending",
    confirmation_alpha_spending=(0.01, 0.05),  # cumulative alpha at initial and maximum looks
    confirmation_early_stop="alpha_spent_pass_or_hard_correctness_failure",
    teacher_samples_per_token=0,
    teacher_topk=32,
    teacher_temperatures=(1.0,),
    diagnostic_shards="auto_up_to_2",
    missing_diagnostic_shards="sequential_disjoint_shard_maps",
    candidate_path_tiers=("selector_only", "edge_xor_orbit"),
    escalation_path_tiers=(),  # research extension may enable siblings, then tilted Viterbi
    rate_keyed_orbit_candidates=4,
    bank_orbit_palette_design="joint_per_rate",
    global_map_fractions_by_rate={
        "1": (0.05, 0.15, 0.35),
        "1.5": (0.05, 0.15, 0.35),
        "2": (0.10, 0.30, 0.60),
        "2.5": (0.25, 0.50, 1.00),
        "3": (0.25, 0.50, 1.00),
        "3.5": (0.25, 0.50, 1.00),  # control/opt-in only
        "4": (0.25, 0.50, 1.00),  # control/opt-in only
    },  # initial development hypotheses; freeze and validate separately for every rate
    shard_churn_candidate_policy="replace_middle_with_consensus_mask",
    exact_replay_candidates=3,
    candidate_execution="compact_overlay_decode",
    candidate_execution_reference="materialized_active_module_scratch",
    compact_map_layout="shared_6bit_or_sequential_shard_11bit",
    candidate_batching="serial",  # "auto_benchmark" only after parity and measured benefit
    proposal_surrogate_weights={
        "coarsened_kl": 0.50,
        "natural_nll": 0.25,
        "soft_margin": 0.25,
    },
    proposal_scale_method="dense_proposal_mad",
    proposal_scale_floors={
        "coarsened_kl": 1e-3,
        "natural_nll": 1e-3,
        "soft_margin": 1e-3,
    },
    proposal_margin_temperature=1.0,
    proposal_workspace_policy="auto_exact",  # dense G_Q, device scores, or host x/g scoring
    proposal_workspace_max_bytes=64 * 1024**3,
    proposal_host_spool="bounded_cpu_or_nvme",
    proposal_host_spool_max_bytes=512 * 1024**3,
    proposal_host_spool_reserved_fraction=0.10,
    proposal_sketch_policy="disabled",  # explicit two-independent-sketch research fallback only
    proposal_sketch_exact_shortlist_actions=2,
    proposal_global_ranking="deterministic_streamed_quantiles",
    selection_rollout_policy="finalists_only",
    selection_rollout_finalists=1,
    selection_rollout_sequences=32,
    confirmation_rollout_sequences=128,
    maximum_sweeps=1,
    maximum_proposal_forward_layer_equivalents=2.0,
    maximum_selection_forward_layer_equivalents=16.0,
    maximum_proposal_backward_layer_equivalents=2.0,
    maximum_exact_selection_replays=12,  # baseline plus all three maps at all three fidelities
    maximum_alignment_replays=1,
    maximum_rollout_replays=3,  # third lane is consumed only after aligned-finalist rejection
    confirmation_replay_reserve=4,
    budget_exhaustion_fallback="last_fully_selection_validated_or_local_root",
    maximum_refined_coordinates=2,
    candidate_scratch_max_bytes=64 * 1024**3,
    full_model_primary_loss="sequence_utility",
    layer_primary_loss="normalized_hidden_error",
    layer_boundary_weighting="equal_after_dense_mad_standardization",
    selection_sequence_utility_weights={
        "teacher_forced_task": 0.0,  # only a loss available from the same forced-logits pass
        "natural_nll": 0.50,
        "critical_margin": 0.50,
    },
    selection_sequence_utility_scale_floors={
        "teacher_forced_task": 1.0,
        "natural_nll": 1e-3,
        "critical_margin": 1e-3,
    },
    finalist_generated_task_loss=None,  # e.g. exact match; requires isolated autoregressive rollout
    critical_margin_policy={"type": "dense_margin_quantile", "quantile": 0.25},
    baseline_determinism_repeats=2,
    nondeterministic_noise_repeats=8,
    numerical_replay_noise_floor=None,  # eta_num, resolved from selection-only baseline repeats
    selection_root_material_gap=None,  # Delta_selection_root; eta_num <= value <= Delta_min
    confirmation_minimum_material_improvement=None,  # Delta_min, required P10 value
    minimum_sequence_agreement=None,  # deployment-specific, disabled by default
    required_guardrails=(
        "coarsened_kl",
        "coarsened_jsd",
        "top1",
        "top5",
        "rollout_disagreement",
    ),
    guardrail_material_tolerances=None,  # production requires a frozen, nonempty deployment mapping
    guardrail_tolerance_method="max_selection_noise_and_predeclared_materiality",
    guardrail_multiplicity="simultaneous_paired_bootstrap",
    proof_ledger_required=True,
    seed=0,
)
```

This is a reference-research schema, not a declaration that every displayed value is production-ready. In particular,
the rate fractions, adaptive survivor rule, material guardrails, workspace thresholds, and confirmation maximum remain
blocked by P3/P5/P6/P10. Production configuration must resolve them from completed proof-record artifacts and fail
closed when the required status or evidence hash is missing.

Keep expensive experiments out of the initial public lifecycle. A separate research-only extension may enable a
second gradient sweep, curvature/HVP probes, layer-boundary instrumentation, wider sibling paths, or normalized
propagation-tilted Viterbi. Each extension declares its own extra forward/backward/Viterbi budget and cannot change
the authority boundary or read confirmation data.

Recommended initial policy:

- always generate, validate, and durably snapshot the deterministic local baseline before optional replay;
- remove the ambiguous legacy `propagated_bank_selection` control; any retained module-output gate is explicitly
  `local_output_gate`, default-off, and forbidden in full-model replay;
- implement the exact auto-selected dense-gradient/device-score/host-xg workspace with complete model compute,
  transfer, I/O, NUMA, headroom, and sequential-shard accounting before sibling streams, tilted Viterbi, sketches, or
  curvature;
- enable propagation-tilted Viterbi only after verified dense-source/factor regeneration passes exact tests;
- keep proposal-surrogate gradients and exact replay acceptance in separate API types;
- reject a nonzero sequence-utility weight when its required target is unavailable; never silently renormalize weights;
- label a run without a task/deployment component as distribution-refined rather than task-recovered;
- use `full_model` replay for maximum-recovery QVQ V4 below W3 once its lifecycle passes validation gates;
- support `none` as the fast path and defer `layer` to an explicitly lower-authority, cost-constrained tier;
- treat W3 as the transition/control rate until full-model data justifies the same default;
- support both Block-LDLQ and YAQA as intra-bank path solvers;
- allow those methods to choose the local fallback, never a replay-refined winner;
- keep local metrics out of replay acceptance below W3 as a code invariant, not a public tuning option;
- `replay_mode="none"` requires no propagation dataset and labels the artifact as locally selected;
- no silent calibration-row fallback;
- evaluate model-wide bounded maps before spending replay budget on individual layers or semantic groups;
- stop refinement at frozen forward/backward/replay/coordinate caps and restore the last fully selection-validated
  artifact, or the exact local root; never confirm a partially evaluated candidate;
- use one proposal sweep on `P` by default; exact replay, output alignment, aligned-finalist rollout, and the
  conditional unaligned fallback use disjoint `S`, followed by one paired adaptive confirmation on sealed `C`;
- freeze proposal/selection budgets, simultaneous paired survivor rules, and the joint complete-prompt confirmation
  reserve before candidate generation;
- use compact immutable selector/orbit maps and a quantization-only overlay decoder; charge reference trellis
  rematerialization, and keep full sibling streams escalation-only;
- preserve the original local baseline as an exact root fallback throughout replay.

## Transaction and serialization contract

Each candidate must record:

- module and semantic-group identity;
- rate, vector size, bank count, codebook version, and shape;
- immutable parent hash and candidate tier;
- compact selector/orbit action map or packed Tier C/D trellis delta, plus SU/SV deltas when applicable;
- configuration and dataset hashes;
- candidate-generation method and seed;
- source-weight, Hessian/factor, RHT, scale, solver-order, and regeneration fingerprints;
- proposal-workspace policy, exact reduction or sketch/shortlist contract, normalization, simultaneous or sequential
  shard schedule, accumulation dtype, peak/persistent/reserved bytes, D2H/host/NVMe bytes, and predicted/measured time;
- sequence-utility definition, normalization constants, confidence method, and noise-floor evidence;
- selection-fidelity stage, sequence/token budget, candidate-survivor rule, and candidate-batch order;
- selection metrics; local/boundary diagnostics remain in a non-authoritative namespace, and confirmation metrics belong
  only to the two frozen final artifacts;
- a hash covering tensors and metadata;
- completed proof-record IDs, every conjunctive evidence-column status, evidence hashes, and the exact design revision
  they validate.

The local baseline transaction must additionally record a root manifest over every module artifact, quantization
configuration, target-module census, ignored dense tensor fingerprints, tied-storage/alias contract, shard index,
`bank_selection_provenance="local"`, deterministic tie-break contract, and exact local objective. These fields make
the root a reconstructible checkpoint, not merely a collection of module buffers. Refined artifacts record
`replay_mode`, replay horizon, proposal/selection/confirmation hashes, parent journal entry, comparator contract, storage policy,
and the local root hash. Final provenance is one of `local`, `layer_replay`, `full_model_distribution_replay`, or
`full_model_task_replay`; a confirmation rollback must serialize `local`, not the requested replay mode. This
distinguishes the authority of each checkpoint without changing inference tensors.

Selection overlay execution must:

1. verify the immutable parent and chunk hashes;
2. prefer the compact-overlay decoder, which applies orbit and bank actions without writing a transformed trellis;
3. otherwise resolve compact actions into one active-module serialized scratch and charge every rematerialization;
4. validate shape, dtype, finiteness, rate, recurrence, and reconstruction before execution;
5. run through the exact QVQ decoder/kernel/epilogue without changing registered tensors or caches;
6. release the compact map or scratch after synchronization;
7. record recoverable candidate failure and continue, while invariant/source/hash failure aborts refinement to the
   untouched local root.

No rejected candidate mutates the checkpoint or becomes visible to concurrent inference. Selection is read-only over the
registered local root, so rollback is normally discarding an overlay manifest. Before enabling an optimized overlay
backend, compare it bit-for-bit with temporarily installing the same serialized payload in an isolated test module.

Final publication is the only mutation transaction. Acquire one model-level exclusive lock, materialize and validate
the complete winner on every owning device, invalidate affected caches, atomically publish the new manifest, save, and
reload-validate. On any failure, reinstall the durable local root before releasing the lock. Keep that root until the
checkpoint has been atomically written and reload-validated. The production checkpoint stores only the winner plus
baseline provenance/hash, so production EBPW and inference VRAM remain unchanged. An optional debug sidecar may retain
the full local artifact; it is never loaded for inference.

## Pre-implementation mathematical proof ledger

No normative step may enter optimized implementation merely because it sounds cheaper or improves a local proxy. Each
step requires a versioned `ProofRecord` in this document or a linked result artifact before implementation begins:

```text
ProofRecord
  id, revision, owner
  algebraic_status, numerical_status, measured_efficiency_status, held_out_accuracy_status
  purpose: correctness | efficiency | accuracy | mixed
  exact baseline and proposed transformation
  algebraic invariant or stated approximation
  operation, transfer, persistent-byte, and peak-byte equations
  finite-precision error contract
  end-to-end target and paired measurement population
  pass/reject/rollback rule
  evidence command, artifact hash, and commit
```

There are four distinct evidence classes:

1. **Algebraic proof** establishes recurrence legality, tensor shapes, linear equivalence, storage bounds, or unchanged
   inference operations.
2. **Numerical proof** bounds floating-point deviation against an FP64/reference oracle under declared shapes, dtypes,
   reduction order, and condition range.
3. **Measured efficiency evidence** validates the predicted operation, transfer, peak-memory, and persistent-storage
   model on each target execution class. A smaller FLOP or byte expression is not a latency proof.
4. **Held-out statistical evidence** is required for accuracy claims. No algebra can prove that a bank map improves a
   nonlinear autoregressive benchmark on an unknown population. The mathematical obligation is instead a frozen
   paired hypothesis, effect target, sample-size/power calculation, and error-controlled decision rule.

A cheap reference-only oracle may be written to discharge a numerical or held-out proof obligation. Production
lifecycle, native-kernel, and scale-out work does not begin until that reference evidence passes. A failed obligation
records the arm as rejected; it is not weakened after results are observed.

Readiness is the conjunction of every evidence class required by a step, never the strongest or maximum status:

```text
ready(step) = all(required_status(step, evidence_class) == "passed")

reference-only implementation:
    required algebraic obligations passed

optimized implementation:
    algebraic + numerical + measured-efficiency obligations passed where applicable

default/production promotion:
    every applicable obligation above + held-out accuracy passed
```

Use only `passed`, `pending`, `not_applicable`, or `rejected` in each evidence column. A mixed record with one passed
column and one pending column remains blocked at the higher maturity level.

For every efficiency claim, log both machine-independent and measured quantities:

```text
C_total = C_teacher_P + C_teacher_S + C_teacher_C + C_local
          + C_live_proposal_forward + C_live_proposal_backward_or_vjp
          + C_action_device + C_action_host + C_recompute
          + sum_f k_f * DeltaTokens_f / FullTokens * C_replay_full
          + C_alignment + C_rollout + C_confirmation + C_escalation

T_total_predicted = C_device/F_device + C_host/F_host
                    + D_D2H/BW_D2H
                    + D_host/BW_host
                    + D_NVMe/BW_NVMe
                    + D_overlay_map/BW_candidate
                    + D_orbit_materialization/BW_candidate

M_peak = M_live_model
         + M_activation_microbatch
         + M_proposal_chunk
         + M_compact_overlay_map
         + M_optional_candidate_scratch
         + M_suffix_workspace.

M_persistent_total <= M_spool_cap - M_reserved.
```

`k_f` is the number of candidate lanes evaluated on the newly opened rows at fidelity `f`. Report forward-layer
equivalents, backward-layer equivalents, Viterbi transitions, decoded scalars, transferred bytes, persistent bytes,
peak bytes, and synchronized wall time. A lower operation count is not a speed proof when it increases transfers or
launches; a benchmark must validate the predicted bottleneck on each target backend.

For paired accuracy, let `d_s = U_candidate(s) - U_local(s)` on independent sequence `s`, where lower is better. Freeze
the minimum material improvement `Delta_min`, type-I error `alpha`, power `1 - beta`, and conservative selection-only
planning estimate `sigma_plan` before opening confirmation. `sigma_plan` is an upper confidence bound or the maximum
frozen estimate across eligible finalists, shards, and declared seeds; it is not the selected winner's potentially
optimistic variance alone. The normal-approximation planning lower bound is

```text
n >= ((z_(1-alpha) + z_(1-beta)) * sigma_plan / Delta_min) ** 2.
```

The actual decision uses the declared paired bootstrap/group-sequential procedure rather than assuming normality.
Every added secondary guardrail consumes the simultaneous or family-wise error budget.

### Required proof records

| ID | Step and purpose | Mathematical obligation before optimized implementation | Target and failure action |
|---|---|---|---|
| P0 | Local-root construction; correctness and fallback accuracy. | Prove legal V4 recurrence/packing and deterministic `argmin`/tie semantics. Show pack -> reconstruct -> save -> reload preserves the selected states and banks exactly. | Bit-exact artifact round trip, finite output, and declared local objective no worse than canonical bank 0. Otherwise abort replay and retain canonical local root. |
| P1 | Replay mode and disjoint teacher capture; efficiency, accuracy, and storage. | Prove `replay_mode=none` gives `C_teacher=C_proposal=C_replay=C_confirmation=0` and requires no replay dataset. For enabled strict replay, prove every forbidden calibration/YAQA/P/S/C prompt-ID intersection is empty. Prove that `P == S` disables confidence-sequence pruning. For top-`k` plus rest, prove stored probability mass sums to one and coarsened KL/JSD matches an explicit bucket oracle. Storage is `N_valid * (k*(b_id+b_logit) + n_temperature*b_lse + b_label + b_mask) + B_index`. | `none` is bit-exact with the local root. Strict replay meets the byte cap and oracle tolerance without changing candidate ordering on a small exact-logit model. Diagnostic reuse is labelled and cannot claim P6. Otherwise increase `k`, storage, or reject compact capture. |
| P2 | Transform-domain proposal adjoint; correctness. | Prove `delta_L = <T_out^*(g_y * SV), x_h @ delta_Q>` from the complete SU/Hadamard/inner/Hadamard/SV/bias forward. Bound finite-difference error by a declared multiple of machine epsilon and conditioning. | FP64 finite-difference and autograd/VJP parity across supported transform widths. Any unexplained residual blocks proposal scoring. |
| P3 | Proposal workspace; efficiency with exactness. | Prove `sum_s w_s <g_s, x_s delta_Q_a> = <sum_s w_s x_s^T g_s, delta_Q_a>`. With `q=max(1,n_live_shards)` and no redundant aggregate, log `M_GQ = 4N_w*q`, `M_scores = N_w*q/4`, `C_dense ~= 2SKN+2AKN`, and `C_direct ~= 2SKN+2ABKN`. Add full live forward/backward/VJP and every recomputation. Device scoring charges `D_D2H >= B*M_scores`; host-xg charges `b_xg*S*(K+N)` and must satisfy deterministic winner parity; module/stripe-major charges its activation/adjoint spool or recomputation. Charge the tiled or full `4KN` microgradient temporary, host/NVMe traffic, NUMA placement, reserved headroom, and every output stream. For sketches, declare the score-error/gap contract, shortlist size, and exact second-pass cost. | Select the lowest measured/predicted-time policy under device/host/NVMe caps. If simultaneous shards do not fit, use one-accumulator sequential disjoint-shard maps and charge their 11-bit temporary layout. A sketch may generate only; uncertified actions retain baseline unless exact-shortlist rescoring passes. Cap failure returns the local root. |
| P4 | Joint bank/orbit action palette; correctness and efficiency. | Prove the edge-XOR recurrence and tail biting for every supported `E`. Prove combined action uniqueness. Reuse four `c` and at most sixteen `c XOR m_b` mixes: `40/64 = 0.625` of naive scalar decodes. | Exact decoder equality and at least 37.5% scalar-decode reduction before transfer/launch effects; then measured non-regression. Palette accuracy must beat one-bank and independently designed controls on disjoint development validation. |
| P5 | Rate-aware three-map trust selection; accuracy at fixed replay count. | Record the Gaussian reference `D(R)=sigma^2*2^(-2R)` and the empirical low-rate path-churn evidence motivating smaller W1/W2 steps. Prove all maps are legal overlays, fractions are caps rather than quotas, and the baseline action remains available. Prove the sequential-shard fallback maps use only their declared action stream. | At most three exact candidate maps and no inference-format change. Rate fractions are hypotheses until held-out exact replay beats or matches 25/50/100 at equal work; failure restores the old schedule for that rate. |
| P6 | Adaptive successive fidelity; efficiency and selection accuracy. | Freeze maps from `P` before opening disjoint `S`. Log incremental work `C_candidates proportional to sum_f k_f*DeltaTokens_f`; cached prefixes require fixed execution geometry or proven batch-partition invariance. Carrying all three through medium and one/two through full costs 131,072/163,840 candidate-token lanes, or 196,608/229,376 including baseline. Freeze a simultaneous paired sampling-error bound, tail/boundedness assumptions, `Delta_select`, pair/look multiplicity, and false-elimination target separately from `eta_num`. | No duplicate or numerically incompatible row reuse. At W1--W2.5 every unresolved candidate advances; all three reach full fidelity when no bound can eliminate one. The schedule must match exhaustive full-fidelity winner selection at its predeclared error target; otherwise use exhaustive full fidelity. `P == S` or a candidate created after observing `S` disables this claim on reused rows and early statistical pruning. |
| P7 | Output alignment then rollout; accuracy and zero inference overhead. | Prove that alignment changes the candidate function `f_theta`, so a rollout of pre-alignment `theta` cannot validate post-alignment `theta'`. Alignment changes only existing SU/SV values, hence `Delta EBPW = 0` and the runtime operation graph is unchanged. Teacher-forced selection precedes isolated rollout of the complete artifact. Generated exact match exists only in rollout; dense-trajectory disagreement cannot veto a ground-truth rescue except through a frozen material guardrail. | Finalist rollout uses isolated state and matches fresh serial execution. Rejection consumes one conditional third lane for the already selection-valid unaligned finalist before Tier C/D escalation; a second rejection restores the local root. |
| P8 | Evidence-gated Tier C/D escalation; efficiency. | With cheap-stage rejection probability `p`, expected work is `C_cheap + p*C_escalation`, strictly below unconditional `C_cheap + C_escalation` for `p < 1`. Accuracy is preserved by exact replay and the immutable root. | Escalate only after the complete cheap candidate, including alignment/rollout, fails. If measured `p` or recovered utility does not justify cost, disable the tier. |
| P9 | Fixed-route MoE replay; correctness and memory. | Prove `Y_candidate = Y_baseline + r_e*(E_candidate(X_e)-E_baseline(X_e))` only while routing and earlier state are identical. Bound residency by active expert chunks, not the full layer. | Exact dense/reference equality under fixed routing. Any router/earlier change invalidates the cache and forces rerouting; uncovered experts retain local root. |
| P10 | Sealed adaptive confirmation; accuracy. | Use the paired primary UCB, cumulative alpha spending, conservative `sigma_plan`, power bound above, and a simultaneous secondary-guardrail envelope. Pre-enumerate a complete-prompt maximum reserve whose sequence and exact token totals jointly fit. First-stage rows contribute once to the final statistic. | Promote only when primary improvement exceeds `Delta_min`, hard correctness passes, and material guardrails pass. Uncertainty may open only predeclared complete prompts; every other outcome restores local root. |
| P11 | Immutable overlays and atomic publication; correctness and concurrency. | Prove registered root hashes and cache versions remain unchanged throughout selection. Prove compact orbit/bank decode equals materialized serialized scratch and charge scratch traffic when the reference path is used. The publication state machine exposes only `{root, winner}` and rollback is idempotent. | Reference overlay equals installed serialized payload on every supported backend; fault injection never exposes a partial artifact. Compact decode or fully charged materialization meets the frozen replay-cost target. |
| P12 | Serialization and inference cost; efficiency/correctness. | Bank refinement changes existing selector/state values but adds no production tensor, byte, branch, lookup, or launch relative to the same banked V4 format. Compute `Delta EBPW = 0` and compare kernel operation graphs. | Exact save/reload reconstruction and no measurable latency/VRAM regression beyond the frozen performance tolerance. Otherwise reject refinement integration, not the local artifact. |
| P13 | Default-policy promotion; end-to-end accuracy. | Compare local-root, refined, one-bank, uniform-rate, and matched-EBPW controls with paired outputs, at least two propagation seeds, and a frozen default-development task suite. Report correct-to-wrong and wrong-to-correct flips, then use a separate untouched suite for confirmatory reporting. Predeclare a minimum material recovery and report its Pareto position against wall time, forward/backward equivalents, peak bytes, and transferred bytes rather than accepting an arbitrarily small positive ratio. | Default-on at a rate requires statistically supported material primary recovery, no material guardrail regression, unchanged EBPW, and a non-dominated recovery/work point under the frozen target. Failure keeps replay opt-in or disabled at that rate. |

Current status at this design revision:

No conforming `ProofRecord` with an exact command, raw artifact hash, environment, baseline, target, and commit is
linked yet. Equations and source reasoning in this document are design obligations, not completed evidence. Therefore
no applicable column is marked `passed` merely from inspection.

| Proof | Algebraic | Numerical | Measured efficiency | Held-out accuracy | Current gate and remaining work |
|---|---|---|---|---|---|
| P0 | `pending` | `pending` | `not_applicable` | `pending` | Blocked; linked recurrence/round-trip proof, lifecycle, fault, and fallback-accuracy evidence remain. |
| P1 | `pending` | `pending` | `pending` | `pending` | Blocked; exact-logit oracle, byte measurement, strict disjointness, and complete-prompt reserve remain. |
| P2 | `pending` | `pending` | `not_applicable` | `not_applicable` | Blocked; linked adjoint proof, FP64 finite differences, and native/reference VJP parity remain. |
| P3 | `pending` | `pending` | `pending` | `not_applicable` | Blocked; linked workspace proof, allocation, CPU/device transfer/I/O, winner parity, and cap-failure evidence remain. |
| P4 | `pending` | `pending` | `pending` | `pending` | Blocked; combined-action uniqueness is an unfinished algebraic obligation, then decoder, speed, and recovery gates remain. |
| P5 | `pending` | `pending` | `pending` | `pending` | Blocked; linked overlay/map proof, equal-work replay, and per-rate fraction evidence remain. |
| P6 | `pending` | `pending` | `pending` | `pending` | Blocked; P/S independence, batch parity, replay savings, exhaustive-winner parity, and false-elimination evidence remain. |
| P7 | `pending` | `pending` | `pending` | `pending` | Blocked; alignment/rollout proof, isolated parity, conditional-lane cost, and rejection-path recovery remain. |
| P8 | `pending` | `not_applicable` | `pending` | `pending` | Blocked; linked expected-cost proof, rejection probability, and recovery per escalation byte/second remain. |
| P9 | `pending` | `pending` | `pending` | `not_applicable` | Blocked; linked fixed-route proof, sharded-MoE equality, route invalidation, and peak residency remain. |
| P10 | `pending` | `pending` | `pending` | `pending` | Blocked; linked decision-rule proof, reserve sizing, power, simultaneous guardrails, and sealed-row enforcement remain. |
| P11 | `pending` | `pending` | `not_applicable` | `not_applicable` | Blocked; compact-overlay/materialized parity, traffic, concurrency stress, and fault injection remain. |
| P12 | `pending` | `pending` | `pending` | `not_applicable` | Blocked; linked serialization proof, save/reload parity, and backend latency/VRAM evidence remain. |
| P13 | `not_applicable` | `not_applicable` | `pending` | `pending` | Blocked; held-out W1--W3 paired evaluations, matched-EBPW controls, and recovery/work Pareto evidence remain. |

The implementation sequence below references these IDs. Do not mark a column `passed` without an exact command, raw
artifact hash, environment, baseline, target, uncertainty calculation where applicable, and commit in the record.

## Validation plan

### Unit tests

- Four-bank tile scoring against an exhaustive small oracle.
- Sequential Block-LDLQ bank selection against an exhaustive small feedback oracle and canonical-bank rollback.
- Sequential YAQA anti-diagonal selection against an exhaustive small two-sided oracle and canonical-bank rollback.
- Deterministic bank/candidate-generation-order tie breaking on each supported quantization backend.
- `replay_mode="none"` invokes no teacher capture, alternate spool, backward pass, or replay and reloads exactly.
- The durable local snapshot is bit-exact after pack, reconstruction, save, and reload.
- The legacy automatic module-output gate cannot run when bounded full-model replay is requested; its calibration-token
  split is rejected as a prompt-disjoint propagation population.
- Bank-selection APIs cannot access local MSE/KLD/top-k fields below W3.
- Local metrics may choose the baseline bank/path but cannot delete requested alternatives when replay is enabled.
- Selector-only bank remaps are legal and reconstruct exactly at W1--W4.
- Edge-XOR automorphisms preserve the V4 recurrence, tail biting, planar packing, and reconstruction for
  `E in {4, 6, 8, 10, 12, 14, 16}`.
- Joint bank/orbit palettes contain no duplicate combined action, and the reused-mix scorer is exact with the naive
  sixteen-action decoder while issuing at most forty rather than sixty-four scalar outputs.
- Aggregate `G_Q` proposal scores match an explicit weighted per-sequence FP64 sum on a small oracle.
- Exact direct-action accumulation matches dense-`G_Q` scores and winners within the declared reduction tolerance.
- Dense-gradient, direct-score, and two-sketch workspace compute, transfer, NVMe, persistent-byte, and peak-byte
  equations match instrumented counters. Data-major device scoring and NVMe read-modify-write schedules produce the
  declared `B*M_scores` D2H and `(2B-1)*M_scores` NVMe bounds; module/stripe-major reports its lower final-score
  transfer plus every added activation/adjoint byte or recomputation. The auto policy rejects a nominally fitting but
  bandwidth- or headroom-infeasible arm before the sweep.
- Host-xg scoring reports `b_xg*S*(K+N)` transfer, NUMA placement, host FLOP/s, and tiled/full microgradient peak; it
  matches the deterministic device/reference winners before the resolver may select it.
- The two-trillion-weight storage oracle accounts for scores, retained actions, maps, temporary chunks, and explicit
  reserve rather than admitting one 500 GB accumulator solely against a nominal 512 GiB cap.
- Sketch score-error/gap certification matches an exact small oracle. Two-seed agreement without a sufficient gap is
  not called certified; the top-`m` union plus exact second-pass rescoring matches its explicit shortlisted oracle and
  sketches cannot reach acceptance APIs.
- Sequential disjoint-shard proposal generation uses one score accumulator, visits each proposal row once, emits the
  declared 11-bit temporary maps, exposes both shard directions and agreement, and never claims an exact aggregate.
- Immutable overlay execution is bit-exact with installing the same serialized candidate on every supported target
  backend; its logical map bytes and reference state-XOR count match the equations above; rejected overlays leave
  registered tensors and cache versions unchanged.
- Serial selection evaluates the baseline once per fidelity; optional candidate batching includes a same-shape baseline
  lane and selects the same winner.
- Proposal and selection prompt IDs are disjoint before a candidate is built. Successive fidelity evaluates every `S`
  sequence ID exactly once per candidate, eliminates only under the simultaneous pair/look bound, reproduces the
  131,072/163,840 candidate-lane counts when one/two maps reach full, and matches the exhaustive full-fidelity winner
  at the declared false-elimination target. Unresolved three-map selection becomes exhaustive.
- A candidate created after any `S` result is visible cannot reuse opened `S` rows under the P6 false-elimination
  claim. It must consume a pre-reserved independent extension or be labelled non-inferential tuning; sealed `C` may
  test only the one frozen finalist and cannot select a runner-up.
- Cached prefix reuse is accepted only under fixed microbatch partition/padding/dispatch/routing or proven batch-
  partition invariance; a changed execution geometry forces a charged recomputation.
- Selection-budget exhaustion returns the last fully selection-validated manifest, never the in-progress candidate.
- Propagation-tilted Viterbi can retain a locally worse path and matches an exhaustive propagated small oracle.
- `lambda=0` propagation-tilted Viterbi is bit-exact with each declared bank-specific local path; positive rescaling
  of either
  normalized term preserves its intended solution, and zero/non-finite normalization scales fail closed.
- `ProposalSurrogate` cannot be passed to replay acceptance and `ReplayScore` cannot be differentiated as a proposal.
- A candidate with better local MSE and worse final output is rejected by exact replay.
- A candidate with worse local MSE and better final output remains eligible and is accepted.
- Layer replay accepts a strict boundary improvement and records `layer_replay` provenance.
- A layer-replay selection loss, tie, non-finite result, or exception discards its overlay; final confirmation failure
  retains the exact local root.
- Prompt-level split disjointness and dataset/token hashes.
- Coarsened top-k-plus-rest KL and coarsened JSD versus explicit bucketed small-model oracles, plus ranking versus
  exact vocabulary KL.
- Optional common-random teacher-sample deltas and rankings versus exact teacher cross-entropy deltas.
- Optional temperature-specific compact targets versus exact low-temperature KL; temperature-1 metadata reuse at
  another temperature is rejected.
- Dense-prefix compact logits are rejected after a self-conditioned candidate trajectory diverges; normalized rollout
  edit distance and candidate-specific EOS masks match an independent oracle.
- Every nonzero broad-selection utility component covers the same primary sequence IDs; a finalist-only rollout subset
  is a separate typed score and cannot be mixed into a larger teacher-forced primary utility.
- Teacher-forced task, rollout task, NLL, and critical-margin component formulas, standardization constants,
  eligibility masks, and degenerate-scale floors match explicit small oracles. Autoregressive exact match is rejected
  from the broad teacher-forced score and is available only through isolated finalist rollout.
- Transform-domain first-order scores against FP64 finite differences through complete `QVQLinear` SU/Hadamard/
  inner/Hadamard/SV/bias execution, including composite transform widths.
- A regression proving that substituting an inner tile into the ordinary dense `X @ delta_W.T` formula is rejected.
- Every optional curvature estimator against an explicit small Hessian/Fisher oracle; mean-gradient squaring and
  isolated per-tile quadratic emissions with omitted cross terms are rejected.
- Microbatch, prompt-shard, and accumulation-order invariance within declared tolerances.
- Complete-map paired sequence reductions against an explicit oracle; selection intervals are labelled algorithmic, and
  token-as-independent or tilewise confidence claims are rejected.
- Verified second-pass regeneration parity and fail-closed behavior for every mismatched source/factor fingerprint.
- Baseline-history sibling capture versus verified on-demand regeneration equality under the frozen solver contract.
- Confirmation rows are inaccessible during sweeps/alignment. For a changed frozen artifact, local and refined are
  evaluated once on each newly opened confirmation shard; an identical artifact finalizes as local without opening
  confirmation.
- Confirmation route coverage cannot alter the MoE expert candidate set.
- Confirmation failure assigns and serializes the exact local-root artifact, never a stale refined-winner variable.
- Frozen comparator primary loss, confidence floor, guardrails, missing metrics, ties, and rollback behavior.
- Empty production guardrails, inconsistent fidelity token/sequence budgets, truncated confirmation prompts, an
  under-reserved 512-sequence teacher population, and uncorrected repeated looks fail configuration validation.
- Cumulative alpha spending, simultaneous secondary-guardrail coverage, material tolerances, and selection-only
  sample-size resolution match independent statistical oracles; six uncorrected zero-tolerance tests are rejected.
- Two unequal baseline repeats cannot resolve a noise floor; use the larger repeat path or fail closed.
- Layer-mode hidden-state objective and guardrails; categorical KL/top-k inputs are rejected.
- Compact-overlay decode equals materialized action-map scratch and the installed final payload; the reference path's
  repeated trellis traffic matches its declared lower bound, and final serialized reconstruction/reload is exact.
- Final atomic publication rolls back after transfer, validation, save, or reload failure.
- A concurrent reader sees the unchanged local root during overlay selection and either the complete root or complete
  winner during final publication; free-threaded stress tests observe no partial model or stale cache.
- Recoverable overlay failure continues from the exact parent; invariant/hash failure aborts to the untouched root.
- Selection scratch and manifests obey the resolved dense/direct/sketch workspace bound and compact-action equations, not
  total alternate trellis streams.
- Conditional selection versus incorrect independent-baseline selection.
- A synthetic two-module case where lower local MSE produces worse final output.
- Block-LDLQ and YAQA candidate generation.
- Attention, SwiGLU, router, and fixed-route expert replay boundaries.
- Output alignment precedes final rollout; aligned rejection evaluates the budgeted unaligned finalist using the
  cached local result before Tier C/D escalation, and Tier-D-only configuration never references an undefined ranked-
  coordinate set.
- Proof-gate tests require the conjunction of every applicable evidence column; one passed algebraic column cannot
  authorize optimized implementation or default promotion while numerical, efficiency, or accuracy evidence is pending.

### Integration gates

- Two-layer Llama 3.2 1B at W1, W1.5, W2, W2.5, and W3.
- Full-model Llama W1, W1.5, W2, and W2.5 evaluation on at least two propagation seeds.
- Local-to-final rank-correlation matrices by rate, showing whether W3 is a safe policy boundary.
- Joint bank/orbit residual diversity, occupancy, selector stability, and combined-action uniqueness across prompt
  shards and seeds.
- Four-bank non-local selection versus one-bank and local-best-bank controls at matched payload.
- Rate-aware model-wide proposals run before layer/semantic splits; maps freeze on `P`, adaptive 8K/32K/65K replay
  runs on disjoint `S`, and the simultaneous pair/look rule matches exhaustive winner selection at its predeclared
  false-elimination target. Every unresolved low-rate map advances, and hard budgets restore the last validated artifact.
- Exact aggregate, simultaneous-shard, sequential-shard, and sketch-shortlist workspace modes report predicted versus
  measured compute, transfer, NVMe, peak-memory, and wall-time costs before the auto resolver is enabled.
- Optional candidate-batched and serial replay select identical maps and produce matching structured scores; every
  batch includes a same-shape baseline lane.
- Finalist-only rollout keeps KV/routing caches isolated and matches fresh serial rollout for aligned and conditional
  unaligned finalists.
- Distribution-refined versus task/deployment-utility-refined controls, with final KLD and benchmark disagreements
  reported rather than hidden by one aggregate.
- A small MoE fixture with route changes and inactive experts.
- A sharded/offloaded MoE model where a complete expert layer cannot reside on one device.
- Exact Torch versus CUDA, MPS, and MLX reconstruction after save/reload on every supported target backend.
- Propagation selection does not change serialized BPW, operations, or latency relative to the same four-bank V4
  artifact; separately report the banked-versus-one-bank selector overhead.
- One changed frozen-artifact confirmation decision with paired uncertainty and no confirmation-driven retries.
- Predeclared two-stage confirmation matches its alpha-spending oracle and cannot alter candidates or thresholds after
  opening the first confirmation shard.
- Strict-disjoint YAQA/P/S capture is the production reference. Explicit fused fast modes match a separate dense
  capture exactly when contracts match and rejects reuse when any prompt, rendering, seed, mask, temperature, or
  evaluation-mode field differs.
- Coarsened JSD matches a top-k-plus-rest oracle; compact targets cannot be reported as exact vocabulary JSD.
- Joint bank/orbit masks selected on development models pass one untouched outer model/prompt test without retuning.
- Every enabled implementation-sequence step has a completed proof record with equations, evidence hashes, target,
  status, and rejection action; an `unproven` step cannot enter optimized implementation.

## Implementation sequence

1. **P0:** preserve sequential Block-LDLQ/YAQA bank selection, whole-proxy rollback, exact reconstruction, and durable
   roots. No replay code precedes the bit-exact local-root proof.
2. **P1:** replace the ambiguous legacy propagation field, implement `replay_mode="none"`, strict P/S/C prompt
   disjointness, compact teacher targets, and explicit YAQA-P/P=S fast modes. Prove that `none` invokes no capture or
   replay and that P=S disables P6 pruning claims.
3. **P2:** implement the transform-domain reference adjoint and validate it against explicit per-sequence FP64
   contractions, autograd, and complete `QVQLinear` finite differences.
4. **P3:** implement exact dense-`G_Q`, device-score, and host-xg reference workspaces, prove score/winner parity, then
   add host-resident, NUMA, NVMe, and module/stripe counters plus deterministic automatic selection under complete
   model-compute/transfer/I/O/headroom caps. Add one-accumulator sequential disjoint-shard generation before any sketch
   fallback. Sketch plus exact-shortlist rescoring remains research-only.
5. **P4:** jointly optimize fixed bank/orbit masks once per codec/rate, prove action uniqueness and edge-XOR legality,
   validate reused decoder subexpressions, and freeze a new codec version only after untouched outer validation.
6. **P11/P12:** add immutable Torch compact-overlay decode plus the materialized-scratch oracle, prove installed-payload
   and serialization parity, then extend the quantization-only decoder to supported native backends without changing
   production inference EBPW, operations, or persistent tensors.
7. **P5/P6:** freeze three rate-aware maps on P, then run adaptive 8K/32K/65K replay on disjoint S with guarded prefix
   reuse. Advance every unresolved W1--W2.5 map, including the third, and match the exhaustive full-fidelity winner
   target before adopting any simultaneous-bound elimination.
8. **P7/P10:** add selection-only output alignment, then finalist rollout, then the sealed paired confirmation procedure.
   Reserve the conditional unaligned rollout and complete-prompt confirmation population before proposal generation; prove joint
   token/sequence capacity and cumulative alpha-spending behavior.
9. **P9:** add bounded native input VJP or verified activation recomputation and fixed-route expert replay before
   large-model or sharded-MoE full-output selection.
10. **P8:** add on-demand baseline-history sibling capture/regeneration only after Tier A/B exact-replay evidence
    predicts positive recovery per charged byte and Viterbi second.
11. **P2/P8:** add normalized propagation-tilted Viterbi as a research-only second pass with `lambda=0` parity and
    predeclared escalation economics.
12. Add layer mode, curvature, a second gradient sweep, sketches, and candidate batching only through new proof records;
    each must show exact/referenced math plus measured final-output gain per compute and memory cost.
13. **P13:** run W1--W3 small/full-model, matched-EBPW, sharded-MoE, save/reload, concurrency, backend, and paired task
    gates before changing any rate default.

## Discussion points and open decisions

1. **Compact actions versus full paths.** Selector and orbit maps are the default. Measure their exact-replay gain
   before enabling sibling streams; compare added recovery per scratch byte and Viterbi second.
2. **Candidate count.** One proposal sweep yields at most three rate-aware maps plus the exact baseline. Under a frozen
   churn trigger, replace the middle correlated step with a shard-consensus mask over the same actions; never add an
   uncharged fourth candidate. A second action-ID stream is allowed only in the charged sequential disjoint-shard
   fallback, where it represents independent propagation evidence and uses the declared 11-bit temporary layout; a
   same-signal second-best stream remains research-only.
3. **Whole-model versus finer coordinates.** Model-wide proposals minimize exact replay count but can hide cancellation
   between good and bad regions. Use model-wide bounded maps first, then admit whole layers and semantic groups only
   after exact rejection or conflicting full-output-gradient votes and only within the frozen compute budget.
4. **Layer/short-window mode.** It is deferred from MVP. If later enabled, reuse output-alignment boundary capture and
   keep it diagnostic or explicitly layer-refined; it cannot prune a full-model candidate below W3.
5. **Gradient curvature model.** A stored diagonal is small, but a valid Fisher/Gauss--Newton estimator may require
   many score-gradient or HVP/VJP passes and its tile cross terms are not separable. Keep it at complete-map rescoring
   until an exact incremental formulation is proven. Test whether it changes final rankings enough to justify cost.
6. **Prompt coverage for MoE.** Inactive experts cannot receive propagation evidence. Decide whether to keep their
   baseline, add routing-coverage prompts, or assign a conservative higher rate.
7. **Router treatment.** Router changes invalidate expert routing caches and are much more expensive. Keep routers
   dense or higher precision initially, then evaluate router-bank selection as a separate feature.
8. **Confirmation rollback scope.** The initial contract makes one final local-root-versus-refined decision. If future
   per-layer attribution is needed, reserve independent confirmation shards or cross-fit before proposal generation.
9. **Metric comparator defaults.** The comparator contract is one predeclared sequence utility plus separate
   guardrails. Validate weights, clustered confidence levels, numerical floors, and non-inferiority tolerances without
   inspecting confirmation. Keep local diagnostics in a different type so they cannot enter acceptance.
10. **Teacher sketch size.** Start with deterministic top-32-plus-rest at temperature one. Validate top-16/top-64,
    extra temperatures, and common-random teacher samples only as charged ablations against exact small-model logits.
11. **YAQA ordering.** YAQA should generate the baseline trellis before propagation selection. Confirm whether a
    second YAQA factor refresh after accepted propagation changes is beneficial enough to justify another backward
    pass.
12. **Second sweep.** Default to one. A second live-gradient sweep may capture changed interactions, but it requires a
    predeclared selection-only gain/churn trigger and must beat the first sweep in recovery per forward/backward
    equivalent.
13. **Joint bank/orbit optimization.** Decide whether one fixed combined palette per rate generalizes across model
    families or whether attention, dense-MLP, and expert roles need separate frozen codec versions. Do not add
    per-model tables until propagated recovery proves that the extra format state is necessary.
14. **W3 boundary.** Treat W3 as a control until rate-by-rate rank-correlation and exact-replay evidence determines
    whether local metrics are reliable enough there. Do not extrapolate the W4 regime downward or the W2.5 regime
    upward without measurement.
15. **Alignment interaction.** Accepted SU/SV alignment changes the propagated residual and can change the preferred
    bank map. The initial implementation stops after one selection-chosen alignment. A future alternating schedule must
    predeclare its maximum selection-only cycles, end with alignment, and freeze before confirmation.
16. **Intra-bank path width.** Tier A/B avoid regenerating paths. If they saturate, compare one baseline-history
    sibling against one propagation-tilted path before wider top-k/tail-biting sets; choose by exact recovery per
    temporary byte and Viterbi second.
17. **Proof status.** Keep algebraic, numerical, measured-efficiency, and held-out-accuracy status separate for every
    proof record. Readiness is the conjunction of every applicable column, never the strongest completed column. A
    reference-only experiment may discharge an evidence obligation, but a pending or rejected arm cannot enter the
    optimized lifecycle merely because it shares code with a passing arm.

## Decision history

This log is reconstructed from `git log --follow` and the file diffs for every revision from `d1fd9265` through
`204a8143`, plus the current 2026-08-15 revision. Replaced choices remain here so future results can justify a
deliberate rollback without making the current normative path ambiguous.

- **`d1fd9265`: initial merged design.** It retained four bank streams, used local/short-window screening, replayed
  finalists, and considered V4 default-on at W1--W3. Replaced: full streams are prohibitive at large scale, and
  sub-W3 local/window screening can discard globally useful error directions.
- **`c579c9c4`: non-local authority.** It made final-output gradients and exact propagated replay authoritative and
  introduced tilted paths and three nested maps. The authority and bounded-map concept remain; tilted Viterbi moved to
  escalation because it is much costlier than legal selector/orbit transforms.
- **`33000275`: local fallback.** It restored a deterministic local bank/path winner as a complete fallback and added
  explicit `none`, `layer`, and `full_model` outcomes. Retained: the local root is mandatory but cannot accept a
  sub-W3 replay refinement.
- **`df8d0bf2`: lifecycle hardening.** It added reproducible source/factor regeneration, bounded spool/recompute
  policies, layer/full-model provenance, and one sealed final confirmation. The correctness contracts remain; layer
  mode and regeneration are deferred from the minimum full-model path.
- **`85cb589c`: mathematical/statistical hardening.** It added the exact transform-domain adjoint, independent-sequence
  comparisons, structured metrics, transactional rollback, and fixed budgets. Retained in intent; repeated candidate
  mutation/rollback is replaced by immutable overlays and one final publication transaction.
- **`e0a98c06`: global-first hardened plan.** It selected fixed full-bank streams as the first replay path,
  per-sequence tile risk tensors, candidate batching, broad rollout, and up to two sweeps. Replaced for efficiency by
  aggregate gradients, selector/orbit maps, serial replay, finalist-only rollout, and one default sweep.
- **`2e89d257`: streamlined propagation plan.** It introduced one aggregate `G_Q` direction, global 25/50/100 maps,
  one survivor after the initial fidelity, YAQA/search teacher reuse by default, finalist rollout before alignment, an
  eight-call replay cap, and six default zero-tolerance distribution guardrails. Retained: immutable overlays, exact
  final-output authority, cheap Tier A/B actions, and sealed confirmation. Replaced: the data-major `G_Q` residency
  claim does not scale, one early survivor is unsafe in the unstable low-rate regime, rollout must validate the aligned
  artifact, and correlated zero-tolerance guards can veto genuine task recovery.
- **2026-08-15 efficiency/accuracy and proof-ledger revision.** It adds an exact dense/direct-action workspace hierarchy,
  optional two-sketch proposal-only fallback, jointly designed bank/orbit palettes, rate-aware trust maps, adaptive
  8K/32K/65K fidelity with prefix reuse, alignment-before-rollout, escalation after complete cheap-path rejection,
  separate search/rollout/alignment/confirmation budgets, material guardrails, and proof records P0--P13.
  Rejected within this revision: a default second-best-action map, because it needs another action-ID stream and breaks
  the six-bit temporary-map proof; mandatory shard accumulators beyond the workspace cap, because retaining two medium
  candidates is cheaper and safer; and arbitrary 75% sequence agreement/six zero-tolerance metric vetoes, because
  they can reject concentrated task recovery without a materiality or multiplicity contract.
- **2026-08-15 balanced-efficiency correction.** It adds transfer/NVMe/headroom terms to direct-action scoring, a
  one-accumulator sequential disjoint-shard fallback, score-gap-aware sketch shortlisting with exact second-pass
  rescoring, sampling-error-aware survivor bounds, a two-finalist low-rate full-fidelity branch, a conditional
  unaligned rollout, joint confirmation sequence/token sizing, and conjunctive proof statuses. Replaced: admitting a
  500 GB score accumulator from nominal capacity alone; treating backend nondeterminism as population uncertainty;
  replacing unavailable shards with nested maps from the same aggregate direction; discarding an unaligned finalist
  before costlier Viterbi escalation; and authorizing work from the strongest completed proof column. The earlier
  rejection of a second action stream still applies to a same-signal second-best map, but not to the charged 11-bit
  temporary layout carrying two genuinely disjoint propagation directions.
- **2026-08-15 strict-selection and scalable-overlay correction.** It separates proposal `P`, replay-selection `S`,
  and sealed confirmation `C`; makes P=S an explicitly unvalidated fast mode; protects every unresolved third low-rate
  map through medium/full fidelity; separates numerical noise, selection gaps, and confirmation materiality; restricts
  broad selection to teacher-forced task losses; adds host-xg scoring and complete forward/backward accounting; and
  makes compact orbit-aware decode the scalable replay path with explicit logical-map/arithmetic cost. Replaced:
  confidence-sequence pruning on rows that generated—or adaptively changed—the maps, unconditional top-two pruning
  after 32 sequences, promotion against only a numerical floor, autoregressive exact match inferred from forced logits,
  uncharged repeated trellis materialization, redundant survivor-rule configuration, and proof columns marked passed
  without linked evidence records.
- **Current normative MVP.** Use the exact auto-selected final-output proposal workspace, Tier A selector remaps, Tier B
  edge-XOR orbits, compact immutable overlays, and at most three rate-aware maps generated on `P`. Replay them on
  disjoint `S`; W1--W2.5 advance every map not eliminated by a simultaneous pair/look bound, including all three to
  full fidelity when unresolved. Alignment is followed by aligned-finalist rollout
  and, only on rejection, one conditional unaligned rollout before escalation. Adaptive sealed confirmation opens
  only its jointly sized complete-prompt reserve. Tier C sibling paths, Tier D tilted Viterbi, sketches, curvature,
  layer replay, and sweep two require completed proof records plus measured escalation evidence.

The offline four-bank mask/permutation search was never intended to run per model quantization. It remains an amortized
codec-version design task keyed by rate; only the two-bit per-tile bank selector and final trellis are model artifacts.

## Decision rule

The governing rule is:

> Local reconstruction, Hessian/YAQA loss, local or layer KLD, and local top-k may optimize a legal path within a
> fixed bank and choose the deterministic local fallback. If replay is disabled, that valid fallback is the final
> artifact. If replay is enabled, local metrics have no authority to prune alternate banks or accept a replacement.
> Layer replay may produce only an explicitly layer-refined artifact. Full-model bank maps are generated first from a
> transform-correct final-output proposal sweep on `P` using an exact dense-gradient, device-score, or host-xg
> workspace and exact selector-only and edge-XOR candidate families. Sibling
> paths or propagation-tilted Viterbi are evidence-gated fallbacks, not prerequisites. The proposal surrogate
> cannot accept a candidate. Rate-aware model-wide maps are replayed on disjoint `S` through compact immutable overlays;
> finer coordinates consume only a frozen budget. Acceptance uses a separately typed sequence-level final-output
> utility and guardrails from the complete live quantized model. Final KLD alone has no authority. The frozen finalist
> must survive alignment-aware finalist rollout when configured and one sealed, predeclared paired confirmation with
> `UCB(U_refined - U_local) < -Delta_min`; otherwise the exact local root is serialized. No
> optimized step enters the lifecycle until every applicable
> algebraic, numerical, measured-efficiency, and held-out-accuracy proof column passes.
