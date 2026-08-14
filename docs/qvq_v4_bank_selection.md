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

The objective is to choose these selectors to reduce error after it propagates through the quantized model,
especially at W1--W3. Local reconstruction error may propose a bank map, but it must not be the final acceptance
authority. A locally larger residual can cancel an existing downstream residual, while a locally smaller residual
can rotate into a high-gain direction and damage final logits.

The design must preserve:

- the V4 planar weight payload;
- two selector bits per 16x16 tile;
- zero new inference operations;
- zero new production codebooks;
- exact Torch/native reconstruction parity;
- bounded GPU residency during quantization;
- atomic fallback to the accepted baseline.

## Bank-selection granularity

The four banks are not selected once per model, layer, or module. They are selected per 16x16 tile. Brute-force
end-to-end replay of every possible map is impossible:

```text
number of complete maps = 4 ** tile_count
```

QVQ therefore separates candidate generation from candidate acceptance:

```text
all four bank reconstructions per tile
        |
        v
local, YAQA, or downstream-gradient proxy
        |
        v
one or more complete serialized bank maps
        |
        v
exact propagated replay of each shortlisted map
        |
        +-- accept
        +-- reject and restore
```

A replay evaluates a complete candidate map for a semantic coordinate. It does not run one full-model forward for
each individual tile.

## Data contract

Keep the following data populations prompt-disjoint:

| Population | Purpose |
|---|---|
| Ordinary calibration | Input Hessian, Block-LDLQ, scale statistics, and sequential quantized replay. |
| YAQA Fisher | Full-model Sketch-B factors when YAQA is enabled. |
| Propagation search | Candidate generation, partial-boundary screening, and conditional search decisions. |
| Propagation confirmation | One independent check after a completed refinement sweep. |
| Final evaluation | Benchmarks that are not used to select banks or tune thresholds. |

If no separate propagation dataset is supplied, split complete prompts before tokenization and remove the selected
prompts from ordinary calibration. Do not reserve the final token rows of the same calibration prompt. Track and
serialize dataset revision, prompt hashes, token hashes, valid token counts, length distribution, rendering contract,
and random seeds.

The first implementation should target at least 65,536 valid search tokens and 65,536 valid confirmation tokens,
with at least 128 independent sequences in each population. These are operational defaults, not universal sample
complexity claims. Fail closed when the requested minimum cannot be met.

## Compact dense-teacher targets

Full vocabulary logits are too large to retain for large models. During a dense teacher pass, store the following per
valid token:

- top-k teacher token IDs and logits, initially k=32;
- teacher log-sum-exp;
- teacher top-1 and top-5 identities;
- teacher top-1 margin;
- one or two categorical samples from the teacher distribution;
- the natural next-token label when available.

For a sample `y ~ p_dense`, the quantity

```text
-log p_quant(y)
```

is an unbiased Monte Carlo estimator of the teacher-to-quantized cross entropy. Teacher entropy is constant across
candidates, so candidate differences estimate forward-KL differences without retaining the full vocabulary tensor.
Small-model tests must compare this compact score with exact vocabulary KL and verify candidate-ranking agreement.

## Stage-wise lifecycle

### Stage 0: teacher capture

Run the dense model once on the propagation search and confirmation prompts. Persist only compact teacher targets
and immutable prompt/token metadata. This stage must finish before dense modules are replaced.

### Stage 1: baseline quantization

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

For V4 W1--W3, produce the four bank-specific trellis streams once while the transformed weight and Hessian/YAQA
factors are resident. Write them directly to CPU or temporary NVMe storage; do not retain four decoded weight
matrices on the GPU.

The temporary four-stream cost is `4R` bits per weight at rate `R`, excluding the live baseline artifact. Examples:

| Rate | Four temporary streams | Bytes per weight |
|---:|---:|---:|
| W1.5 | 6 bits/weight | 0.75 |
| W2 | 8 bits/weight | 1.00 |
| W2.5 | 10 bits/weight | 1.25 |
| W3 | 12 bits/weight | 1.50 |

This storage is temporary and must be released layer by layer after refinement. A low-disk mode may regenerate bank
streams on demand, trading Viterbi compute for storage, but it must not silently change arithmetic or candidate order.

### Stage 2: one downstream-sensitivity pass

After the complete baseline quantized model exists, run one propagation-search forward/backward pass. For module
output `y`, candidate change `delta_y`, live downstream gradient `g_y`, and a diagonal Fisher/Gauss--Newton estimate
`D_y`, score a candidate with

```text
delta_L ~= <g_y, delta_y> + 0.5 * delta_y.T @ D_y @ delta_y.
```

This proxy can prefer a candidate with worse local MSE when it cancels accumulated downstream error. Recompute live
sensitivity after an accepted sweep; do not reuse dense-model gradients as if they described the quantized model.

Use hooks to consume activations and gradients as the backward traversal reaches a subset. Reconstruct one bank at a
time or use a bounded bank batch, accumulate four scalar scores per tile, then free the activation, gradient, and
decoded candidate workspace. Persist only scores, selectors, and packed candidate state streams.

### Stage 3: partial-boundary screening

Use the existing hook/exception mechanism to stop at the nearest meaningful nonlinear boundary:

| Candidate coordinate | First required replay boundary |
|---|---|
| QKV | Completed attention output. |
| Attention O | Attention residual output. |
| Dense MLP gate/up | Completed activation-and-product output. |
| Dense MLP down | MLP residual output. |
| Expert gate/up/down | Aggregated MoE output. |
| Router | Complete rerouted MoE output. |

Score all four banks from one captured input; do not rerun capture four times. Candidates that are non-finite or
catastrophically worse at every available boundary may be rejected early. A small local-MSE regression alone is not
a rejection criterion at low rates.

Survivors advance through a two-to-four-layer propagation window. Throw an early-stop exception at the end of the
window and retain only the best candidate plus candidates inside a predeclared uncertainty band.

### Stage 4: exact final-logit acceptance

Only shortlisted candidates run to final logits. The prefix before the candidate must not be rerun when an exact
cached boundary input is available:

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
final-logit score
```

Candidate installation must use its serialized trellis and packed bank selectors, then reconstruct through the same
contract used after checkpoint reload. Never score an unpublished dense staging matrix.

Selections are conditional. If `A` is the current accepted set and `c` is a candidate:

```text
delta(c | A) = L(A union {c}) - L(A).
```

Install one candidate, evaluate it in the current accepted model, and either retain it or restore the exact snapshot.
Do not score every module independently against the original baseline because this discards cross-module effects.

### Stage 5: sweep confirmation and output alignment

Do not spend the independent confirmation set after every candidate. Snapshot the model at the beginning of a sweep,
make conditional search-set decisions, and evaluate confirmation once after the completed sweep. Roll back the entire
sweep when confirmation fails.

Run fixed-trellis output alignment only after final bank/trellis selection. Then run one more full-model search and
confirmation gate because the optimal SU/SV values depend on the selected decoded matrix.

## Hierarchical replay coordinates

Begin with the largest semantically valid coordinate and split only after rejection:

```text
whole decoder layer
        |
        +-- accepted
        |
        +-- rejected
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
If the combined candidate fails, split QKV from O.

Dense gate/up remain coupled through the multiplicative activation. Test `gate + up + down` first, then split
`gate + up` from `down`. Do not promote gate and up independently from their linear outputs.

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
assignments and reroute. Do not refine experts that receive no search or confirmation tokens; retain the baseline and
report their missing coverage. Use additional routing-coverage prompts before making a production default.

Do not fuse individual expert gate/up weights merely to make replay convenient; that duplicates packed tensors and
can cause OOM. Group only compatible active experts for dispatch.

## Memory hierarchy

The intended peak GPU residency is:

```text
one active dense or quantized subset
+ one decoded bank candidate or bounded bank batch
+ one activation microbatch
+ required suffix workspace
```

Use the following placement:

| Object | Preferred placement |
|---|---|
| Live baseline quantized model | Normal sharding/offload device map. |
| Active bank reconstruction | Owning accelerator only. |
| Four bank trellis streams | Temporary NVMe or CPU memory. |
| Bank scores and selector maps | CPU memory. |
| Boundary activations | Pinned CPU memory, streamed by microbatch. |
| Teacher sketches | CPU or NVMe. |
| Rejected candidate snapshots | CPU/NVMe, module or subset scoped. |

Process decoder layers in forward order. Keep only the current accepted boundary state, produce the next boundary
after settling the layer, and release the previous one. Do not retain every layer's full activation tensor.

Prefetch the next candidate stream asynchronously, but never make correctness depend on completion of an unchecked
transfer. Candidate installation and rollback must remain atomic under the owning module's lifecycle lock.

## Compute budget

If every module candidate were replayed through the full suffix, a model with `L` layers, `G` groups per layer, and
`K` candidates per group would cost approximately

```text
K * G * (L + 1) / 2
```

full-forward equivalents per sweep. This is not the production plan.

The staged plan spends:

1. one dense teacher pass;
2. existing partial-hook baseline captures;
3. one quantized forward/backward sensitivity pass per sweep;
4. cheap boundary and short-window partial replays;
5. final-logit suffix replay for finalists only;
6. one complete search and one confirmation replay per sweep.

The target is no more than one finalist per semantic group and normally one whole-layer finalist. Rejected whole-layer
candidates trigger finer attention/MLP/expert work only for that layer. Record actual forward-layer equivalents,
tokens, active experts, bytes transferred, and wall time instead of reporting only a model-level duration.

## Acceptance metrics

Return a structured score rather than treating mean KLD as sufficient:

- sampled teacher cross entropy;
- top-k-plus-other-bucket KL;
- low-temperature KL;
- teacher top-1 margin loss;
- top-1 disagreement;
- top-5 overlap loss;
- non-finite count;
- activation norm/cosine and covariance drift at partial boundaries.

Continuous distribution and margin metrics rank candidates. Top-1, top-5, range, and non-finite counts are
guardrails. Search-set improvement must exceed measured replay/reload noise, and no guardrail may have a statistically
clear regression. Confirmation must repeat the direction on prompt-disjoint data.

Final ARC, GSM8K, STEM, History, MMLU, or similar tasks are evaluated only after the bank-selection policy and
thresholds are frozen. Report correct-to-wrong and wrong-to-correct flips plus paired uncertainty, not only net score.

## Proposed configuration

The public configuration should be nested rather than adding more unrelated fields directly to `QVQConfig`:

```python
PropagationConfig(
    enabled=True,
    mode="full_model",
    dataset=None,
    search_valid_tokens=65_536,
    confirmation_valid_tokens=65_536,
    minimum_sequences=128,
    teacher_samples_per_token=2,
    teacher_topk=32,
    short_window_layers=4,
    candidates_per_group=1,
    maximum_sweeps=2,
    screen_valid_tokens=16_384,
    seed=0,
)
```

Recommended initial policy:

- default on for QVQ V4 with four banks at W1--W3;
- support both Block-LDLQ and YAQA candidate generators;
- explicit `enabled=False` always disables it;
- no silent calibration-row fallback;
- maximum one completed second sweep unless the first sweep passes confirmation;
- preserve the original baseline as an exact fallback candidate.

## Transaction and serialization contract

Each candidate must record:

- module and semantic-group identity;
- rate, vector size, bank count, codebook version, and shape;
- packed trellis, packed bank selectors, SU, and SV;
- configuration and dataset hashes;
- candidate-generation method and seed;
- local, boundary, window, search, and confirmation metrics;
- a hash covering tensors and metadata.

Installation must:

1. synchronize pending transfers;
2. acquire the owning module lock;
3. snapshot registered tensors and backend caches;
4. install tensors on the module's actual owning device;
5. invalidate decoded-selector and kernel caches;
6. run exact reconstruction and shape/dtype validation;
7. evaluate;
8. restore exactly on rejection, exception, non-finite output, or failed confirmation.

No rejected candidate may mutate the checkpoint or become visible to concurrent inference. Only the accepted artifact
is serialized, so production EBPW and inference VRAM remain unchanged.

## Validation plan

### Unit tests

- Four-bank tile scoring against an exhaustive small oracle.
- Prompt-level split disjointness and dataset/token hashes.
- Compact teacher score versus exact full-vocabulary KL on a small model.
- Serialized candidate reconstruction and reload equality.
- Atomic rollback after rejection, exception, transfer failure, and non-finite output.
- Conditional selection versus incorrect independent-baseline selection.
- A synthetic two-module case where lower local MSE produces worse final output.
- Block-LDLQ and YAQA candidate generation.
- Attention, SwiGLU, router, and fixed-route expert replay boundaries.
- Output alignment after accepted trellis changes.

### Integration gates

- Two-layer Llama 3.2 1B at W1.5, W2, W2.5, and W3.
- Full-model Llama W2 and W2.5 evaluation on at least two propagation seeds.
- A small MoE fixture with route changes and inactive experts.
- A sharded/offloaded MoE model where a complete expert layer cannot reside on one device.
- Exact Torch versus CUDA, MPS, and MLX reconstruction after save/reload on every supported target backend.
- No change to final serialized BPW, inference operations, or inference latency outside measurement noise.
- Search improvement repeated on independent confirmation with paired uncertainty.

## Implementation sequence

1. Add `PropagationConfig`, prompt-level data splitting, and compact teacher-target capture.
2. Extend quantization results with serialized four-bank candidate streams and complete metadata hashes.
3. Implement transactional candidate installation and exact rollback tests.
4. Integrate the existing full-model refiner after all QVQLinear modules have been installed.
5. Add live-gradient tile scoring and structured acceptance metrics.
6. Add hook-based boundary and short-window screening.
7. Add attention/MLP semantic grouping and fixed-route MoE expert-delta replay.
8. Run output alignment after selection and gate it through full-model confirmation.
9. Add suffix caching, candidate prefetch, and bounded-memory optimizations without changing selection results.
10. Run the small-model, full-model, sharded-MoE, save/reload, and backend gates before changing defaults.

## Discussion points and open decisions

1. **Temporary storage versus Viterbi recomputation.** Four bank streams minimize repeated quantization compute but can
   require `4R` temporary bits per weight. Measure NVMe bandwidth and total spool size before choosing the default.
2. **Candidate count.** One gradient-derived candidate per group minimizes replay. Multiple candidates may recover
   from proxy error but multiply suffix work. Start with one plus the exact baseline.
3. **Whole-layer versus semantic-group first.** Whole-layer proposals minimize replay count but can hide cancellation
   between good and bad subsets. Use whole-layer first and split rejected or ambiguous layers.
4. **Short-window length.** Two layers are cheaper; four layers reveal more residual and routing amplification. Select
   from measured candidate-ranking agreement with final logits, not local KLD.
5. **Gradient curvature model.** Diagonal Fisher is inexpensive but approximate. Test whether block or low-rank
   curvature changes final candidate rankings before adding its storage and solver complexity.
6. **Prompt coverage for MoE.** Inactive experts cannot receive propagation evidence. Decide whether to keep their
   baseline, add routing-coverage prompts, or assign a conservative higher rate.
7. **Router treatment.** Router changes invalidate expert routing caches and are much more expensive. Keep routers
   dense or higher precision initially, then evaluate router-bank selection as a separate feature.
8. **Confirmation rollback scope.** Whole-sweep rollback is simplest and safest. Per-layer attribution may recover
   more gains but requires additional confirmation replays.
9. **Metric comparator.** A single scalar is easy to optimize but can hide top-k regressions. Prefer a structured
   comparator with predeclared primary losses and non-inferiority guardrails.
10. **Teacher sketch size.** Validate top-16, top-32, and top-64 plus one versus two teacher samples per token against
    exact KL rankings before fixing the default.
11. **YAQA ordering.** YAQA should generate the baseline trellis before propagation selection. Confirm whether a
    second YAQA factor refresh after accepted propagation changes is beneficial enough to justify another backward
    pass.
12. **Second sweep.** A second live-gradient sweep captures changed cross-module interactions. Enable it only when the
    first sweep passes confirmation and the remaining improvement exceeds its measured compute cost.

## Decision rule

The governing rule is:

> Local reconstruction, YAQA, and downstream-gradient approximations may generate and screen V4 bank maps. Only an
> exact serialized candidate replayed through the real quantized downstream path may be accepted, and the completed
> sweep must survive an independent prompt-level confirmation gate.
