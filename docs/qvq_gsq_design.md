# QVQ + GSQ accuracy-recovery design

Status: design proposal for PR 212 follow-up work

This document defines how GSQ should be integrated with QVQ at ultra-low bit
rates. The goal is complementary composition: QVQ remains responsible for its
format, codebooks, transforms, YAQA initialization, trellis constraints, and
runtime serialization; GSQ supplies a calibration-time search objective that
selects better legal QVQ representations. GSQ must not turn QVQ into scalar
GPTQ, bypass QVQ's format invariants, or introduce a second incompatible
packing path.

## Executive summary

QVQ and scalar GSQ solve different problems:

| Component | QVQ owns | GSQ owns |
| --- | --- | --- |
| Representation | Trellis/codebook payload, vector size, window layout, bank metadata | No representation changes |
| Initialization | YAQA/Fisher quantization, transforms, candidate-zero payload | No replacement of YAQA |
| Search variables | Legal QVQ tile/path choices and, in later phases, explicitly enabled QVQ metadata choices | Categorical relaxation, temperature schedule, candidate scoring, hard selection |
| Objective | YAQA metric construction and runtime-aware calibration inputs | Calibration reconstruction objective and stage schedule |
| Runtime | `QVQLinear`, kernels, payload compatibility, reload | No runtime kernel or payload interpretation |
| Validation | Native QVQ pack/decode/reload parity | Independent held-out improvement over the QVQ baseline |

The recommended first production slice is fixed-metadata QVQ-GSQ:

1. Run ordinary QVQ/YAQA and retain the canonical serialized payload.
2. Build a bounded set of legal decoded QVQ tile alternatives.
3. Optimize categorical tile selectors with a GSQ-style relaxed objective.
4. Materialize the best hard selector, preserving all QVQ metadata.
5. Install the selected candidate and continue the sequential model-prefix capture.
6. Serialize through the native `QVQLinear` path and reload before held-out scoring.

Learning QVQ scales, Hadamard transforms, bank families, or codebooks is not
part of this first slice. Those variables are coupled to payload legality and
runtime behavior and require separate ablations.

## Current implementation and boundary

The repository already contains an experimental format-aware path in
`gptqmodel/quantization/qvq_gsq.py`. It is deliberately narrower than scalar
GSQ:

- `TrellisCandidateAdapter` decodes and repacks the actual QVQ format.
- `refine_trellis_candidates` applies a Gumbel-softmax relaxation over tile
  candidates and returns a hard payload.
- `refine_trellis_fisher` uses prepared input/output Fisher factors.
- `baseline_bitflip_candidates` creates a shared prototype candidate pool.
- QVQ configuration currently rejects GSQ scale learning and restricts GSQ to
  plain YAQA formats without alignment, replay, SwiGLU search, activation
  quantization, scale search, or spectral refinement.

This path should be treated as the starting engine, not as permission to reuse
`GSQScalarTrainingModule` directly. Scalar GSQ assumes a groupwise scalar grid
and trainable scalar scales. QVQ has vector codebooks, trellis history,
overlapping windows, optional bank selectors, and transform metadata. A scalar
weight substitution would produce candidates that cannot be represented by a
valid QVQ payload.

The QVQ processor already has the right lifecycle concepts: dense capture,
YAQA target construction, candidate installation, propagation, and native
serialized-tensor reconstruction. The integration should extend those seams,
not create a parallel QVQ loader or a second model replay system.

## Representation model

For a QVQ linear layer, let:

- `c` be the serialized trellis/window payload;
- `b` be fixed bank selectors, when the format is banked;
- `S_U`, `S_V` and Hadamard metadata be the configured transforms;
- `D(c, b)` be the exact QVQ decoder in the inner coordinate system;
- `W(c, b, S_U, S_V)` be the effective runtime weight after the QVQ
  transform contract.

The GSQ search variable is initially only `c`, with `b`, codebook, vector size,
window size, and transforms fixed. Candidate selection must happen in the
inner coordinate system used by QVQ's existing YAQA code. This prevents an
accidental transpose, inverse-transform mismatch, or double application of a
Hadamard transform.

For tile `t` and candidate `j`, the decoded inner tile is `D[t, j]`. GSQ
relaxes the hard choice with:

```text
p[t, j] = softmax((z[t, j] + g[t, j]) / temperature)
W_tilde[t] = sum_j p[t, j] * D[t, j]
```

where `g` is fixed Gumbel noise for one optimization run. At the end, the
candidate with the best measured hard objective is selected. The payload is
then repacked using the native QVQ adapter. The relaxed tensor is never
serialized and never passed to a runtime kernel.

## Objective and math

### Local linear objective

For captured input activations `X` and a dense teacher weight `W_teacher`, a
candidate QVQ weight `W_candidate` can be scored with:

```text
E = W_candidate - W_teacher
L_local = || X E R ||_F^2 / normalizer
```

`R` is optional. When the YAQA output factor is available, use the existing
Fisher form:

```text
L_fisher = tr(G E^T H E)
H = L_H L_H^T
G = L_G L_G^T
```

with QVQ's already prepared factors. The factor convention must be recorded in
the result and must not receive an additional hidden damping or normalization.
The baseline candidate is always candidate zero, and a candidate is accepted
only if the hard objective is finite and improves the baseline on the search
calibration slice.

### Sequential block objective

The local objective is useful for candidate generation, but the accuracy target
is the Llama block output. The staged block objective should mirror the
paper-aligned scalar GSQ order:

1. **Q/K stage:** optimize Q and K candidates against the Q/K objective using
   the captured input factor. Q and K may be fitted independently.
2. **Attention stage:** install the selected Q/K candidates, retain dense V/O
   as the teacher prefix, and optimize V/O jointly against the residual plus
   attention output.
3. **MLP stage:** retain the fitted attention prefix on both teacher and
   student paths, refresh the MLP baseline if the selected policy requires it,
   and optimize gate/up/down against the post-attention block objective.

The critical invariant is that every later teacher contains the already
selected earlier candidates. A dense teacher override after Q/K or attention
selection silently trains against a prefix that the final model does not use.
That was the source of the earlier scalar GSQ regression and must be prevented
in QVQ as well.

### Candidate objective versus final model objective

Candidate search may use a YAQA Fisher proxy for speed. It must be followed by
the hard candidate's actual block/output evaluation. The acceptance record
should contain:

- candidate-zero and selected hard local loss;
- candidate-zero and selected hard block loss;
- input/output factor provenance and normalization;
- selected tile IDs and the number of non-zero selections;
- native payload hash;
- post-serialization reload parity;
- independent held-out KL, MSE, top-1, top-5, and top-10 metrics.

## Candidate construction by QVQ format

### W1–W3.5 P32 / banked formats

The initial W3 target is the V2B2-P32 path. Candidate construction must:

- keep the codebook version, vector size, trellis window, and bank selectors
  fixed;
- decode candidates through `decode_p32_window_tiles`;
- treat each complete 16x16 tile as the categorical unit;
- repack with `repack_p32_planar_to_window` before export;
- verify `pack(unpack(payload)) == payload` for every selected payload;
- validate both the canonical bank and the alternate-bank metadata in the
  native `QVQLinear` reload path.

`baseline_bitflip_candidates` is acceptable for a smoke test because it gives
the stochastic and deterministic arms a shared candidate pool. It should not
be the final candidate generator. Arbitrary packed-word bit flips do not
provide a useful neighborhood in codebook space and can waste the optimizer's
budget. The production generator should enumerate or sample legal trellis
neighbors, codebook alternatives, and bounded path substitutions while
preserving circular-window dependencies.

Bank selection should remain fixed in phase one. A later bank-selection phase
can add a categorical bank choice, but it must select complete legal bank
payloads and must be scored with the same independent confirmation stream.

### W4–W8 non-banked planar formats

The existing adapter supports the non-banked V2/L16 planar family. This is the
cleanest first QVQ-GSQ integration target above W3 because it has no bank
selector coupling. The candidate generator can operate over planar trellis
states, decode with `decode_trellis_tiles`, and serialize the selected trellis
directly through the existing QVQ tensors.

The W4 path must not silently switch to scalar groupwise W4. `group_size` is
not a QVQ format parameter; QVQ's tile/trellis geometry remains authoritative.

## Integration architecture

### 1. Keep the search engine format-aware

Extend `gptqmodel/quantization/qvq_gsq.py` with interfaces equivalent to:

```python
candidate_pool = build_qvq_candidates(
    baseline_payload,
    qvq_metadata,
    candidate_policy,
)
result = refine_qvq_candidates(
    candidate_pool,
    target_inner_weight,
    input_factor,
    output_factor,
    objective="fisher_or_stage",
)
```

The engine should return a payload and diagnostics, not a dense replacement
weight. It must not know how a Llama block is installed or how a full model is
saved.

### 2. Keep model staging in `QVQProcessor`

`QVQProcessor` should own:

- capture and prefix replay;
- stage ordering;
- teacher/student module construction;
- candidate acceptance and confirmation;
- replacement of the live module before the next capture;
- final attachment of serialized QVQ tensors.

This keeps GSQ's optimization policy separate from QVQ's model lifecycle and
allows the same candidate engine to be reused for linear-only refinement.

### 3. Keep runtime and export in QVQ

The selected result must be converted to the same tensor dictionary consumed by
`QVQLinear`. No GSQ-specific runtime module should be introduced. The export
contract must include all tensors required by the chosen format:

- `trellis`;
- `SU` and `SV` when configured;
- `bank_ids` and `bank_alt_id` for banked formats;
- bias and rank-8 metadata where applicable.

## Packing and reload contract

Packing is part of the algorithm's accuracy boundary, not an afterthought.
Unpacked fitted weights must never be used as the final evidence.

For scalar GSQ, the current export path computes integer codes from the fitted
weight and learned scales, stores scales in FP16, and then dequantizes through
`TorchLinear`. This introduces a final representational projection that the
unpacked optimizer did not see. The W3 experiment demonstrated the effect:
unpacked GSQ looked dramatically better, while the packed/reloaded result was
still better but materially less so. The correct interpretation is the packed
result.

For QVQ, the design should avoid a second quantization boundary: optimize the
decoded native payload candidate, select a hard payload, serialize it with the
native QVQ writer, reload it into `QVQLinear`, and score that reloaded module.
The candidate decoder and runtime decoder must share the same code path or have
an explicit parity test.

Every experiment must assert:

```text
serialized payload == reloaded payload
decoded(selected payload) == decoded(reloaded payload)
runtime(selected payload) ~= canonical decoded reference
```

The last check should report mean/max error and use the existing QVQ tolerance
for the relevant dtype and kernel.

## Calibration and disjointness

The search and held-out streams must be disjoint by token sequence or prompt
fingerprint, depending on the data format. Training rows must never be reused
for final top-k/KL claims.

The PR 212 Llama validation used:

- Llama-3.1-8B-Instruct;
- FineWeb-Edu `sample-10BT` at the recorded dataset revision;
- seed-42 shuffled concatenation with no inserted EOS;
- 128 training sequences of length 4096 for the bounded block experiment;
- 32 held-out sequences totaling 6,367 scored tokens;
- zero exact train/held-out sequence intersection.

The 128-row slice is a bounded diagnostic, not a paper-scale reproduction of
4096 calibration sequences. A full QVQ-GSQ claim must record its complete
calibration count, token budget, tokenizer hash, dataset revision, and
disjointness audit.

## Evidence from PR 212

These results are block-0 propagation experiments. They demonstrate the
packing and metric contract; they are not full 32-layer model claims.

### Packed W3, group size 128

Configuration: signed GPTQ initialization, damping 0.01, 20 epochs, 2000 Q/K
steps, BF16 training, native `TorchLinear` GPTQ_V2 packing and exact state
reload.

| Metric | Packed GPTQ | Packed GSQ |
| --- | ---: | ---: |
| KL teacher-to-candidate | 0.28602 | 0.09792 |
| Logit MSE | 0.88452 | 0.40319 |
| Top-1 agreement | 82.94% | 90.34% |
| Top-5 overlap | 82.15% | 88.64% |
| Top-10 overlap | 82.10% | 88.30% |

GSQ reduced KL by 65.8% and MSE by 54.4% after packing/reload. Payload reload
was exact for both arms.

### Packed W4, group size 128

The same model, streams, initialization, damping, training budget, and packed
reload protocol were used at W4.

| Metric | Packed GPTQ | Packed GSQ |
| --- | ---: | ---: |
| KL teacher-to-candidate | 0.05331 | 0.02136 |
| Logit MSE | 0.37861 | 0.10995 |
| Top-1 agreement | 92.82% | 94.80% |
| Top-5 overlap | 91.49% | 93.59% |
| Top-10 overlap | 91.40% | 93.62% |

GSQ reduced KL by 59.9% and MSE by 71.0% after packing/reload. Payload reload
was exact for both arms.

The artifacts were produced under `/tmp/qvq-pr212-llama31-8b-w3-block0-paper`
and `/tmp/qvq-pr212-llama31-8b-w4-block0-paper`. They should be copied to a
durable experiment archive before being used as long-term evidence.

## Validation matrix for implementation work

Each new QVQ-GSQ change should run the following matched matrix:

| Axis | Required arms |
| --- | --- |
| Representation | Native QVQ baseline vs QVQ-GSQ |
| Search | GSQ candidate relaxation vs deterministic matched candidate search |
| Format | W3 P32/V2B2, then W4 non-banked planar |
| Scope | Q/K local, attention stage, MLP stage, full block propagation |
| Persistence | In-memory candidate, serialized payload, reloaded `QVQLinear` |
| Metrics | Calibration loss, held-out KL, MSE, top-1, top-5, top-10 |
| Data | Disjoint train/confirmation/held-out streams |

Minimum acceptance criteria for the first production slice:

1. GSQ never changes QVQ codebook, transform, bank, or kernel metadata.
2. Disabled GSQ returns a byte-identical native QVQ baseline.
3. Candidate payloads round-trip through pack/unpack exactly.
4. Native QVQ reload matches the canonical decoded reference within the
   established runtime tolerance.
5. Hard candidate calibration loss is no worse than candidate zero.
6. Held-out metrics are reported after reload, not from the relaxed tensor.
7. The candidate pool and confirmation streams are explicitly disjoint.
8. The baseline QVQ arm is bit-for-bit matched between the comparison runs.
9. A candidate that improves local loss but regresses held-out top-k beyond the
   configured tolerance is rejected.
10. Full-model claims are reserved for a complete layer propagation/export
    test; block-local results are labeled as such.

## Rollout plan

### Phase 0: contract and regression harness

- Add QVQ candidate-pool protocol and metadata validation.
- Add payload, decode, and native-reload parity tests.
- Add an explicit train/confirmation/held-out disjointness checker.
- Preserve current GSQ-disabled byte identity.

### Phase 1: fixed-metadata tile GSQ

- Replace random bit-flip candidates with legal trellis neighbor candidates.
- Support W3 P32/V2B2 with fixed bank selectors.
- Score with the existing YAQA factors and hard candidate guard.
- Validate block-0 packed/reloaded results on Llama-3.1-8B.

### Phase 2: sequential Llama block staging

- Add Q/K, attention, and MLP teacher-prefix staging.
- Install each hard QVQ candidate before capturing the next stage.
- Add full block propagation and packed/reload scoring.

### Phase 3: W4–W8 planar QVQ

- Add native legal planar candidate neighborhoods.
- Validate W4 first, then higher rates.
- Keep non-banked format restrictions explicit in configuration.

### Phase 4: optional metadata optimization

Only after fixed-metadata search is stable, evaluate separately:

- bank selector search;
- module/output scale search;
- transform optimization;
- codebook adaptation.

Each is a different algorithmic contribution and must have its own payload and
reload regression. They should not be silently enabled by `gsq=True`.

## Performance and memory guidance

QVQ candidate search should be bounded by decoded candidate bytes, not just the
number of candidates. Decode candidates tile-wise and reuse the shared input
factor. Keep one hard baseline and one candidate tile block resident where
possible. Do not materialize all full dense candidate matrices for every
projection.

For long-context Llama calibration, use the existing capture/offload mechanisms
and microbatch limits. Attention objectives at sequence length 4096 can exceed
GPU memory even on a 96-GiB device when eager attention uses a large
microbatch. Candidate optimization should therefore expose a microbatch size
and report the actual capture/storage policy.

## Non-goals

This design does not:

- replace QVQ's YAQA initializer;
- make scalar GPTQ group size meaningful inside QVQ;
- introduce scalar GSQ scales into QVQ;
- optimize the relaxed tensor as a runtime weight;
- claim paper reproduction from a 128-row diagnostic;
- combine GSQ with QVQ replay/alignment/activation/spectral features before
  their interaction is separately validated;
- accept an unpacked result as final evidence.

The intended result is a clean division of labor: QVQ defines what weights are
legal and how they run; GSQ chooses among those legal weights using a stronger
calibration objective and staged teacher prefixes.
