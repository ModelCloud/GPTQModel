# QVQ quantization preparation and lifecycle integration plan

## Status and intent

This document defines the migration of the QVQ comparison and sweep workflow from
`scripts/compare_qvq_codecs_llama_qkvo.py` into GPT-QModel's supported model lifecycle. It is an implementation plan,
not a statement that every stage described below already exists.

The target public flow is:

```text
GPTQModel.load(source)
    -> prepared_artifact = model.prepare(stage=PrepareStage.QUANTIZATION, ...)
    -> model.quantize(prepared_artifact=prepared_artifact)
    -> validate live packed model
    -> model.save(output)
    -> GPTQModel.load(output)
    -> evaluate reloaded packed model
```

The integration must preserve the standalone harness's validated QVQ math and single-GPU performance while adding:

- model-tree-driven target selection for every supported architecture;
- the existing subset, layer-scope, shell/turtle, multi-GPU, packing, save, and reload lifecycle;
- reusable calibration, Hessian, factorization, and YAQA preparation artifacts;
- exact live-versus-reloaded validation;
- bounded memory for dense, very large dense, and MoE checkpoints;
- a thin declarative sweep runner instead of a second quantization implementation.

The lifecycle must never silently substitute a different calibration population, seed policy, feedback geometry,
bank-selection policy, quantization objective, or inference representation. Any intentional algorithm change remains
an explicit QVQ configuration arm with its own A/B gate.

## Current gap

The comparison harness currently performs a useful but parallel lifecycle:

1. loads a dense Hugging Face model directly;
2. discovers selected linear modules;
3. captures pristine dense activation Hessians;
4. optionally captures full-model YAQA Sketch-B factors;
5. calls `quantize_qvq_linear()` directly;
6. writes reconstructed dense weights back into `nn.Linear` modules;
7. compares the dense teacher and reconstructed-weight candidate.

That path is suitable for codec research, but it bypasses important production behavior:

- `GPTQModel.load()` and LazyTurtle/meta-shell materialization;
- the model adapter's `module_tree` and calculation groups;
- `ModuleLooper`, `SubsetPlan`, layer scoping, and device assignment;
- packed `QVQLinear` installation and native inference;
- checkpoint metadata, save, reload, and resharding;
- lifecycle cleanup, failure rollback, and bounded cache ownership;
- representative packed-kernel evaluation.

It can therefore disagree with production even when its dense reconstructed weights are correct. The script must
eventually become an orchestrator of the supported lifecycle, not remain an alternate implementation.

## Design principles

### Accuracy first

Preparation is part of the quantization algorithm. Reordering capture, changing the source activations, changing
padding exclusion, or changing an RHT seed can alter trellis states and bank selectors. The first integrated profile
must reproduce the standalone harness exactly before introducing sequential replay or other alternatives.

### Preparation is explicit and reusable

Calibration tokenization, pristine Hessian capture, shared input geometry, and Sketch-B collection are expensive and
often invariant across rates or codec arms. They need a named, fingerprinted artifact with an explicit lifetime rather
than hidden processor dictionaries.

### The model tree is authoritative

Target modules and same-activation groups must come from model-adapter structure and object relationships. Do not
infer Q/K/V or gate/up groups from string suffixes. Calculation-group metadata should identify modules that receive
the same activation and may share a Gram matrix and transformed input factorization.

### One quantization implementation

The processor and the research harness must call the same QVQ reference/native quantizer, packer, and validators.
Research-only controls may be exposed through explicit configuration, but the script must not reproduce feedback,
bank selection, packing, or reconstruction logic.

### Packed reload is authoritative

Dense reconstructed-weight replay remains a useful diagnostic. Promotion decisions, however, must use the saved and
reloaded packed `QVQLinear` model through its selected inference backend.

### Large-model memory stays bounded

The design must not require a second fully materialized CPU model or retain all source weights. LazyTurtle remains the
checkpoint owner, and source tensors are materialized only for the active layer or bounded module group.

## Public lifecycle and state machine

Add an explicit staged preparation operation rather than overloading `quantize()` with another opaque prepass:

```python
prepared_artifact: QVQPreparedArtifact = model.prepare(
    stage=PrepareStage.QUANTIZATION,
    config=QVQPrepareConfig(
        calibration=calibration,
        validation_calibration=validation_calibration,
        yaqa_calibration=yaqa_calibration,
        calibration_concat_size=None,
        calibration_sort="desc",
        batch_size=1,
        layer_scope=None,
        execution=ExecutionConfig.AUTO,
        cache_dir=None,
    ),
)

model.quantize(
    prepared_artifact=prepared_artifact,
    layer_scope=None,
)
```

`PrepareStage` is the generic lifecycle dispatch enum. `PrepareStage.QUANTIZATION` selects the quantization preparation
contract, and the concrete config selects `QVQPreparationProcessor`. A non-QVQ quantizer can register another processor
behind the same stage without adding another top-level model method.

`model.prepare()` returns one concrete subclass of `BasePreparedArtifact`. The stage and config select the processor
and expected return subclass. QVQ preparation returns `QVQPreparedArtifact`; another quantization method may return its
own subclass with completely different typed fields.

Use the singular name `prepared_artifact`, not `prepared_payload`. The object is a validated lifecycle result with
provenance, parsing, and managed-storage behavior, not an opaque tensor transport bundle.

The model state machine becomes:

```text
LOADED
  -> PREPARING
  -> PREPARED
  -> QUANTIZING
  -> QUANTIZED
  -> SAVED
  -> RELOADED
  -> EVALUATED
```

Failure transitions are transactional:

- preparation failure returns the model to `LOADED` and releases partial captures;
- quantization failure restores the current dense module and leaves the immutable preparation valid when safe;
- packing or serialization failure never marks the model `QUANTIZED` or `SAVED`;
- validation failure rejects the artifact and preserves the last independently accepted baseline;
- cleanup failures are reported without masking the primary exception.

`quantize()` remains backward compatible. If no `prepared_artifact` is supplied, it internally runs the quantization
preparation stage for the minimum required state. Supplying an artifact skips only work proven reusable by its
manifest and fingerprint; it does not bypass validation.

## Prepared artifact

The public return type is `BasePreparedArtifact`; the runtime value is the method-specific subclass selected by the
preparation registry. The artifact is a frozen dataclass. It may contain typed component dataclasses, typed mappings,
detached tensors, or managed read-only cache handles, but never source model weights or untyped processor state.

### Base artifact contract

The common base provides uniform parsing and consumption without forcing every quantizer to produce the same data:

```python
@dataclass(frozen=True, kw_only=True)
class BasePreparedArtifact:
    kind: str
    stage: PrepareStage
    schema_version: int
    producer: str
    fingerprint: str
    manifest: PreparationManifest

    def validate(self, context: PrepareValidationContext) -> None:
        ...

    def to_dict(self) -> dict[str, object]:
        ...

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BasePreparedArtifact":
        ...
```

QVQ aggregates its multiple preparation products in one typed subclass:

```python
@dataclass(frozen=True, kw_only=True)
class QVQPreparedArtifact(BasePreparedArtifact):
    input_hessians: Mapping[str, HessianArtifact]
    input_factors: Mapping[str, FactorizationArtifact]
    sketch_b: SketchBArtifact | None
    propagation: PropagationArtifact | None
    validation_replay: ValidationReplayArtifact | None
    compander: CompanderArtifact | None
```

The internal component types are ordinary frozen dataclasses owned by QVQ; they do not need to inherit
`BasePreparedArtifact` because they are not independent lifecycle return values. Typed mappings support arbitrarily
many modules and calculation groups without weakening the top-level return contract. A component containing external
tensor storage uses an `ArtifactStorage` handle with checksum, dtype, shape, location, and ownership information rather
than an unstructured path.

Calibration and validation datasets are inputs to `model.prepare()`, not fields of `QVQPreparedArtifact`. Their
identity, selected row IDs, preprocessing settings, counts, and digests belong in the manifest. When later
quantization needs reusable validation products, `validation_replay` may contain only derived teacher logits, masks,
or bounded replay handles. It must not retain or duplicate the raw calibration dataset.

An in-memory tensor field must be detached, non-gradient, artifact-owned, and treated as read-only. Freezing the
dataclass does not make a PyTorch tensor immutable, so validation recomputes its checksum before consumption whenever
the artifact crosses a lifecycle or process boundary. Large or shared tensors should prefer `ArtifactStorage` handles
to make ownership and mutation boundaries explicit.

`kind` is a stable serialized discriminator, not a Python class name. Deserialization uses a registry from
`(stage, kind, schema_version)` to the permitted `BasePreparedArtifact` subclass parser. Unknown kinds, unsupported
versions, malformed component mappings, and unexpected subclasses fail closed.

Consumers validate and narrow the return type once at the lifecycle boundary:

```python
qvq_artifact = require_prepared_artifact(
    prepared_artifact,
    artifact_type=QVQPreparedArtifact,
)
```

Its `sketch_b` field is then statically typed. The accessor validates the expected subclass, stage, fingerprint,
manifest, and current model/config context before returning the narrowed artifact.

The required `PreparationManifest` is the authority for the complete object. It records the stage, config, producer,
component fields, dependencies, checksums, storage locations, and lifecycle schema. Removing, replacing, or adding a
component without regenerating the manifest invalidates the artifact. Its own checksum excludes the manifest's
fingerprint field to avoid a self-referential digest.

### Identity and provenance

The manifest records:

- model repository/path, checkpoint shard fingerprints, config fingerprint, and model class;
- tokenizer fingerprint and normalization version;
- calibration, validation, and YAQA dataset fingerprints;
- exact source row IDs, row ordering, and a stable token-batch digest;
- padding-mask digest, valid-token counts, sequence counts, and exclusion counts;
- model-tree version, target full names, object roles, calculation-group IDs, and layer scope;
- QVQ format/version, rate-independent controls, dtype, block size, damping policy, and device family;
- seed policy and all derived seed/sign checksums;
- implementation/schema version and checksums for every stored tensor.

The artifact is rejected when any field affecting its math differs. It must never be accepted based only on matching
tensor shapes.

### Stored QVQ data

Depending on the requested arms, `QVQPreparedArtifact` may contain:

- pristine input Hessians/Grams, shared by exact same-activation groups;
- sample counts and padding-exclusion metadata;
- RHT input signs and transformed Hessians;
- stabilized block-LDL factors, effective damping, retry count, block size, and source-Hessian checksum;
- YAQA input/output Sketch-B factors and collection statistics;
- held-out module-boundary propagation inputs/targets when requested;
- dense-teacher logit references or a separately fingerprinted teacher cache;
- optional model-level frozen compander data.

It must not contain raw calibration/validation datasets or dense source weights. Large derived tensor artifacts may
live in a versioned cache directory and be represented by managed handles rather than one monolithic Python
serialization.

### Shared-geometry cache key

Reusable input preparation is keyed by all math-relevant provenance:

```text
source Hessian checksum/version
+ calculation-group identity
+ input-sign checksum
+ damping policy and effective damping
+ factor block size
+ tensor dtype
+ device/backend
+ QVQ preparation schema version
```

Q/K/V may share input geometry, as may gate/up, when the model tree proves they consume the same activation. Output
signs, output Hessians, output-channel geometry, weights, trellis, and selectors remain module-specific.

## Compatibility profile

The first implementation must provide a `standalone_compat` preparation profile matching the validated Llama 3.2 1B
sweeps:

| Setting | Required value |
|---|---|
| Calibration rows | exact requested source rows |
| Row ordering | descending length, matching `prepare_calibration_dataset` |
| Concatenation | disabled (`None`/`0`) |
| Calibration batch | 1 |
| Sequence truncation | disabled; preserve full source row length |
| Padding | excluded from Hessians and metrics |
| Hessian source | untouched dense model activations |
| QVQ seed | one explicit global seed, defaulting to the sweep's `18240` when requested |
| Quantization math | FP32 RHT/feedback/reference calculations where currently required |
| Damping | exactly matched configuration and retry policy |
| Tail candidates | exactly matched configuration |
| Propagation | disabled unless the arm explicitly enables it |
| Output alignment | disabled unless the arm explicitly enables it |
| Target scope | model-tree-selected modules matching the requested semantic scope |

Current `QVQProcessor` derives module-specific seeds from module names. That remains a valid production policy, but it
is not compatible with the standalone global-seed sweeps. Seed policy must be explicit, serialized, and tested; the
integration must not silently mix the two policies.

The compatibility profile's expected parity is exact for:

- prepared Hessians and sample counts;
- RHT signs and transformed Hessians;
- stabilized factors and effective damping;
- trellis states and packed words;
- bank/family selectors;
- reconstructed FP32 weights and proxy decisions.

Inference may differ only within the established packed-kernel contract: quantization output is exact, while packed
inference must remain finite and within `2e-3` drift from the dense decoded-weight reference.

## Preparation execution

Preparation exposes only the memory/materialization schedule. Device count and parallel placement are orthogonal and
remain automatic:

```python
class ExecutionConfig(str, Enum):
    AUTO = "auto"
    DENSE = "dense"
    PER_LAYER = "per_layer"
```

Do not add `DISTRIBUTED` or backend-specific values. Assigning multiple GPUs does not change preparation math or
materialization semantics; the existing model-tree and subset planner automatically distribute ready work over the
assigned devices.

### `ExecutionConfig.DENSE`

Use when the complete dense source model fits comfortably across the assigned device plan. This mode reproduces the
current research harness most directly:

1. keep an untouched dense source model resident;
2. capture all requested pristine input Hessians in a single forward traversal;
3. capture Sketch-B in a separate backward traversal when YAQA is enabled;
4. build reusable group geometry;
5. release temporary activations and gradients;
6. quantize from the immutable preparation.

No module may be quantized before pristine capture completes.

When multiple GPUs are assigned, the lifecycle may shard or parallelize dense preparation automatically. The caller
does not select a separate distributed mode.

### `ExecutionConfig.PER_LAYER`

Use for checkpoints that require a LazyTurtle/meta shell. For each decoder layer:

1. materialize the untouched dense layer from `LazyTurtle` with `role="quant_source"`;
2. replay the layer's pristine input rows once;
3. capture every calculation group's Hessian from the same activation population;
4. produce and persist the pristine output rows needed by the next layer;
5. build and store the layer's preparation data;
6. quantize and pack the layer;
7. release source weights, hooks, temporary activations, and unneeded shard ranges;
8. advance using the pristine dense outputs, not quantized outputs, while the compatibility profile is active.

This preserves untouched-dense capture without materializing the entire model. The pristine row cache must be bounded,
CPU/disk spillable, and padding-aware. Existing `shell_module_materialize()` and LazyTurtle alias resolution remain
the only checkpoint materialization authority.

Multiple assigned GPUs automatically process independent ready layers/subsets when dependencies and memory permit.
The execution value remains `PER_LAYER` because the materialization boundary is still one layer.

### `ExecutionConfig.AUTO`

Select `DENSE` when the source model is already dense-resident and measured memory headroom is sufficient. Select
`PER_LAYER` for a LazyTurtle/meta shell or when the dense preparation memory estimate exceeds the configured budget.
Report the selected execution value and reason.

After that selection, the model tree and `SubsetPlan` assign complete preparation units over every assigned device. A
unit owns its materialized source tensors, input rows, hooks, captures, factorization, and cleanup.

Requirements:

- no module's Hessian is reduced twice;
- reductions use deterministic FP32 accumulation where exact parity is required;
- conditional MoE experts record routed-token counts and explicit zero-sample behavior;
- common Q/K/V and gate/up captures remain colocated when that avoids transfers;
- cross-device artifacts are copied only after producer events complete;
- per-device memory budgets and spill thresholds are explicit;
- a multi-GPU plan that benchmarks slower than the matched single-GPU plan automatically falls back.

## Hessian and factorization ownership

Shared Hessian capture belongs in a reusable lifecycle attachment or QVQ preparation component, not in the sweep
script. `QVQProcessor` consumes the result.

For each model-tree calculation group:

1. install one masked activation capture at the authoritative shared input;
2. accumulate one FP32 Gram with valid-token masking;
3. normalize once using the recorded sample count;
4. derive each required input-sign transformation;
5. stabilize and factor each unique transformed geometry once;
6. provide immutable references to all consumers;
7. release device copies after the final consumer event.

The factorization helper must perform bounded SPD recovery internally and reuse the successful Cholesky. It returns:

```text
stabilized Hessian
input/output L factors
effective damping
retry count
source checksum and geometry metadata
```

It must not probe stabilization and then repeat the successful factorization. Shape, device, and dtype checks alone
are insufficient; block size and originating Hessian identity are part of the contract.

## YAQA and Sketch-B integration

`prepare_yaqa()` should become a specialized preparation stage rather than an isolated dense-only prepass. The exact
Sketch-B contract remains:

- one gradient sample per independent sequence;
- padding excluded;
- no replacement of the ordinary activation-Hessian calibration stream;
- FP32 factor accumulation unless an explicitly validated alternative is selected;
- input/output factors tied to the exact target module and source model fingerprint;
- activation checkpointing and workspace policy recorded in telemetry.

The initial dense implementation may continue using full-model backward. The scalable roadmap is:

1. batch Sketch-B collection with correct per-sequence semantics and masks;
2. accumulate Gram matrices directly without materializing batch-by-input-by-input intermediates;
3. use one device-side finite flag per batch rather than per-module host synchronizations;
4. make checkpointing memory-budget-aware;
5. stream factors to bounded host/disk storage after their final backward contribution;
6. define routed-token and zero-sample semantics before enabling MoE;
7. add distributed reduction only after exact single-device parity is established.

`QVQPreparedArtifact` must support reuse of one validated Sketch-B collection across compatible V2, V2B2-P32, and
V2B4-P64 YAQA arms. Bank/family selection and trellis results are not reusable preparation data.

For V2B2-P32, preserve both YAQA modes:

- fixed family: reuse the Block-LDLQ-selected alternative family and optimize the P32 selector schedule;
- family reselection: evaluate all alternative families under the complete YAQA objective.

Every banked YAQA result is compared with an independently encoded V2+YAQA oracle. Fallback restores that exact
artifact atomically, not a Block-LDLQ V2 result or a mixed-bank baseline.

## Quantization execution

`QVQProcessor` remains responsible for QVQ-specific execution:

1. validate the preparation against the live model and effective module config;
2. obtain the immutable module/group geometry;
3. materialize the current dense source module if needed;
4. call `quantize_qvq_linear()` once through the supported reference/native dispatch;
5. record exact trellis, selectors, scales, losses, timing, and fallback decisions;
6. stage immutable packed tensors on the host when required;
7. replace the source leaf with `QVQLinear` through the normal finalization path;
8. release source tensors and preparation references after their final consumer;
9. preserve transaction state for rollback.

The processor must consume semantic target metadata supplied by the model tree. BaseQModel should gain only neutral
preparation hooks and state handling; QVQ conditionals should not spread throughout generic lifecycle code.

### Pristine versus sequential geometry

Two explicit policies are required:

- `geometry_source="pristine"`: every Hessian observes the untouched dense model, matching the research sweep;
- `geometry_source="sequential"`: later layers observe accepted quantized predecessors, matching conventional layer
  replay.

Neither is universally superior. They are distinct algorithms and must have separate cache fingerprints and A/B
results. The compatibility profile uses `pristine`.

## Propagation and validation data

Calibration streams have separate purposes and must be independently fingerprinted:

- calibration rows: activation Hessians;
- YAQA rows: Sketch-B/Fisher factors;
- validation rows: module/layer/final-output candidate selection;
- evaluation rows: authoritative reporting only.

Rows may be disjoint by explicit source row IDs. Holding out the final fraction of calibration is not equivalent to an
independent validation set and must be reported as such.

The first supported downstream acceptance callback is teacher-forced final-logit KL on a disjoint validation set. Its
contract must include:

- a frozen dense teacher fingerprint and cached teacher logits or deterministic replay recipe;
- identical token IDs, masks, positions, and model inputs for teacher and candidate;
- bias-aware module math;
- forward KL plus configurable Top-1/Top-5/Top-10 and margin guardrails;
- comparison against the current accepted artifact, not merely canonical bank zero;
- atomic rollback of trellis, selectors, scales, packed tensors, and dense reconstruction;
- no access to the final evaluation rows.

Module-local held-out output MSE may remain a cheaper explicitly named mode. It must not be described as end-to-end
propagation.

## Packing, save, reload, and inference

The lifecycle must use the existing QVQ format contract:

- install a real `QVQLinear`, not a dense reconstructed `nn.Linear`;
- serialize only format-owned tensors and metadata;
- validate fractional rates without integer truncation;
- preserve selector shape/range and bank/family metadata;
- save through supported sharding and offload paths;
- reload through normal backend selection;
- fail closed on missing, stale, or mismatched selector metadata;
- compare live packed output with saved-and-reloaded packed output exactly where deterministic.

The all-zero V2B selector path must remain bit-for-bit canonical V2 through quantization, packing, save, reload, and
every inference backend. Unsupported backend/format combinations must fail clearly or select a tested portable
fallback; they must not reinterpret the payload.

## Sweep orchestration

The replacement comparison script becomes a thin manifest runner. A sweep arm specifies:

```text
source model and tokenizer
prepared artifact or serialized artifact/cache reference
QVQ config and rate
rounding mode
layer/module scope
seed policy
validation callback policy
target device set
output checkpoint and report paths
```

For each `(rate, arm)` it runs a fresh worker/model:

```text
load shell/model
-> validate or create preparation
-> quantize
-> validate live packed model
-> save
-> reload
-> evaluate
-> atomically publish report
```

The outer scheduler assigns independent arms to GPUs. Existing lifecycle planning owns module placement inside one
arm. A script must not manually move individual layers in conflict with `SubsetPlan`.

Preparation may be reused across arms only when the complete fingerprint matches. Mutable candidate state is never
shared across concurrent workers.

## Telemetry and reporting

Every stage emits structured events with a stable run/arm/module identity:

- dataset load/tokenization and row IDs;
- requested and selected `ExecutionConfig`, assigned devices, and selection reason;
- source materialization and release;
- Hessian capture tokens, bytes, and elapsed time;
- shared-group reuse count;
- SPD retries, damping, and factorization time;
- Sketch-B batch/sequence/token progress, checkpoint recompute, storage, and peak memory;
- quantization rate, geometry, objective, trellis batch, and kernel path;
- per-module encode, selector, fallback, and memory metrics;
- packing/finalization, save, reload, and inference backend;
- validation/evaluation partial metrics;
- CPU RAM, pinned RAM, device allocation/reservation, and spill bytes.

Benchmark reports separate:

1. preparation;
2. encode;
3. pack/finalize;
4. save;
5. reload;
6. evaluation.

The standalone harness's core timer must be compared only with equivalent preparation+encode work. Save/reload and
authoritative evaluation are reported separately rather than making the integrated lifecycle appear slower by adding
new correctness stages to one side of the A/B.

## Correctness and coverage gates

### Unit tests

- preparation state transitions, idempotence, invalidation, and failure cleanup;
- `BasePreparedArtifact` subclass registration, serialization, parsing, and checked consumption;
- artifact/manifest schema and component checksum round trip;
- unknown kind/version, wrong subclass, malformed component mapping, missing dependency, and mutated-tensor rejection;
- no raw calibration or validation dataset retained after preparation;
- calibration sort, no-concat, full-length rows, masks, and exact sample counts;
- model-tree target and calculation-group discovery without name-prefix assumptions;
- exact shared-versus-independent Hessian and factor parity;
- factor provenance, block-size mismatch, damping retry, and factorization call count;
- global and module-derived seed policies;
- pristine versus sequential geometry separation;
- dense versus per-layer preparation parity on a tiny model;
- LazyTurtle materialization/release and bounded source ownership;
- YAQA sequence semantics, masks, batching, cache reuse, and failure cleanup;
- V2, V2B2-P32 fixed/reselected, and V2B4-P64 YAQA paths;
- independent bank-zero/V2+YAQA fallback and exact rollback;
- selector range, stride, shape, save/reload, and mutation failures;
- biased propagation callback and current-baseline comparison;
- empty batches, partial chunks, non-finite inputs, and near-tie decisions;
- CPU reference plus CUDA/MPS/MLX guards where supported.

New logic requires measured line and branch coverage. Coverage claims must identify the files/regions measured; do not
claim whole-repository 100% from a focused suite.

### Integration tests

1. tiny model: prepare, quantize, pack, save, reload, infer;
2. Llama 3.2 1B compatibility: standalone versus lifecycle on identical rows and settings;
3. full 16-layer Q/K/V/O scope;
4. all linear modules except embeddings and LM head;
5. V2, V2B2-P32, and V2B4-P64 at supported half-step rates;
6. Block-LDLQ and YAQA factorial arms;
7. single-GPU and multi-GPU exactness;
8. shell/turtle streamed preparation with a sharded checkpoint;
9. interrupted/failing preparation and quantization with no leaked hooks, tensors, or partial packed modules;
10. saved/reloaded Evalution smoke and dense-teacher metric comparison.

### Numerical gates

For the standalone compatibility profile:

- Hessians, signs, factors, trellis states, packed words, and selectors: exact;
- reconstructed quantized weights: exact;
- live versus reloaded packed output: exact when backend execution is deterministic;
- packed inference versus decoded dense reference: finite and maximum allowed drift `<= 2e-3`;
- no quality regression in relative L2, KL, Top-1/Top-5/Top-10, or configured task metrics.

Any nondeterministic GPU operation needs a dense reference, repeated-run spread, and a justified tolerance. Decoded
text alone is never an accuracy gate.

## Performance and memory gates

Integration is promoted only after matched A/B evidence. It is not enough for the lifecycle to be more general.

### Single GPU

- preparation+encode must not regress against the standalone equivalent;
- shared Q/K/V and gate/up Hessian/factor reuse must remain active;
- no repeated immutable bank/codebook construction;
- no per-module full-device synchronization in hot paths;
- preparation cache reuse must reduce total multi-arm sweep time;
- packed inference must meet existing per-shape backend baselines.

### Multi-GPU

- exact output parity with the matched single-GPU policy;
- positive wall-time movement before enabling by default;
- no device left idle while independent ready work exists, subject to memory and dependency constraints;
- no Python thread-pool implementation when the repository's device scheduler/thread infrastructure applies;
- bounded cross-device transfers and explicit CUDA stream ownership.

### Memory

- no full second dense model for shell/turtle execution;
- bounded pristine activation cache with spill accounting;
- bounded Hessian/Sketch-B/factor cache and final-consumer release;
- no retained source module after successful packing;
- peak CPU RAM, pinned RAM, and per-device VRAM reported for every benchmark;
- OOM-risk estimates before materialization, with fail-closed fallback to a lower-memory mode.

## Implementation phases

### Phase 0: lock the oracle

- freeze representative standalone manifests and expected artifacts;
- record exact row IDs, token digests, seeds, configs, and dense metrics;
- add a machine-readable compatibility report;
- split timing into preparation, encode, and evaluation.

Exit: the standalone oracle is reproducible from one manifest.

### Phase 1: preparation API and dense compatibility

- add `PrepareStage`, `BasePreparedArtifact`, its parser registry, preparation state/protocol, and `model.prepare()`;
- move model-tree target resolution and shared Hessian capture behind the lifecycle;
- add explicit seed policy and pristine geometry mode;
- consume `prepared_artifact` in `QVQProcessor`;
- retain existing `quantize()` behavior when no prepared artifact is supplied.

Exit: Llama 3.2 1B dense-resident lifecycle matches the standalone tensors exactly.

### Phase 2: packed lifecycle sweep

- convert the comparison harness to lifecycle calls;
- install `QVQLinear`, save, reload, and evaluate packed models;
- add preparation reuse across rates/arms;
- publish stage-separated telemetry and reports.

Exit: all existing QVQ sweep arms run without direct `quantize_qvq_linear()` or dense weight mutation in the script.

### Phase 3: per-layer pristine preparation

- integrate preparation units with LazyTurtle materialization;
- maintain bounded pristine layer inputs independently of quantized replay;
- add spill, cleanup, and interrupted-run recovery;
- validate dense-resident versus streamed exactness.

Exit: `ExecutionConfig.PER_LAYER` can prepare and quantize a sharded model without full dense CPU/GPU residency.

### Phase 4: YAQA preparation scaling

- expose reusable Sketch-B artifacts through preparation;
- preserve exact dense semantics while adding bounded batching/workspaces;
- support streamed dense layers where mathematically valid;
- define and test MoE routed-token semantics before enabling MoE.

Exit: YAQA no longer requires an unbounded single-device dense lifecycle for supported dense models.

### Phase 5: automatic multi-GPU preparation and quantization

- map preparation units through model-tree subset planning;
- add deterministic reductions and event-safe ownership;
- benchmark single versus multi-GPU on A100-class and RTX 4090 devices where available;
- automatically choose the faster valid plan without introducing a separate execution value.

Exit: multi-GPU improves matched wall time without accuracy or memory regression.

### Phase 6: production downstream selection

- wire disjoint teacher-forced final-logit KL validation;
- cache/fingerprint dense teacher outputs;
- add Top-N and margin guardrails and atomic rollback;
- keep module-local scoring as a separately named lower-cost option.

Exit: propagation-aware selection measures true downstream logits and never consumes evaluation rows.

## Ownership map

| Concern | Owner |
|---|---|
| Public state/API and neutral preparation hooks | `BaseQModel` |
| Base artifact dataclass, registry, manifest, and checked accessor | generic preparation lifecycle |
| Model targets and calculation groups | model adapter `module_tree` |
| Shell/source materialization | `BaseQModel.shell_module_materialize()` and `LazyTurtle` |
| Device/subset scheduling | `ModuleLooper` / `SubsetPlan` |
| QVQ preparation validation and quantization | `QVQProcessor` |
| Hessian/Sketch-B reusable data | typed `QVQPreparedArtifact` fields plus bounded cache storage |
| QVQ math | `gptqmodel.quantization.qvq` |
| Packed module and backend selection | `QVQLinear` and existing backend lifecycle |
| Save/reload/sharding | existing GPT-QModel checkpoint lifecycle |
| Sweep scheduling | thin script/manifest runner |
| Authoritative metrics | reloaded packed model and Evalution integration |

## Non-goals

- Do not change QVQ codec geometry merely to fit the lifecycle.
- Do not make YAQA state part of the inference checkpoint.
- Do not infer same-activation groups from module name strings.
- Do not store raw calibration or validation datasets in `QVQPreparedArtifact`.
- Do not retain source weights in preparation artifacts.
- Do not treat module-local MSE as final-logit propagation.
- Do not include save/reload time in only one side of a quantization-speed A/B.
- Do not enable a multi-GPU or low-memory approximation without measured exactness and performance evidence.
- Do not remove the standalone diagnostic until lifecycle parity and artifact migration are complete.

## Promotion decision

The lifecycle replaces the standalone quantization path only when all of the following hold:

1. compatibility-profile tensors and decisions are exact;
2. packed save/reload inference satisfies the `2e-3` drift contract;
3. matched model-quality metrics do not regress;
4. single-GPU preparation+encode is no slower;
5. multi-arm sweeps are faster through safe preparation reuse;
6. streamed execution keeps memory bounded without changing math;
7. failure injection proves atomic rollback and cleanup;
8. all supported backends fail closed or pass their explicit parity gates;
9. the sweep script contains orchestration only, with no duplicate QVQ quantization implementation.

Until then, the integrated path is opt-in and every report records which lifecycle/profile produced it.
