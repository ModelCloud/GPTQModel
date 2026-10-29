<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Public API proposal: simple operations, composable plans

**Status:** revision 0.5 design sketch, 2026-10-29. None of these new APIs is
implemented by this PR. Names and signatures remain open for discussion.
The [main proposal](component-quantization-design.md) defines numerical semantics;
the [parallel contract](quantization-parallel-execution.md) defines legal layouts;
the [data contract](quantization-data-contract.md) defines `QDataSet`, modality
schemas, lifecycle roles, preparation, streaming, and provenance.

The public surface should have three levels: a direct operation for common work,
an immutable fluent plan for composition, and typed extension contracts for
advanced methods. They all compile to the same resolved plan. Users should not
have to assemble capture hooks, optimizer callbacks, finalizers, or process groups
to quantize a supported model.

## 1. Common operations stay small

```python
model = GPTQModel.load("source-checkpoint")
multimodal_calibration = CalibrationDataSet(
    MultiModalDataSet(image_text_samples),
    route="image_text",
    scenarios=("prefill", "decode"),
)

# One target, one method. The operation compiles and executes a plan.
run = model.quantize(
    target="language",
    recipe=GPTQ(bits=4, group_size=128),
    data=multimodal_calibration,
    devices=Local(gpus="all"),
)
run.save("language-quantized")

# A data-free method does not acquire calibration requirements from the wrapper.
run = model.quantize(target="vision.encoder", recipe=RTN(bits=8))
run.save("mixed-model")
```

An ordinary method configuration is shorthand for `Recipe(method)`. Direct
operations return a run/result object, with installed revision, reports, status,
and saved artifact references. They do not return a partially configured model
builder. Every direct operation can be previewed through the plan form below.
`run.save()` saves that run's assembled revision, including untouched components;
it cannot silently save a later run's changes. `model.save()` explicitly saves
the current assembled revision. Both record validation coverage and provisional
status where applicable.

`quantize` handles selected dense weights. `requantize` requires an explicit
source policy when replacing a packed representation. `repack` is a supported
numerically lossless layout conversion. Adapter generation, norm tuning, and
activation calibration are separately named operations on existing representations.

```python
run = model.requantize(
    target="vision.encoder",
    recipe=GPTQ(bits=4, group_size=64),
    source=SourceSnapshot("source-checkpoint"),
    data=multimodal_calibration,
)
```

The component names in these examples are adapter-provided selections. On an
encoder-free model, `vision.encoder` fails with a report of available owners and
routes. Selecting a vision **route** for calibration does not create vision-only
ownership of a shared multimodal backbone.

## 2. Declarative composition without accidental serialization

```python
plan = (
    Plan()
    .data(calibration=multimodal_calibration)
    .devices(Local(gpus=[0, 1, 2, 3]))
    .add(Quantize(
        target="vision.encoder", recipe=GPTQ(bits=8), name="vision"
    ))
    .add(Quantize(
        target="language", recipe=GPTQ(bits=4), name="language"
    ))
    .evaluate(suite=held_out_suite, data=held_out_data)
)

resolved = plan.compile(model)
print(resolved.explain())
run = model.run(resolved)
run.save("mixed-model")
```

`add()` registers intent; its call order does not demand execution order. The
compiler resolves architecture dependencies, method stages, sources, and writes.
For progressive candidate-conditioned fitting, the language observations wait
for the required vision revision. Native/derived teacher capture can occur at
different legal points. A data-free pair may have no numerical dependency.

All builder methods return a new value. Building a variant never mutates an
earlier plan or a shared recipe. Compilation freezes selectors, data/source
identities, options, stage versions, and the selected execution layout. Running
a compiled plan against a changed composition requires validation/recompilation,
not hidden rebinding to whichever weights happen to be live.

Compilation can resolve read-only metadata, but it does not consume dataset
cursors, execute a model, profile devices, or fit weights. Required materialization,
content validation, capture, and profiling are explicit execution stages. Checks
that depend on unseen samples remain pending until those stages complete; a
plan preview cannot certify unknown coverage or held-out status.

## 3. Explicit ordering and parallel branches

For users who want explicit operation order, provide ordinary named methods:

```python
audio_calibration = CalibrationDataSet(
    AudioDataSet(audio_text_samples), route="audio_text",
)
omni_calibration = CalibrationDataSet(
    MultiModalDataSet(aligned_image_audio_text_samples),
    route="image_audio_text",
)

plan = (
    Plan()
    .data(calibration=omni_calibration)
    .devices(Local(gpus=[0, 1, 2, 3]))
    .parallel(
        Quantize("vision.encoder", GPTQ(bits=8),
                 data=multimodal_calibration, name="vision"),
        Quantize("audio.encoder", GPTQ(bits=8),
                 data=audio_calibration, name="audio"),
    )
    .then(Quantize("language", GPTQ(bits=4), name="language"))
    .then(TuneNorms(
        target="language", source=SourceSnapshot("source-checkpoint"),
        data=norm_training_data, name="norms",
    ))
    .evaluate(suite=held_out_suite, data=held_out_data)
)
```

| Expression | Exact proposed meaning |
| --- | --- |
| `add(operation)` | Add a named operation without an insertion-order edge. Required architecture/method dependencies still apply. |
| `then(operation)` | Add completion edges from the current terminal operations to this operation. Completion means its declared stable installed result/artifacts, not merely a worker submission. |
| `parallel(a, b)` | Fork from the current terminal operations into branches that may overlap; subsequent `then` joins all branch terminals. The compiler rejects an inter-branch dependency or write conflict rather than pretending the branches are independent. |
| `operation.after("name")` | Add a targeted completion dependency without globally serializing unrelated operations. Names resolve within the assembled plan/fragment namespace. |
| `evaluate(suite=..., data=...)` | Add final evaluation of the assembled terminal composition. The suite declares tasks, metrics, and acceptance rules; data binds the `evaluate` role and may instead be supplied as a plan default. |
| `refine(policy)` | Add a declared bounded composition-refinement program with permitted writes, sources, selection data, and stopping criteria. It does not replace intrinsic method optimization. |

`parallel()` permits legal overlap; device count and memory still bound actual
concurrency. The preview shows feasible placement and serialized bottlenecks.
Strict physical-layout requirements belong in the parallelism policy below.
The example assumes an adapter with independent vision/audio owners and the
named routes. The language corpus covers both modalities; an image-text corpus
alone cannot satisfy audio coverage. Shared-backbone owners require a different
selection and must not be presented as independent encoder branches.
Calling `.parallel(vision, language)` is invalid when the selected language
program requires the newly quantized vision output. The error points to the
dependency and suggests `then` or `add`; it must not silently change the objective.

Explicit order can constrain a legal schedule, not override numerical dependencies.
A conflicting order reports a dependency cycle with stage IDs and source roles.
An intentional text-first experiment uses separately declared snapshots/observation
policies; spelling `then(vision)` alone cannot erase its context-change effects.

Observation choice is an explicit part of the experiment. For example, independent
GPTQ fitting against one source's inputs can be requested as follows:

```python
native_inputs = Observations.from_source(SourceSnapshot("source-checkpoint"))
independent = (
    Plan()
    .data(calibration=multimodal_calibration)
    .devices(Local(gpus=[0, 1, 2, 3]))
    .parallel(
        Quantize("vision.encoder", GPTQ(bits=8),
                 observations=native_inputs, name="vision"),
        Quantize("language", GPTQ(bits=4),
                 observations=native_inputs, name="language"),
    )
    .evaluate(suite=held_out_suite, data=held_out_data)
)
```

Here both fitting streams use the named frozen source, so the language fit need
not wait for quantized vision. This is the independent baseline from the main
design, not the progressive default or joint optimization. Evaluation uses the
assembled candidate's own state. `observations=` controls a method's supported
caller-selectable observation policy; it cannot overwrite intrinsic teacher,
clean/noisy-stream, or initializer requirements. Methods with multiple roles
expose typed named bindings and reject ambiguous or unsupported policies. Optional
refinement must separately declare its write scope, source, selection data,
pass/resource budget, and stopping criteria.

The final evaluation reads the chosen terminal revision after all fitting and
refinement. Its results cannot drive automated refitting, early stopping, or
candidate selection. Those decisions consume `train`, `validate`, or `select`
data. A final failure reports unmet criteria; it does not restart refinement on
the same held-out set. Adding a later mutating operation requires evaluation of
the new terminal revision, and any earlier feedback must be recorded as used.

## 4. Stack recipes without flattening their lifecycles

```python
scalar = Recipe(GPTQ(bits=4, group_size=128))
with_adapter = scalar.use(EoRA(rank=32))
higher_precision = scalar.configure(bits=8)

staged_awq = Recipe(AWQ(bits=4, group_size=128)).use(
    StagedGSQ(epochs=2)
)

# Each binding is a QDataSet; the composed program declares its required roles.
staged_data = DataBindings(
    initialize=awq_initialization_data,
    train=reconstruction_data,
    validate=held_out_validation,
)
job = Quantize("language", staged_awq, data=staged_data, name="language")
```

`Recipe.use(*extensions)` is the single composition entry point. Each call includes
typed behavior in a new immutable recipe. Dispatch follows the registered type of
each supplied object, with no category argument for the caller. Building a recipe
does not execute its lifecycle stages.

| Typed object | Contribution to the recipe |
| --- | --- |
| `EoRA(rank=32)` | Adapter fitting stages and their residual/base/data requirements. |
| `StagedGSQ(epochs=2)` | Intrinsic training program, teacher freeze points, initializer, and training/validation roles. |
| `SmoothMAD()` | Supported preparation transforms, observations if required, and explicit write effects. |
| `GGUFFormat()` | Output representation and codec requirements, including exact-payload validation. |

Each type registers a versioned `RecipeExtension` descriptor with its category,
configuration schema, compatibility rules, effects, and lifecycle constraints.
The registry resolves one descriptor unambiguously; future extensions use the
same protocol without coordinator branches for class names. Unknown types,
ambiguous registrations, unsupported combinations, or conflicting singleton
options fail with the offending object identified. Repeatable extensions must
declare that capability and have distinct instance identities. Adding a second
codec or a second nonrepeatable modifier never silently replaces the first.

`recipe.use(a, b)` and `recipe.use(a).use(b)` produce the same immutable
specification. The resolved program determines stage execution order. For
order-sensitive preparation transforms, extension order is an explicit constraint
at their supported insertion points; incompatible constraints fail compilation.
Staged AWQ GSQ still freezes its teacher between scaling and clipping, and EoRA
still consumes the uncorrected base and residual. Required training remains
visible when composition refinement is disabled.

`configure()` returns a recipe with validated changes to its base method's typed
options and revalidates every attached extension. The base method is supplied
to `Recipe(method)`; another quantizer is not an extension that replaces it.
Dataset bindings and devices retain their scoped `.data(...)` and
`.devices(Local(...))` entry points. They are not recipe extensions.

Preparation and export use the same composition operation:

```python
gguf_recipe = (
    Recipe(GGUF(quant_type="Q4_K"))
    .use(SmoothMAD())
    .use(GGUFFormat())
)
```

The method/codec validates this particular combination. A preparation extension
declares its write scope and cannot change unselected neighbors. A format
extension specifies the output representation; any quantization fused with
encoding remains a declared stage of the selected method. A global rotation or
structural change can instead be a plan operation with its full target scope.

A separately requested post-quantization adapter uses `GenerateAdapter(...)` on
the packed base and explicit source/data roles. It is not the same request as
requantizing the base with a modifier.

## 5. Selectors and overrides stay explicit

Use typed semantic selectors for the common cases and escaped exact paths/globs
when needed. Selectors resolve to canonical ownership and operator roles, not
just `nn.Linear` paths. Globs, regular expressions, and exact names must not share
an ambiguous string syntax.

```python
text_rules = (
    Rules()
    .preserve(Select.role("router"))
    .override(Select.role("attention_projection"), Recipe(GPTQ(bits=8)))
)

job = Quantize(
    target="language",
    recipe=Recipe(GPTQ(bits=4, group_size=128)),
    rules=text_rules,
    name="language",
)
```

Rules are scoped to the operation's target. Precedence is plugin defaults, then
explicit base recipe configuration, then explicitly selected overrides. Overlapping
overrides that disagree require an explicit priority or disjoint selectors;
insertion order is not precedence. Preservation is a hard write exclusion and
applies to coupled transforms too. A deliberately different write scope requires
an explicit new plan edit, not a silent exception inside a method.
A recipe-valued override replaces the selected recipe, including its modifiers.
To retain a recipe while changing selected fields, pass a value derived with
that recipe's `configure()`; do not guess which nested defaults should merge.

Zero matches, ambiguous ownership, missing routes, unsupported selected operators,
and conflicting shared-weight recipes fail with a concrete report. Optional
targets can be marked optional explicitly and appear as skipped in the preview.
An error such as “unsupported model” should instead identify the failed operator,
method stage, numerical format, or missing replay/parallel capability.

Data selection is separate from weight selection. `QDataSet` describes source,
schema/modalities, identity, streaming/replay capabilities, and per-run cursors.
`TextDataSet`, `ImageDataSet`, `AudioDataSet`, `VideoDataSet`,
`MultiModalDataSet`, and custom subclasses validate common sample structures.
`CalibrationDataSet(QDataSet)` is a delegating role-aware view, so calibration
can wrap any modality without multiple-inheritance combinations.

A plan-wide `.data(calibration=...)` call supplies a default role binding; an
operation can supply a plain `QDataSet` when it has exactly one unambiguous data
requirement or a `DataBindings` value for several roles:

```python
gsq_data = DataBindings(
    initialize=initialization_data,
    train=reconstruction_data,
    validate=held_out_validation,
)
staged_gsq = Recipe(GPTQ(bits=4)).use(StagedGSQ(epochs=2))
job = Quantize("language", staged_gsq, data=gsq_data, name="language")
```

Observation/calibration, training, initialization, adapter, validation,
selection, profiling, and final evaluation data are distinct typed requirements.
Plugins can register namespaced roles without new top-level API parameters.
Explicit aliases record intentional reuse; roles requiring disjoint samples
reject overlap. A data-free method does not open a default dataset or initialize
a tokenizer/processor merely because a plan binding exists.

Requirements come from the complete composed program. A staged initializer may
use `initialize` in place of the base method's ordinary `calibration` role;
composition does not blindly union every standalone requirement. Unknown roles
and unused operation-specific bindings fail early. Plan defaults may be unused
by an operation, but `explain()` shows their actual consumers. A plain dataset is
never silently copied into every role of a multi-role recipe.

`Plan.calibrate(dataset)` may be convenience sugar for
`Plan.data(calibration=dataset)`, but the resolved and serialized form is a role
binding. Architecture adapters, rather than the dataset or quantizer, compile
raw samples into route-specific model inputs. Preparation identity includes the
processor/tokenizer/template and modality transforms so caches cannot cross
incompatible routes.

## 6. Multi-GPU use is easy; exact layouts are available

The common request selects devices and budgets, then lets compilation choose
supported parallelism, recording the result. Fluent plans use
`.devices(Local(...))`; direct operations use `devices=Local(...)`.

```python
devices = Local(
    gpus=[0, 1, 2, 3],
    memory_per_gpu="70GiB",
    host_memory="256GiB",
    scratch="/scratch/quantization",
)
plan = plan.devices(devices).parallelism(Parallelism.auto())
```

Automatic placement chooses among declared legal layouts. It never changes the
algorithm, silently approximates statistics, or invents distributed support for
a single-device method. Device names refer to the supplied/local resource pool;
the API does not implicitly allocate cloud infrastructure. External launchers
can supply a typed distributed device context through the same `devices` entry
point for multiple nodes.

Advanced policies can require a particular stage layout:

```python
parallel = (
    Parallelism.auto()
    .for_stage(
        "language/observe", DataParallel(workers=4), required=True
    )
    .for_stage(
        "language/gsq/attention", DataParallel(workers=4), required=True
    )
)
plan = plan.parallelism(parallel)
```

Stage IDs above are illustrative IDs exposed by a selected plugin's preview,
not universal required stage names. A role selector can select compatible stages
without hardcoding IDs, but unknown or ambiguous matches fail. Each requirement
is checked against the resolved method, architecture, data, and resources.
Unsupported layouts report alternatives; a `required=True` request never falls
back silently.

Mesh layouts for supported tensor/expert/pipeline/context partitioning use typed
axis and ownership descriptors from the parallel contract. Configuration must
distinguish model partitioning from data replication and independent expert
fitting. Avoid a single `parallel=True` switch that conceals all three.

## 7. Reusable fragments and configuration files

```python
def perception_plan(vision_recipe, audio_recipe, vision_data, audio_data):
    return Plan().parallel(
        Quantize("vision.encoder", vision_recipe, data=vision_data, name="vision"),
        Quantize("audio.encoder", audio_recipe, data=audio_data, name="audio"),
    )

language_data = DataBindings(
    calibration=omni_calibration,
    adapter=omni_calibration.alias(reason="shared EoRA observation corpus"),
)
plan = (
    Plan()
    .devices(devices)
    .include(
        perception_plan(higher_precision, higher_precision,
                        multimodal_calibration, audio_calibration),
        namespace="senses",
    )
    .then(Quantize("language", with_adapter, rules=text_rules,
                   data=language_data, name="language"))
    .evaluate(suite=held_out_suite, data=held_out_data)
)
```

Fragments are values with named inputs/outputs and local operation IDs. Including
one namespaces operation IDs (`senses/vision`), while component selectors continue
to refer to the declared model owners unless explicitly rebound. Conflicting
fragment defaults require a deliberate override at inclusion. Including a plan
does not execute it or mutate the model.

Python and serialized specifications represent the same versioned graph and
typed options. Serialization uses registry IDs, method/schema versions, dataset
descriptors, and named bindings; it does not pickle arbitrary Python callbacks or
embed unbounded calibration tensors. In-memory datasets must be bound explicitly
or materialized into an identified dataset artifact before portable execution.

```yaml
schema: gptqmodel.plan/v1
datasets:
  multimodal_calibration:
    binding: multimodal_calibration
    schema: image-text-conversation/v1
data:
  calibration: {dataset: multimodal_calibration}
devices:
  kind: local
  gpus: [0, 1, 2, 3]
parallelism: auto
operations:
  - id: vision
    kind: quantize
    target: {component: vision.encoder}
    recipe: {method: gptqmodel/gptq@1, bits: 8}
  - id: language
    kind: quantize
    target: {component: language}
    after: [vision]
    recipe: {method: gptqmodel/gptq@1, bits: 4, group_size: 128}
```

This is an explicitly ordered two-operation plan. An unordered `add` plan omits
user ordering edges but still acquires mandatory numerical dependencies at
compilation. List order in the file is not an implicit correctness constraint.
Fluent `then` and `parallel` serialize to explicit edges/groups, not opaque method
call history. Editing resource choices produces a new execution layout; editing
teacher/recipe/data choices changes the numerical plan identity as well.

Recipe composition serializes typed extensions, not Python method names. For
example, the EoRA recipe above has the following recipe value; its containing
operation must also bind the declared `calibration` and `adapter` roles:

```yaml
method: gptqmodel/gptq@1
bits: 4
group_size: 128
extensions:
  - type: gptqmodel/eora@1
    rank: 32
```

Extension list order is preserved for declared order-sensitive constraints. The
same registry and compatibility checks apply to Python and serialized plans;
deserialization cannot execute unregistered callables or skip validation.

## 8. Preview, errors, and extension contracts

`explain()` should answer: what changes, what executes, why this order, which
teacher/data each stage uses, what can overlap, what needs a collective, what
cannot fit, and what validation will be performed. Offer a compact summary and
an expandable graph/table, both derived from the same resolved plan.

An illustrative preview might show:

```text
senses/vision   writes vision.encoder   GPUs 0-1   capture replicas + local solves
senses/audio    writes audio.encoder    GPUs 2-3   capture replicas + local solves
language       waits for both encoder revisions  supported 4-GPU method program
evaluation     reads final composition           held-out task checks

Preserved: selected routers, unselected components, declared aliases
Unmeasured: estimated duration until a cost profile is supplied
```

Actual placement depends on method capabilities and budgets; this is not a
performance prediction. The preview includes all coupled write effects and
reports if a chosen recipe requires an additional source or a wider scope.

Future methods register typed configuration, lifecycle compilation, artifact and
codec schemas, data requirements, parallel capabilities, and conformance cases.
They participate in the same recipe/plan API without adding method-name branches
to the coordinator or new top-level flags for every option. New behavior that
cannot be expressed requires a versioned contract extension, not a raw hook that
bypasses planning.

Proposed API checks include typed `.use()` dispatch, chained/variadic/file
equivalence, unknown/ambiguous/duplicate extension rejection, full recipe
revalidation after `configure()`, direct/plan semantic equivalence, immutable reuse,
fragment namespacing, Python/file round-trips with explicit bindings, stable
resolved IDs, override-conflict diagnostics, invalid parallel groups, cycle
explanations, unsupported required layouts, and correct reporting of zero-match
or shared multimodal selectors. Dataset checks cover single-role shorthand,
multi-role bindings, explicit aliases/disjointness, custom schemas, and streaming
cursor compatibility. These are design acceptance criteria, not tests of an
implemented API.
