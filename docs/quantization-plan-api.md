<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Public API proposal: simple operations, composable plans

**Status:** revision 0.3 design sketch, 2026-10-27. None of these new APIs is
implemented by this PR. Names and signatures remain open for discussion.
The [main proposal](component-quantization-design.md) defines numerical semantics;
the [parallel contract](quantization-parallel-execution.md) defines legal layouts.

The public surface should have three levels: a direct operation for common work,
an immutable fluent plan for composition, and typed extension contracts for
advanced methods. They all compile to the same resolved plan. Users should not
have to assemble capture hooks, optimizer callbacks, finalizers, or process groups
to quantize a supported model.

## 1. Common operations stay small

```python
model = GPTQModel.load("source-checkpoint")

# One target, one method. The operation compiles and executes a plan.
run = model.quantize(
    target="language",
    recipe=GPTQ(bits=4, group_size=128),
    calibration=multimodal_calibration,
    resources=Resources.local(gpus="all"),
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
    calibration=multimodal_calibration,
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
    .calibrate(multimodal_calibration)
    .resources(Resources.local(gpus=[0, 1, 2, 3]))
    .add(Quantize(
        target="vision.encoder", recipe=GPTQ(bits=8), name="vision"
    ))
    .add(Quantize(
        target="language", recipe=GPTQ(bits=4), name="language"
    ))
    .evaluate(held_out_suite)
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

## 3. Explicit ordering and parallel branches

For users who want explicit operation order, provide ordinary named methods:

```python
plan = (
    Plan()
    .calibrate(multimodal_calibration)
    .resources(Resources.local(gpus=[0, 1, 2, 3]))
    .parallel(
        Quantize("vision.encoder", GPTQ(bits=8), name="vision"),
        Quantize("audio.encoder", GPTQ(bits=8), name="audio"),
    )
    .then(Quantize("language", GPTQ(bits=4), name="language"))
    .then(TuneNorms(
        target="language", source=SourceSnapshot("source-checkpoint"),
        data=norm_training_data, name="norms",
    ))
    .evaluate(held_out_suite)
)
```

| Expression | Exact proposed meaning |
| --- | --- |
| `add(operation)` | Add a named operation without an insertion-order edge. Required architecture/method dependencies still apply. |
| `then(operation)` | Add completion edges from the current terminal operations to this operation. Completion means its declared stable installed result/artifacts, not merely a worker submission. |
| `parallel(a, b)` | Fork from the current terminal operations into branches that may overlap; subsequent `then` joins all branch terminals. The compiler rejects an inter-branch dependency or write conflict rather than pretending the branches are independent. |
| `operation.after("name")` | Add a targeted completion dependency without globally serializing unrelated operations. Names resolve within the assembled plan/fragment namespace. |
| `evaluate(suite)` | Add evaluation of the assembled terminal composition, with explicit data/criteria. It does not silently add a refitting operation. |
| `refine(policy)` | Add a declared bounded composition-refinement program with permitted writes, sources, selection data, and stopping criteria. It does not replace intrinsic method optimization. |

`parallel()` permits legal overlap; device count and memory still bound actual
concurrency. The preview shows feasible placement and serialized bottlenecks.
Strict physical-layout requirements belong in the parallelism policy below.
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
    .calibrate(multimodal_calibration)
    .resources(Resources.local(gpus=[0, 1, 2, 3]))
    .parallel(
        Quantize("vision.encoder", GPTQ(bits=8),
                 observations=native_inputs, name="vision"),
        Quantize("language", GPTQ(bits=4),
                 observations=native_inputs, name="language"),
    )
    .evaluate(held_out_suite)
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

## 4. Stack recipes without flattening their lifecycles

```python
scalar = Recipe(GPTQ(bits=4, group_size=128))
with_adapter = scalar.with_modifier(EoRA(rank=32))
higher_precision = scalar.configure(bits=8)

staged_awq = Recipe(AWQ(bits=4, group_size=128)).with_modifier(
    StagedGSQ(epochs=2)
)

job = Quantize("language", staged_awq, name="language")
```

`configure()` returns a recipe with validated changes to its base method's typed
options. Unknown/method-inapplicable keys fail; there is no giant bag of unrelated
parameters. Modifier options belong to their own typed objects and have stable
names/versions. Duplicate incompatible modifiers are errors, not last-writer wins.

`with_modifier()` is declarative composition, not a promise to run the modifier
strictly after the quantizer. Staged AWQ GSQ must capture a teacher between scaling
and clipping; EoRA needs residual/uncorrected-base handoffs. The compiler expands
the compatible combined lifecycle. Required training is visible even when
composition refinement is disabled.

Supported preparation and export can also be named explicitly:

```python
gguf_recipe = (
    Recipe(GGUF(quant_type="Q4_K"))
    .prepare(SmoothMAD())
    .export(GGUFFormat())
)
```

The method/codec validates this particular combination. `prepare` declares
transforms and their write scope; it is not permission to change unselected
neighbors. `export` specifies output representation; it does not add a second
lossy quantizer to an already-fitted payload. A global rotation or structural
change can instead be an explicit plan operation with its full target scope.

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

Data selection is separate from weight selection. A plan-wide calibration suite
is a default binding; an operation can supply its own `calibration=` or a method's
named `data=` roles. Training, initialization, adapter, selection, and held-out
data are distinct typed requirements. A data-free method does not consume a
default calibration suite just because one is present.

## 6. Multi-GPU use is easy; exact layouts are available

The common request supplies resources and lets compilation choose supported
parallelism, recording the result:

```python
resources = Resources.local(
    gpus=[0, 1, 2, 3],
    memory_per_gpu="70GiB",
    host_memory="256GiB",
    scratch="/scratch/quantization",
)
plan = plan.resources(resources).parallelism(Parallelism.auto())
```

Automatic placement chooses among declared legal layouts. It never changes the
algorithm, silently approximates statistics, or invents distributed support for
a single-device method. Device names refer to the supplied/local resource pool;
the API does not implicitly allocate cloud infrastructure. External launchers
can supply a typed distributed resource context for multiple nodes.

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
def perception_plan(vision_recipe, audio_recipe):
    return Plan().parallel(
        Quantize("vision.encoder", vision_recipe, name="vision"),
        Quantize("audio.encoder", audio_recipe, name="audio"),
    )

plan = (
    Plan()
    .calibrate(multimodal_calibration)
    .resources(resources)
    .include(perception_plan(higher_precision, higher_precision), namespace="senses")
    .then(Quantize("language", with_adapter, rules=text_rules, name="language"))
    .evaluate(held_out_suite)
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
data:
  calibration: {binding: multimodal_calibration}
resources:
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
codec schemas, parallel capabilities, and conformance cases. They participate in
the same recipe/plan API without adding method-name branches to the coordinator
or new top-level flags for every option. New behavior that cannot be expressed
requires a versioned contract extension, not a raw hook that bypasses planning.

Proposed API checks include direct/plan semantic equivalence, immutable reuse,
fragment namespacing, Python/file round-trips with explicit bindings, stable
resolved IDs, override-conflict diagnostics, invalid parallel groups, cycle
explanations, unsupported required layouts, and correct reporting of zero-match
or shared multimodal selectors. These are design acceptance criteria, not tests
of an implemented API.
