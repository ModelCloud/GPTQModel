# Quantization Protocol

## Overview

This document proposes a next-generation quantization configuration protocol for `gptqmodel`.

The protocol is designed to be:

- clean and concise for humans
- pipeline / stage based
- explicit about matching and override behavior
- flexible about quantization method vs exported representation
- future-proof for weight, activation, output, and KV-cache quantization

The user-facing protocol root is intentionally shallow. It consists of:

- `version`
- `stages`

It may be authored through:

- a Python DSL
- YAML / JSON serialization of the same protocol

The Python and YAML forms below describe the same protocol.
Python is the ergonomic builder API.
YAML is the portable serialized form.


## Design Goals

1. One matching system only.
   Rules match model objects. Stages do not rematch. Actions do not rematch the whole model in normal use.

2. Keep the common case short.
   The common case should need only:
   - `match`
   - `weight` / `input` / `output` / `kv_cache`
   - `prepare`
   - `quantize`
   - `export`

3. Make overrides readable.
   A narrower rule should be able to skip quantization, replace defaults, or stop later rules without confusing `+` / `-` syntax.

4. Make partial overrides cheap.
   A narrower rule should be able to override only `bits`, `group_size`, or another single leaf field without restating the full quantizer configuration.

5. Separate quantization from representation.
   `quantize` answers how quantized values are produced.
   `export` answers how those values are encoded into final tensors and metadata.

6. Keep backend-specific terms internal.
   Terms such as `*input_quantizer`, `*weight_quantizer`, or packer-specific tensor names should not be the primary user-facing API.


## Protocol Root

Python:

```python
version = 2

stages = [
    Stage(
        name="ptq",
        rules=[
            Rule(
                match="*",
                aliases=None,
                actions=[],
                stop=False,
                weight=None,
                input=None,
                output=None,
                kv_cache=None,
            ),
        ],
    ),
]
```

YAML:

```yaml
version: 2
stages:
  - name: ptq
    rules:
      - match: "*"
        aliases: null
        actions: []
        stop: false
        weight: null
        input: null
        output: null
        kv_cache: null
```

A stage is an ordered execution boundary.
A rule is the only normal matcher.
Each rule may configure one or more tensor targets.


## Match Selectors

`Rule.match` may be either:

- a single selector string
- a list of selector strings

Selector prefixes:

- no prefix or `+:` means positive/include
- `-:` means negative/exclude

This lets one rule express "match everything except ..." without adding a second skip rule.

Python:

```python
Rule(
    match=["*", "-:.*layer2.*"],
    weight={
        "quantize": gptq(bits=4, sym=True, group_size=128),
        "export": {"format": "gptq"},
    },
)
```

YAML:

```yaml
- match:
    - "*"
    - "-:.*layer2.*"
  weight:
    quantize:
      method: gptq
      bits: 4
      sym: true
      group_size: 128
    export:
      format: gptq
```

Recommended semantics:

- a rule matches if at least one positive selector matches
- any matching negative selector removes that module from the rule
- `*` is a special match-all shorthand
- every other selector string is interpreted as regex by default
- for exact module-name matches, use an anchored escaped regex such as `^model\.layers\.0\.self_attn\.q_proj$`


## Internal Implementation

An implementation may compile the user-facing protocol into an internal typed object such as:

```python
Plan(version=2, stages=[...])
```

That internal root object is for parser/runtime organization.
It should not be required in user-facing examples or config files.
Normal user configs have one protocol root for one quantization run or artifact, not multiple user-facing plans.


## Authoring Surfaces

The protocol should support both of these as first-class authoring surfaces:

- Python DSL for programmatic authoring, helpers, and composition
- YAML for checked-in configs, serialization, export metadata, and non-Python tooling

Example equivalence:

Python:

```python
Rule(
    match="*",
    weight={
        "quantize": gptq(bits=4, sym=True, group_size=128),
        "export": {"format": "gptq"},
    },
)
```

YAML:

```yaml
match: "*"
weight:
  quantize:
    method: gptq
    bits: 4
    sym: true
    group_size: 128
  export:
    format: gptq
```


## Stages

A `Stage` exists to define:

- execution order
- calibration / replay boundary
- save / emit boundary

A stage does not introduce a second targeting system.

Example:

Python:

```python
stages = [
    Stage(
        name="balance",
        rules=[
            Rule(
                match=".*self_attn$",
                actions=[smoothquant(alpha=0.5)],
            ),
        ],
    ),
    Stage(
        name="ptq",
        rules=[
            Rule(
                match="*",
                weight={
                    "quantize": gptq(bits=4, sym=True, group_size=128),
                    "export": {"format": "gptq"},
                },
            ),
        ],
    ),
]
```

YAML:

```yaml
stages:
  - name: balance
    rules:
      - match: ".*self_attn$"
        actions:
          - method: smoothquant
            alpha: 0.5
  - name: ptq
    rules:
      - match: "*"
        weight:
          quantize:
            method: gptq
            bits: 4
            sym: true
            group_size: 128
          export:
            format: gptq
```


## Rules

A `Rule` contains:

- `match`: the only normal matcher
- optional `aliases`: named references relative to the matched object
- optional `actions`: rule-scoped operations
- optional tensor-target sections:
  - `weight`
  - `input`
  - `output`
  - `kv_cache`
- optional `stop`: stop applying later rules to the same matched object

Fields omitted by a narrower rule inherit from earlier matching rules unless an explicit replace mode is used.

Recommended shape:

Python:

```python
Rule(
    match="*",
    aliases=None,
    actions=[],
    stop=False,
    weight={...},
    input={...},
    output={...},
    kv_cache={...},
)
```

YAML:

```yaml
match: "*"
aliases: null
actions: []
stop: false
weight: {}
input: {}
output: {}
kv_cache: {}
```


## Matching

Recommended match forms:

- exact module path: `"model.layers.0.self_attn.q_proj"`
- wildcard / glob: `"*"` or `"*.q_proj"`
- regex: `".*self_attn$"`

Rules are evaluated top-to-bottom inside a stage.

There is no stage-level matcher and no normal action-level global matcher.


## Aliases

`aliases` is optional.

It exists only to name reusable relative references under the matched object.
It is not a second model-wide matching language.

Example:

Python:

```python
Rule(
    match=".*self_attn$",
    aliases={"proj": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    actions=[
        record_stats(targets="@proj"),
        inspect_outliers(targets="@proj"),
    ],
)
```

YAML:

```yaml
match: ".*self_attn$"
aliases:
  proj:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
actions:
  - method: record_stats
    targets: "@proj"
  - method: inspect_outliers
    targets: "@proj"
```

Use `aliases` only when the same relative subset must be reused.
If the action naturally operates on the matched object, omit `aliases`.


## Actions

`actions` is a rule-scoped list of operations that run in the context of the rule match.

Examples:

- `smoothquant(alpha=0.5)`
- `awq_balance(ratio=0.7)`
- `calibrate_router(...)`

Default behavior:

- an action operates on the current rule match
- an action does not rematch the whole model
- if an action needs a reusable relative subset, it may use rule `aliases`
- actions should prefer the canonical structure of the matched object over user-written sub-matching in the common case

Important:

- `actions` are for rule-scoped or cross-target behavior
- `prepare` is for target-local pre-quant behavior

This keeps placement clear:

- SmoothQuant or AWQ-like balancing: `actions`
- local weight clip / pad / smoother: `weight.prepare`


## Tensor Targets

The protocol supports these first-class tensor targets:

- `weight`
- `input`
- `output`
- `kv_cache`

This makes the protocol future-proof for:

- weight-only quantization
- activation quantization
- output quantization
- cache quantization


## Target Sections

Each tensor target may define:

- `prepare`
- `quantize`
- `export`

Recommended shape:

Python:

```python
weight={
    "prepare": [...],     # optional
    "quantize": ...,      # optional
    "export": ...,        # optional
}
```

YAML:

```yaml
weight:
  prepa []
  quantize: null
  export: null
```


### `prepare`

`prepare` is for target-local pre-quant transformations.

Examples that belong in `weight.prepare`:

- `clip.mad(k=2.75)`
- `clip.percentile(percentile=99.5)`
- `pad.columns(multiple=4, semantic=True)`

Examples that belong in `input.prepare`:

- `clamp.range(min=-6, max=6)`
- `normalize.rms(eps=1e-5)`

Placement rule:

- local target-only modification -> `prepare`
- cross-target or rule-context modification -> `actions`


### `quantize`

`quantize` defines how the target is quantized.

Examples:

- `gptq(bits=4, sym=True, group_size=128)`
- `rtn(bits=4, sym=True)`
- `mxfp4(mode="dynamic", block_size=32, scale_bits=8)`
- `int8(calibration=observer("max"))`
- `skip()`

`skip()` is target-scoped.
It does not remove `prepare` and it does not disable other targets.

Important:

- `quantize` is a structured object, not a replace-only scalar
- later rules may patch only specific quantizer fields
- omitted fields inherit from earlier matching rules

Example base quantizer:

```yaml
weight:
  quantize:
    method: gptq
    bits: 4
    sym: true
    group_size: 128
```

Example narrower override:

```yaml
weight:
  quantize:
    bits: 8
```

The override changes only `bits`.
It does not require restating `method`, `sym`, or `group_size`.


### `quantize.fallback`

`fallback` belongs inside `quantize`.

It is not:

- an `action`
- a `prepare` step
- an `export` setting

It is a quantizer-local fallback policy for methods that depend on calibration or activation evidence and may not have enough usable samples for every matched unit.

This is the right place because fallback changes how the quantizer solves a target when evidence is insufficient.
It does not change which modules are matched, and it does not change the exported format family.

Primary use cases:

- GPTQ with too few Hessian / activation samples for a module
- AWQ with missing or too-sparse captured activations for a layer group
- future activation-aware weight quantizers
- MoE routing cases where some experts receive little or no calibration traffic

Recommended protocol shape:

Python:

```python
Rule(
    match="*",
    weight={
        "quantize": {
            "method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "fallback": {
                "strategy": "rtn",
                "threshold": "0.5%",
            },
        },
        "export": {
            "format": "gptq",
        },
    },
)
```

YAML:

```yaml
- match: "*"
  weight:
    quantize:
      method: gptq
      bits: 4
      group_size: 128
      sym: true
      fallback:
        strategy: rtn
        threshold: 0.5%
    export:
      format: gptq
```

Recommended fields:

- `strategy`: fallback quantization strategy
- `threshold`: minimum evidence threshold before fallback triggers
- `smooth`: optional smoothing used only inside the fallback path

Example:

```yaml
weight:
  quantize:
    method: awq
    bits: 4
    group_size: 128
    fallback:
      strategy: rtn
      threshold: 1.0%
      smooth:
        type: mad
        k: 2.75
```

Recommended threshold semantics:

- integer / float: absolute minimum observed samples or tokens
- percent string such as `"0.5%"`: minimum observed coverage relative to expected calibration traffic
- `true`: enable quantizer default threshold
- `false` or `null`: disable fallback

Initial runtime contract:

- GPTQ: evaluate fallback per matched module
- AWQ: evaluate fallback at the quantizer's natural scaling group or layer subgroup
- future methods: evaluate fallback at the quantizer's natural solve unit

The protocol should not force one global fallback scope.
Fallback should use the quantizer's native solve scope.

Important separation:

- `quantize.method = gptq` with `fallback.strategy = rtn` means:
  GPTQ is still the primary method
- if the module or group is under-sampled, fallback quantization uses RTN-like weight-only solving
- the rule's `export` still controls the final encoded representation

Example:

```yaml
weight:
  quantize:
    method: gptq
    bits: 4
    fallback:
      strategy: rtn
      threshold: 0.5%
  export:
    format: gptq
```

This does not mean "export as RTN".
It means:

- primary solve path: GPTQ
- low-evidence fallback solve path: RTN
- final export family: GPTQ

That matches current `gptqmodel` behavior more closely than treating fallback as a separate stage or a weight-only top-level config.

Patch-first override behavior should apply here too.

Example base rule:

```yaml
- match: "*"
  weight:
    quantize:
      method: gptq
      bits: 4
      group_size: 128
      fallback:
        strategy: rtn
        threshold: 0.5%
```

Example narrower MoE override:

```yaml
- match: ".*experts\\.[0-9]+\\..*"
  weight:
    quantize:
      fallback:
        threshold: 2.0%
```

Effective result for expert modules:

- `method = gptq`
- `bits = 4`
- `group_size = 128`
- `fallback.strategy = rtn`
- `fallback.threshold = 2.0%`

This is how fallback should fit into the new protocol:

- nested under `target.quantize`
- inherited and patchable like other quantizer fields
- supported only for quantizers that actually depend on calibration / activations
- independent from `export`


### `export`

`export` defines the final encoded representation for the target.

The canonical form of `export` should be a structured object.
String export should be treated only as shorthand for very simple cases.

Canonical fields:

- `format`: logical exported family such as `gptq`, `awq`, `fp8`, `fp4`, `gguf`, `native`
- `variant`: family-specific subtype such as `gemm`, `gemv`, `e4m3fn`, `e5m2`, `nvfp4`, `mxfp4`, `q4_k_m`
- `impl`: concrete exporter or runtime implementation such as `default`, `llm_awq`, `marlin`, `transformer_engine`, `modelopt`
- `version`: exporter layout / schema version
- `options`: exporter-specific knobs that should not be promoted to top-level DSL fields

Python:

```python
weight={
    "export": {
        "format": "awq",
        "variant": "gemm",
        "impl": "llm_awq",
        "version": 2,
    },
}
```

YAML:

```yaml
weight:
  export:
    format: awq
    variant: gemm
    impl: llm_awq
    version: 2
```

Examples:

- `{"format": "gptq"}`
- `{"format": "awq", "variant": "gemm"}`
- `{"format": "awq", "variant": "gemv"}`
- `{"format": "fp8", "variant": "e4m3fn", "impl": "transformer_engine"}`
- `{"format": "fp4", "variant": "nvfp4", "impl": "modelopt"}`
- `{"format": "gguf", "variant": "q4_k_m"}`

Shorthand:

- `"gptq"` == `{"format": "gptq"}`
- `"native"` == `{"format": "native"}`

If omitted, the engine may use the quantizer's native export.

Like `quantize`, `export` should be patchable by narrower rules.

Example:

```yaml
- match: "*"
  weight:
    export:
      format: awq
      variant: gemm
      impl: llm_awq
      version: 2

- match: ".*small_proj$"
  weight:
    export:
      variant: gemv
```

Effective result for `small_proj`:

- `format = awq`
- `variant = gemv`
- `impl = llm_awq`
- `version = 2`


## Patch-First Override Model

Rules should be treated as patches over an accumulated effective configuration.

This is the main simplification for dynamic overrides.

Common case:

- a broad rule defines defaults
- a narrower rule patches only the fields it wants to change
- unchanged fields inherit automatically

Python:

```python
Rule(
    match="*",
    weight={
        "quantize": gptq(bits=4, sym=True, group_size=128),
        "export": {"format": "gptq"},
    },
)

Rule(
    match=".*down_proj$",
    weight={
        "quantize": {"bits": 3},
    },
)
```

YAML:

```yaml
- match: "*"
  weight:
    quantize:
      method: gptq
      bits: 4
      sym: true
      group_size: 128
    export:
      format: gptq

- match: ".*down_proj$"
  weight:
    quantize:
      bits: 3
```

Effective result for `down_proj`:

- `method = gptq`
- `bits = 3`
- `sym = true`
- `group_size = 128`
- `export.format = gptq`

This is the intended replacement for the current `gptqmodel` dynamic override style where a base rule applies to all modules and narrower matches override only selected fields.


## Advanced Replace Mode

Patch merging should be the default.
Explicit replacement should be available only as an escape hatch.

Recommended advanced control:

- `mode: replace`

Python:

```python
Rule(
    match="layer0.qkv",
    weight={
        "mode": "replace",
        "prepare": [pad.columns(multiple=4, semantic=True)],
        "quantize": skip(),
    },
)
```

YAML:

```yaml
match: "layer0.qkv"
weight:
  mode: replace
  prepa
    - method: pad.columns
      multiple: 4
      semantic: true
  quantize:
    method: skip
```

`mode: replace` is advanced.
Users should not need it for normal per-layer overrides like changing only `bits`.


## Why `export` Is Separate From `quantize`

These are different questions:

1. How is the quantized state computed?
2. How is that state emitted as final tensors and metadata?

Example:

Python:

```python
weight={
    "quantize": rtn(bits=4, sym=True),
    "export": {"format": "gptq", "impl": "default"},
}
```

YAML:

```yaml
weight:
  quantize:
    method: rtn
    bits: 4
    sym: true
  export:
    format: gptq
    impl: default
```

This means:

- use RTN to produce the quantized weight state
- encode that state in GPTQ-style exported tensors

For GPTQ itself:

Python:

```python
weight={
    "quantize": gptq(bits=4, sym=True, group_size=128),
    "export": {"format": "gptq", "impl": "default"},
}
```

YAML:

```yaml
weight:
  quantize:
    method: gptq
    bits: 4
    sym: true
    group_size: 128
  export:
    format: gptq
    impl: default
```

Here, GPTQ packing is part of export realization.
It does not need to be a separate first-class user concept.

Conceptually:

```text
W
  -> quantize(method=GPTQ)
  -> logical quantized state
  -> export("gptq")
  -> qweight + scales + qzeros + g_idx + metadata
```

So `export` is the correct user-facing property, while internal packing details remain backend implementation details.
This is also why the canonical `export` form should be an object rather than a string-only enum.


## Activation Quantization

The protocol should expose activation quantization through tensor targets, not backend-internal names.

Example:

Python:

```python
Rule(
    match="*",
    input={
        "quantize": mxfp4(mode="dynamic", block_size=32, scale_bits=8),
        "export": {
            "format": "fp4",
            "variant": "mxfp4",
            "impl": "modelopt",
        },
    },
)
```

YAML:

```yaml
match: "*"
input:
  quantize:
    method: mxfp4
    mode: dynamic
    block_size: 32
    scale_bits: 8
  export:
    format: fp4
    variant: mxfp4
    impl: modelopt
```

This corresponds conceptually to installing an input activation quantizer on matched modules.

In NVIDIA Model Optimizer terms, this is cleaner than exposing `*input_quantizer` directly.

Important:

- `input` means the activation entering the matched module
- `output` means the activation leaving the matched module
- these are tensor-target concepts, not inserted-submodule names


## Activation-aware GPTQ

If `weight.quantize = gptq(...)` and `input.quantize = ...` coexist, the weight quantizer may need to know whether it should optimize using full-precision or quantized inputs.

Recommended future-proof parameter:

Python:

```python
gptq(
    bits=4,
    sym=True,
    group_size=128,
    activation_mode="ignore",   # or "fake", later possibly "real"
)
```

YAML:

```yaml
method: gptq
bits: 4
sym: true
group_size: 128
activation_mode: ignore
```

Meaning:

- `"ignore"`: classic weight-only GPTQ
- `"fake"`: optimize with fake-quantized inputs active
- `"real"`: reserved for future real low-bit activation flow


## Merge And Override Semantics

Rules compose top-to-bottom within a stage.

Recommended semantics:

- broader rules define defaults
- narrower rules refine or override
- target sections merge recursively by default
- target-local lists such as `prepare` append by default
- quantizer leaf fields such as `bits`, `sym`, and `group_size` are last-match-wins
- exporter leaf fields such as `format`, `variant`, `impl`, and `version` are last-match-wins
- `skip()` is target-scoped
- `stop=True` prevents later rules from changing the same matched object
- if `quantize.method` changes, previous quantizer-specific fields should be discarded unless explicitly repeated
- if `export.format` changes, previous format-specific export fields should be discarded unless explicitly repeated

Example: global default plus narrow skip

Python:

```python
Rule(
    match="*",
    weight={
        "prepare": [pad.columns(multiple=4, semantic=True)],
        "quantize": gptq(bits=4, sym=True, group_size=128),
        "export": {"format": "gptq", "impl": "default"},
    },
    input={
        "quantize": mxfp4(mode="dynamic", block_size=32, scale_bits=8),
        "export": {
            "format": "fp4",
            "variant": "mxfp4",
            "impl": "modelopt",
        },
    },
)

Rule(
    match="layer0.qkv",
    weight={
        "quantize": skip(),
    },
)
```

YAML:

```yaml
- match: "*"
  weight:
    prepa
      - method: pad.columns
        multiple: 4
        semantic: true
    quantize:
      method: gptq
      bits: 4
      sym: true
      group_size: 128
    export:
      format: gptq
      impl: default
  input:
    quantize:
      method: mxfp4
      mode: dynamic
      block_size: 32
      scale_bits: 8
    export:
      format: fp4
      variant: mxfp4
      impl: modelopt

- match: "layer0.qkv"
  weight:
    quantize:
      method: skip
```

Effective result for `layer0.qkv`:

- keep `weight.prepare = [pad.columns(...)]`
- skip `weight.quantize`
- keep default `input.quantize = mxfp4(...)`

Example: base config plus bits-only override

Python:

```python
Rule(
    match="*",
    weight={
        "quantize": gptq(bits=4, sym=True, group_size=128),
        "export": {"format": "gptq", "impl": "default"},
    },
)

Rule(
    match=".*(q_proj|k_proj)$",
    weight={
        "quantize": {"bits": 8},
    },
)
```

YAML:

```yaml
- match: "*"
  weight:
    quantize:
      method: gptq
      bits: 4
      sym: true
      group_size: 128
    export:
      format: gptq
      impl: default

- match: ".*(q_proj|k_proj)$"
  weight:
    quantize:
      bits: 8
```

Effective result for `q_proj` and `k_proj`:

- `method = gptq`
- `bits = 8`
- `sym = true`
- `group_size = 128`

Example: explicit replacement plus stop

Python:

```python
Rule(
    match="layer0.qkv",
    stop=True,
    weight={
        "mode": "replace",
        "prepare": [pad.columns(multiple=4, semantic=True)],
        "quantize": skip(),
    },
)
```

YAML:

```yaml
match: "layer0.qkv"
stop: true
weight:
  mode: replace
  prepa
    - method: pad.columns
      multiple: 4
      semantic: true
  quantize:
    method: skip
```

This means:

- replace inherited weight config with only the fields given here
- do not let later rules change `layer0.qkv`


## Execution Semantics

Within a stage, recommended engine order is:

1. evaluate rules in order
2. resolve matches
3. resolve optional aliases
4. run rule `actions`
5. run target `prepare`
6. collect calibration / replay data required by quantizers
7. run target `quantize`
8. run target `export`
9. emit stage outputs

Stage order then defines the full pipeline order.


## How Rule Actions And Target Config Work Together

This is the intended pattern:

Python:

```python
Rule(
    match=".*self_attn$",
    actions=[smoothquant(alpha=0.5)],
)

Rule(
    match="*",
    weight={
        "prepare": [clip.mad(k=2.75)],
        "quantize": gptq(bits=4, sym=True, group_size=128),
        "export": {"format": "gptq", "impl": "default"},
    },
    input={
        "quantize": mxfp4(mode="dynamic", block_size=32, scale_bits=8),
        "export": {
            "format": "fp4",
            "variant": "mxfp4",
            "impl": "modelopt",
        },
    },
)
```

YAML:

```yaml
- match: ".*self_attn$"
  actions:
    - method: smoothquant
      alpha: 0.5

- match: "*"
  weight:
    prepa
      - method: clip.mad
        k: 2.75
    quantize:
      method: gptq
      bits: 4
      sym: true
      group_size: 128
    export:
      format: gptq
      impl: default
  input:
    quantize:
      method: mxfp4
      mode: dynamic
      block_size: 32
      scale_bits: 8
    export:
      format: fp4
      variant: mxfp4
      impl: modelopt
```

Meaning:

- each matched attention block first receives the `smoothquant(...)` action in its own rule context
- later, the global rule supplies default weight and input quantization policy
- submodules that the action touches inside those attention blocks still receive the global defaults from the later rule

The user does not need to add action-level rematching such as `on=[...]` in the normal case.
The rule already defines the scope.


## Recommended Authoring Patterns

### 1. Global weight default

Python:

```python
Rule(
    match="*",
    weight={
        "quantize": gptq(bits=4, sym=True, group_size=128),
        "export": {"format": "gptq", "impl": "default"},
    },
)
```

YAML:

```yaml
match: "*"
weight:
  quantize:
    method: gptq
    bits: 4
    sym: true
    group_size: 128
  export:
    format: gptq
    impl: default
```

### 2. Weight and input together

Python:

```python
Rule(
    match="*",
    weight={
        "quantize": gptq(bits=4, sym=True, group_size=128),
        "export": {"format": "gptq", "impl": "default"},
    },
    input={
        "quantize": mxfp4(mode="dynamic", block_size=32, scale_bits=8),
        "export": {
            "format": "fp4",
            "variant": "mxfp4",
            "impl": "modelopt",
        },
    },
)
```

YAML:

```yaml
match: "*"
weight:
  quantize:
    method: gptq
    bits: 4
    sym: true
    group_size: 128
  export:
    format: gptq
    impl: default
input:
  quantize:
    method: mxfp4
    mode: dynamic
    block_size: 32
    scale_bits: 8
  export:
    format: fp4
    variant: mxfp4
    impl: modelopt
```

### 3. Rule-scoped balancing action

Python:

```python
Rule(
    match=".*self_attn$",
    actions=[smoothquant(alpha=0.5)],
)
```

YAML:

```yaml
match: ".*self_attn$"
actions:
  - method: smoothquant
    alpha: 0.5
```

### 4. Skip weight quantization but keep other defaults

Python:

```python
Rule(
    match="layer0.qkv",
    weight={
        "quantize": skip(),
    },
)
```

YAML:

```yaml
match: "layer0.qkv"
weight:
  quantize:
    method: skip
```

### 5. Quantize with one method, export as another format

Python:

```python
Rule(
    match=".*down_proj$",
    weight={
        "quantize": rtn(bits=4, sym=True),
        "export": {"format": "gptq", "impl": "default"},
    },
)
```

YAML:

```yaml
match: ".*down_proj$"
weight:
  quantize:
    method: rtn
    bits: 4
    sym: true
  export:
    format: gptq
    impl: default
```

### 6. Override only one quantizer field

Python:

```python
Rule(
    match="*",
    weight={
        "quantize": gptq(bits=4, sym=True, group_size=128),
        "export": {"format": "gptq", "impl": "default"},
    },
)

Rule(
    match="model.layers.0.self_attn.q_proj",
    weight={
        "quantize": {"bits": 2},
    },
)
```

YAML:

```yaml
- match: "*"
  weight:
    quantize:
      method: gptq
      bits: 4
      sym: true
      group_size: 128
    export:
      format: gptq
      impl: default

- match: "model.layers.0.self_attn.q_proj"
  weight:
    quantize:
      bits: 2
```

### 7. Override only one exporter field

Python:

```python
Rule(
    match="*",
    weight={
        "quantize": awq(bits=4, sym=True, group_size=128),
        "export": {
            "format": "awq",
            "variant": "gemm",
            "impl": "llm_awq",
            "version": 2,
        },
    },
)

Rule(
    match=".*small_proj$",
    weight={
        "export": {"variant": "gemv"},
    },
)
```

YAML:

```yaml
- match: "*"
  weight:
    quantize:
      method: awq
      bits: 4
      sym: true
      group_size: 128
    export:
      format: awq
      variant: gemm
      impl: llm_awq
      version: 2

- match: ".*small_proj$"
  weight:
    export:
      variant: gemv
```


## Real Test-Derived Examples

The examples below are translations of real repo tests into the proposed protocol.
They preserve the tested quantization intent, but they do not try to mirror every harness detail such as evaluation tasks, prompt text, or temporary save paths.


### 1. GPTQ with per-module overrides

Source tests:

- `tests/test_dynamic.py`

Current tested behavior:

- base quantization is 4-bit GPTQ
- base group size is 128
- `up_proj` and `gate_proj` are overridden to 8-bit
- `down_proj` keeps 4-bit but overrides group size to 32

Python:

```python
Stage(
    name="ptq",
    rules=[
        Rule(
            match="*",
            weight={
                "quantize": {
                    "method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                },
                "export": {
                    "format": "gptq",
                    "impl": "default",
                },
            },
        ),
        Rule(
            match=".*\\.up_proj.*",
            weight={
                "quantize": {"bits": 8},
            },
        ),
        Rule(
            match=".*\\.gate_proj.*",
            weight={
                "quantize": {"bits": 8},
            },
        ),
        Rule(
            match=".*\\.down_proj.*",
            weight={
                "quantize": {"bits": 4, "group_size": 32},
            },
        ),
    ],
)
```

YAML:

```yaml
stages:
  - name: ptq
    rules:
      - match: "*"
        weight:
          quantize:
            method: gptq
            bits: 4
            group_size: 128
          export:
            format: gptq
            impl: default
      - match: ".*\\.up_proj.*"
        weight:
          quantize:
            bits: 8
      - match: ".*\\.gate_proj.*"
        weight:
          quantize:
            bits: 8
      - match: ".*\\.down_proj.*"
        weight:
          quantize:
            bits: 4
            group_size: 32
```

This is the clearest example of why the protocol uses patch-first override semantics.
The narrower rules change only the leaf fields they care about.


### 2. AWQ GEMM full-model quantization

Source tests:

- `tests/test_awq.py`
- `tests/models/awq/test_llama3_2.py`
- `tests/models/model_test.py`

Current tested behavior:

- method is AWQ
- bits are 4
- group size is 128
- one tested export target is AWQ GEMM
- one tested runtime backend is `BACKEND.TORCH_AWQ`

Python:

```python
Stage(
    name="ptq",
    rules=[
        Rule(
            match="*",
            weight={
                "quantize": {
                    "method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                },
                "export": {
                    "format": "awq",
                    "variant": "gemm",
                },
            },
        ),
    ],
)
```

YAML:

```yaml
stages:
  - name: ptq
    rules:
      - match: "*"
        weight:
          quantize:
            method: awq
            bits: 4
            group_size: 128
            sym: true
          export:
            format: awq
            variant: gemm
```

In `tests/test_awq.py`, the same pattern is also exercised with other export variants such as:

- `gemv`
- `gemv_fast`
- `llm_awq`

That is exactly why `export` needs to be an object and not just a single string token.


### 3. RTN with weight smoothing and AWQ GEMM export

Source tests:

- `tests/test_weight_only_config.py`

Note:

- the repo currently exercises RTN primarily through weight-only/config and format-conversion tests rather than a `tests/models/*` full-model test case

Current tested behavior:

- method is RTN
- bits are 4
- group size is 128
- smoothing uses `SmoothMAD(k=1.5)`
- export target is AWQ GEMM

Python:

```python
Stage(
    name="weight_only",
    rules=[
        Rule(
            match="*",
            weight={
                "prepare": [
                    {"method": "smooth.mad", "k": 1.5},
                ],
                "quantize": {
                    "method": "rtn",
                    "bits": 4,
                    "group_size": 128,
                },
                "export": {
                    "format": "awq",
                    "variant": "gemm",
                },
            },
        ),
    ],
)
```

YAML:

```yaml
stages:
  - name: weight_only
    rules:
      - match: "*"
        weight:
          prepa
            - method: smooth.mad
              k: 1.5
          quantize:
            method: rtn
            bits: 4
            group_size: 128
          export:
            format: awq
            variant: gemm
```

Related repo test:

- `tests/test_format_conversion_flow.py` also verifies that RTN can export to GPTQ format with `RTNConfig(bits=4, format=FORMAT.GPTQ, offload_to_disk=False)`

That is a concrete existing example of the protocol's `quantize != export` split.


## Migration From Current `gptqmodel`

Current `gptqmodel` configuration is primarily weight-centric.

A straightforward migration path is:

- base `QuantizeConfig` -> broad default rule
- `dynamic` positive override -> narrower rule
- partial `dynamic` override fields -> quantizer patch fields such as `weight.quantize.bits`
- `-:` negative skip -> target-scoped `skip()`
- smoothers / preprocessors -> `weight.prepare`
- method vs output representation split -> `quantize` vs `export`

This keeps current intent while making the protocol ready for activation and cache quantization.


## Non-Goals

The protocol should not make these primary user concepts:

- inserted quantizer submodule names
- internal packer tensor names
- stage-level rematching
- action-level global rematching in normal usage


## Final Shape

The recommended protocol shape is:

Python:

```python
version = 2

stages = [
    Stage(
        name="ptq",
        rules=[
            Rule(
                match=".*self_attn$",
                actions=[smoothquant(alpha=0.5)],
            ),
            Rule(
                match="*",
                weight={
                    "prepare": [clip.mad(k=2.75)],
                    "quantize": gptq(
                        bits=4,
                        sym=True,
                        group_size=128,
                        activation_mode="fake",
                    ),
                    "export": {"format": "gptq", "impl": "default"},
                },
                input={
                    "quantize": mxfp4(
                        mode="dynamic",
                        block_size=32,
                        scale_bits=8,
                    ),
                    "export": {
                        "format": "fp4",
                        "variant": "mxfp4",
                        "impl": "modelopt",
                    },
                },
            ),
            Rule(
                match="layer0.qkv",
                weight={
                    "quantize": skip(),
                },
            ),
        ],
    ),
]
```

YAML:

```yaml
version: 2
stages:
  - name: ptq
    rules:
      - match: ".*self_attn$"
        actions:
          - method: smoothquant
            alpha: 0.5
      - match: "*"
        weight:
          prepa
            - method: clip.mad
              k: 2.75
          quantize:
            method: gptq
            bits: 4
            sym: true
            group_size: 128
            activation_mode: fake
          export:
            format: gptq
            impl: default
        input:
          quantize:
            method: mxfp4
            mode: dynamic
            block_size: 32
            scale_bits: 8
          export:
            format: fp4
            variant: mxfp4
            impl: modelopt
      - match: "layer0.qkv"
        weight:
          quantize:
            method: skip
```

This keeps the model concise:

- one matcher
- optional aliases
- rule-scoped actions
- first-class tensor targets
- local `prepare`
- explicit `quantize`
- explicit `export`
- readable override and stop semantics
