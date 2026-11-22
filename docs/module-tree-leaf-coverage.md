# Module-tree leaf coverage

`module_tree` is the only declaration of which decoder weights GPT-QModel will
quantize. Before issue #4174 the tests compared a definition with itself: they
proved every declared spec resolved, but a `Linear`/`Conv`/expert leaf that the
tree never mentioned was invisible. Quantization completed successfully and
left those weights in the checkpoint dtype.

This page describes the fail-closed leaf-coverage contract added in
`tests/module_tree/test_registry_leaf_coverage.py` and the per-model
`after_defuser_conversion` hook used for remote-code Phi-3 checkpoints.

## Old flow: declared specs only

```mermaid
flowchart TD
    A["MODEL_MAP entry"] --> B["definition.simple_layer_modules"]
    B --> C["declared spec list"]
    C --> D["create_named_modules"]
    D -->|"spec matches a module"| E["quantize"]
    D -->|"spec misses"| F["coverage warning or strict error"]
    G["layer Linear/Conv/expert leaf"] --> H{"declared?"}
    H -->|"no"| I["silently stays in source dtype"]
    H -->|"yes"| C
    style I fill:#f8d7da,stroke:#b02a37
```

The undeclared leaf (`I`) never enters the declared-spec list, so neither the
strict check nor the old coverage report could see it. The runtime warning in
`LayerModuleCoverageTracker.record_undeclared` also only inspected
`nn.Linear`, `nn.Conv1d` and `nn.Conv2d`, so a 3-D expert parameter was doubly
invisible.

## New flow: registry leaf inventory

```mermaid
flowchart TD
    A["MODEL_MAP model_type"] --> B["AutoConfig.for_model"]
    B --> C{"local config?"}
    C -->|"no"| S["pytest.skip: needs network"]
    C -->|"yes"| D["definition.loader.from_config on meta"]
    D --> E["_convert_model_with_defuser"]
    E --> F["get_layers_with_prefixes"]
    F --> G["ModulePlan.from_tree"]
    G --> H["_leaf_paths: Linear, Conv, Embedding, expert tensors"]
    H --> I["_declared: exact spec or tagged ancestor"]
    I -->|"all covered"| J["pass"]
    I -->|"leaf uncovered"| K["pytest.fail with leaf + layer count"]
    style S fill:#fff3cd,stroke:#997404
    style K fill:#f8d7da,stroke:#b02a37
```

`_leaf_paths` adds a module when it is one of `SUPPORTS_MODULE_TYPES` or when
it directly owns a routed/shared expert parameter (2-D or 3-D). `_declared`
accepts an exact `ModulePlan.find()` hit or a `:!`/`:?` tag on an ancestor, so a
definition can explicitly exclude a whole checkpoint-owned subtree.

## Phi-3 remote-code conversion

Phi-3 128k longrope checkpoints load remote `modeling_phi3.py`. Its `Phi3MLP`
class path is not the installed Transformers class path that Defuser registers,
so Defuser leaves `mlp.gate_up_proj` fused while the definition declares
`gate_proj`/`up_proj` and `layer_modules_strict` stays true.

```mermaid
flowchart LR
    A["remote Phi3ForCausalLM"] --> B["_convert_model_with_defuser"]
    B --> C{"Defuser split Phi3MLP?"}
    C -->|"native class"| D["gate_proj / up_proj"]
    C -->|"remote class"| E["Phi3QModel.after_defuser_conversion"]
    E --> F["chunk gate_up_proj on dim 0"]
    F --> D
    D --> G["ModulePlan resolves gate/up/down"]
    style E fill:#fff3cd,stroke:#997404
```

## GLM-4V fused gate/up conversion

Transformers 5.22 builds `Glm4vTextMLP` with one fused `gate_up_proj` linear
whose output is chunked into gate and up, while `Glm4vGPTQ` declares
`mlp.gate_proj`/`mlp.up_proj` (`gptqmodel/models/definitions/glm4v.py:16-25` on
`461b42023`). `defuser.replace_fused_blocks()` can replace the class before
construction, but the module-plan registry sweep and any direct
`loader.from_config()` call never run that patch. `_convert_model_with_defuser`
still calls `after_defuser_conversion` after the conversion attempt
(`gptqmodel/models/loader.py:1011-1016`), so GLM-4V defuses there.

```mermaid
flowchart TD
    A["Glm4vGPTQ.loader.from_config"] --> B{"Defuser class patch<br/>applied before construction?"}
    B -- "yes" --> C["LinearGlm4vTextMLP<br/>gate_proj / up_proj / down_proj"]
    B -- "no" --> D["Glm4vTextMLP<br/>gate_up_proj + down_proj"]
    D --> E["_convert_model_with_defuser"]
    E --> F["Glm4vGPTQ.after_defuser_conversion"]
    F --> G["chunk gate_up_proj on dim 0"]
    G --> H["bind _defused_glm4v_mlp_forward"]
    H --> C
    C --> I["ModulePlan resolves gate/up/down"]
    D -. "before #4481: no hook,<br/>plan misses mlp.gate_proj" .-> X["strict ValueError"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X,D failure;
```

| Mechanism | Location | Effect |
| --- | --- | --- |
| No GLM-4V conversion hook | `glm4v.py:16-25` on `461b42023` | Fused `gate_up_proj` survives conversion |
| Declared gate/up leaves | `glm4v.py:16-25` on `461b42023` | Strict plan cannot resolve the fused weight |
| `after_defuser_conversion` call | `gptqmodel/models/loader.py:1011-1016` | Hook runs even when Defuser reports no change |
| New split + forward rebind | `gptqmodel/models/definitions/glm4v.py:39-98` | gate/up/down owned by the plan; module stays runnable |

| Situation | Before | After |
| --- | --- | --- |
| Defuser class patch applied | split at construction | unchanged; hook finds no `gate_up_proj` |
| Direct `loader.from_config()` | fused weight, strict `mlp.gate_proj` miss | split weight, strict resolution succeeds |
| Defuser reports no structural change | hook absent | hook splits anyway |
| Forward after split | class forward still referenced `gate_up_proj` | instance forward uses gate/up/down |
| Second conversion pass | not applicable | idempotent, no `gate_up_proj` to split |

## Failure and ambiguity contract

| Situation | Before | After |
| --- | --- | --- |
| Decoder layer path drifted (Voxtral) | zero layers, nothing quantized | `pytest.fail`: no decoder layers resolved |
| Linear/Conv leaf absent from tree | silent source-dtype weight | `pytest.fail` with path and layer count |
| 3-D expert parameter absent from tree | silent source-dtype weight | `pytest.fail`; 3-D owners are in the inventory |
| Leaf under an explicit `:!`/`:?` subtree | undeclared | covered by the ancestor tag |
| Config only available via remote code | not built | `pytest.skip` with the exception reason |
| Reviewed out-of-scope defect | unknown | explicit `KNOWN_UNDECLARED` xfail with reason |

## Coverage

`tests/module_tree/test_issue_4174_model_trees.py` pins the real layouts for
Voxtral, Nemotron-H MoE, Ling `bailing_moe`, Llama-4 Maverick, Gemma-4 MoE,
Phi-3 longrope, Zamba/Zamba2, PhiMoE, Falcon-H1, HRM-Text, Jamba, GRIN-MoE and
native Kimi-Linear. The registry test runs every `MODEL_MAP` entry whose config
is locally constructible.

`tests/module_tree/test_granitemoehybrid_model_tree.py` covers Granite MoE
Hybrid, which the registry sweep previously exempted. Its `state_dict()` owners
include a Mamba `conv1d`, a gated `norm`, parameters owned directly by the
`mamba` container, and the MoE router; the test requires an explicit
quantize/exclude policy for each. The exemption is gone, so the registry sweep
builds the model on every run.

Issue #4174 introduced this contract.
