# Model tree resolution

How GPT-QModel decides what each module *is*: its role (`q`/`k`/`v`/`o`,
`gate`/`up`/`down`), whether it belongs to a MoE block, which expert family it
belongs to, and whether it should be quantized at all.

The short version: **the architecture's declared `module_tree` is authoritative;
runtime module names are not consulted.** Introduced in PR #3565.

For a practical tutorial, the complete tag reference, terminal/JSON inspection,
and new-model requirements, see the [module-plan guide](module-plan-guide.md).
The [DeepSeek V4.1 Flash inventory](deepseek-v41-module-inventory.md) distinguishes
verified checkpoint structure from remaining integration work.

The [post-quantization fusion planner](fusion-planner.md) consumes this same
compiled plan to select compatible Llama QKV and gate/up groups. Fusion is
disabled by default, supports model-specific overrides, and preserves checkpoint
owners as views into a single fused packed allocation.

## Why it is shaped this way

Before this change the answer to "what is this module?" was derived from the
*string* of the live module name:

```mermaid
flowchart LR
    subgraph OLD["BEFORE — runtime name guessing"]
        direction TB
        N["module full name<br/>model.layers.7.mlp.experts.3.w1"] --> H1{"'.expert' in name?"}
        N --> H2{"'.experts.' in name?"}
        N --> H3{"endswith('mlp.down_proj')?"}
        N --> H4{"'.attn' in name?"}
        N --> H5{"endswith(declared)?"}
        H1 --> R1["is MoE"]
        H2 --> R2["group key"]
        H3 --> R3["rotation target"]
        H4 --> R4["attention"]
        H5 --> R5["quantizable"]
    end
    style OLD fill:#fff1f0,stroke:#d4380d
    style H1 fill:#ffe7ba
    style H2 fill:#ffe7ba
    style H3 fill:#ffe7ba
    style H4 fill:#ffe7ba
    style H5 fill:#ffe7ba
```

Each of those rules fails *silently*: no exception, just the wrong grouping or
the wrong quantization. A model that renames `gate_proj` to `w1`, exposes
`shared_mlp` instead of `shared_experts`, or adds an extra attention projection
was mis-classified by construction.

The repository already declared the structure of every supported architecture in
`module_tree` — decoder boundaries (`#`), leaf order, aliases, `:moe` markers,
and `!`/`?` exclusion flags. That declaration only drove layer extraction; role
logic re-guessed the rest. This design makes the declaration the single source of
truth and compiles it once per model instance.

## Resolution flow

```mermaid
flowchart TD
    M["model loaded<br/>AutoModel / checkpoint"] --> RES["_resolve_effective_module_tree()"]
    RES -->|1| INJ["model.gptqmodel_effective_module_tree<br/>explicit injection"]
    RES -->|2| CLS["cls.module_tree<br/>registered definition"]
    INJ --> OVR
    CLS --> OVR["apply module_tree_overrides()<br/>per-method: gptq / awq / exl3"]
    OVR -->|None| ERR["ValueError:<br/>no explicit module_tree registered<br/>fail closed"]
    OVR -->|tree| FRZ["_freeze_module_tree()<br/>deep immutable"]
    FRZ --> ATTR["self.effective_module_tree = frozen<br/>self.module_tree = effective"]
    ATTR --> PLAN["ModulePlan.from_tree(effective)<br/>compiled once"]
    PLAN --> USE["model.module_plan()<br/>public and internal accessor"]
    USE --> HOOK["moe_lifecycle_hooks.module_plan<br/>same plan on per-instance hook copy"]
    PLAN --> MDL["model.gptqmodel_effective_module_tree<br/>handed to loader"]
    style PLAN fill:#d9f7be,stroke:#237804,stroke-width:2px
    style ERR fill:#ffa39e,stroke:#a8071a
    style FRZ fill:#e6f7ff
```

Three properties matter here:

- **One tree per instance.** The tree is deep-frozen and stored on the instance,
  so a class definition can be reused by later instances without dispatch state
  leaking between them.
- **Compiled once.** `ModulePlan` is built at construction, not re-derived per
  module lookup.
- **Fails closed.** If no definition and no injection exist, loading raises
  instead of guessing a structure.

## `ModulePlan`

`gptqmodel/models/module_plan.py` turns the declaration into an immutable
inventory of `ModuleSpec(template, flags)` plus role and MoE queries.

```mermaid
flowchart LR
    T["module_tree<br/>['model','layers','#',...]"] --> B["find decoder boundary<br/>variant.index('#')"]
    B --> W["walk()<br/>dict → keys / list → leaves"]
    W --> S["ModuleSpec<br/>template + flags"]
    S --> IDX["specs map<br/>template → spec"]
    S --> PAR["_by_parent map<br/>parent → specs"]
    IDX --> Q["queries"]
    PAR --> Q
    Q --> Q1["roles / child_roles"]
    Q --> Q2["is_moe / moe_roots"]
    Q --> Q3["expert_containers"]
    Q --> Q4["shared_expert_containers"]
    Q --> Q5["expert_root / same_moe_family"]
    Q --> Q6["is_expert / concrete_paths"]
    style S fill:#d9f7be,stroke:#237804
```

### Inspection API

The compiled plan is attached to every model instance and is retrieved through
one public method:

```python
plan = model.module_plan()

plan.print()                     # TUI tree, ANSI auto-detected
plan.print(color=False)          # plain log-safe output
plan.print(max_depth=2)          # cap very large trees
text = plan.render(color=False)
payload = plan.to_dict()         # JSON-friendly extraction/debug data
json_text = plan.to_json()
summary = plan.summary()
```

`ModulePlan` and `ModuleSpec` are exported from both `gptqmodel` and
`gptqmodel.models`. `model.module_plan()` raises if called before the model has a
compiled plan; it never guesses or recompiles a tree. All internal consumers
resolve the plan through this accessor rather than reading the backing field.

The tree renderer marks:

- `[quantize]` — declared as a quantization leaf.
- `[excluded]` / `[capture-only]` — non-quantized declarations.
- `[moe]`, `[expert-container]`, `[shared-expert]`, `[expert]` — MoE structure.
- `[roles=q]`, `[in=x]`, `[group=0]` — projection roles, shared input, and group.
- `[container]` — structural parent, not itself a quantization target.

### Declaration syntax

```
"mlp:moe"                 ^    ^
  │   │                   │    └── flags: moe root
  │   └── leaf/container name
  └── the declared path segment
```

| Token form | Meaning | Produced role |
| --- | --- | --- |
| `q_proj:0:q` | slot `0`, role `q` | `q` |
| `w1:0:gate` | explicit tag wins over the name | `gate` |
| `mlp:moe` | MoE root marker | `spec.is_moe`, not a projection role |
| `x:!` / `x:?` | not quantized / capture-only | `quantize = False` |
| `"experts": {"#": (...)}` | routed-expert expansion | `{expert_index}` → `\d+` |
| `gate_up_proj` | fused projection | `("gate","up")` |

### Matching is exact

Templates compile to an anchored regular expression (`^…$`) with `#` replaced by
`\d+`. There is no `startswith`, `endswith`, or `in` fallback:

```mermaid
flowchart LR
    A["lookup: mlp.experts.3.w1"] --> E{"exact template?"}
    E -->|yes| H["spec"]
    E -->|no| R{"anchored regex<br/>^mlp\.experts\.\d+\.w3$"}
    R -->|match| H
    R -->|no| NONE["None →<br/>caller treats as undeclared"]
    style NONE fill:#fff1f0,stroke:#a8071a
```

Because the pattern is anchored and placeholder-aware, `mlp_extra.gate_proj`
can never match the declared `mlp` branch, while `mlp.experts.3.gate_proj`
matches the declared `mlp.experts.#.gate_proj` template.

### Shared experts without an allow-list

`shared_expert_containers()` finds shared/parallel expert branches
**structurally**: a direct projection family under a MoE root that is not the
routed `#` container and carries no router-only role. Definitions therefore do
not need to spell `shared_experts`, `shared_expert`, or `shared_mlp`; declaring
the branch is enough.

An explicitly numbered expert also belongs to the routed container. For example,
Hymba declares `moe.experts.0` instead of a `#` placeholder. Its projection
children make `moe.experts` an expert container, and the numbered branch is not
classified as a shared expert. Expert-family queries still require the full
path to be declared in the tree.

```mermaid
flowchart TD
    ROOT["declared MoE root"] --> CHILD{"declared child structure"}
    CHILD -->|"# placeholder"| ROUTED["expert_containers()"]
    CHILD -->|"numeric child with projection leaves"| ROUTED
    CHILD -->|"other projection family"| SHARED["shared_expert_containers()"]
    ROUTED --> FIND{"ModulePlan.find(path)"}
    FIND -->|"declared"| GROUP["is_expert() / expert_root()"]
    FIND -->|"undeclared"| REJECT["False / None"]
    style REJECT fill:#ffa39e,stroke:#a8071a
```

## What each subsystem asks

Rotation reads normalization membership from explicit `norm` flags; projection
roles remain `q/k/v/o/gate/up/down`. A supplied plan also makes online Hadamard
setup reject paths outside the declared decoder layers. The public helper's
legacy default applies only when no plan is supplied.

```mermaid
flowchart TD
    PLAN["ModulePlan"] --> BASE["models/base.py<br/>is_moe_module() · AWQ roles<br/>stage-block classification"]
    PLAN --> MOE["models/moe_lifecycle.py<br/>expert / shared containers<br/>dotted MoE roots"]
    PLAN --> IOL["models/loader.py · writer.py<br/>declared-path quantizability<br/>rotation down-proj"]
    PLAN --> ROT["rotation/rotation.py<br/>:norm :q :k :v :o :gate :up :down"]
    PLAN --> LOOP["looper/module_looper.py<br/>expert_root() group keys<br/>role-based attention"]
    PLAN --> SUB["looper/stage_subset.py<br/>weight_only_looper.py<br/>gate+up in roles<br/>co-location"]
    PLAN --> GSQ["quantization/gsq_training*.py<br/>declared projection inventory<br/>attention/MLP partition"]
    style PLAN fill:#d9f7be,stroke:#237804,stroke-width:2px
```

### MoE membership and grouping

```mermaid
flowchart LR
    P["model.layers.9.mlp.experts.5.w2"] --> REL["strip layer prefix → layer-relative<br/>mlp.experts.5.w2"]
    REL --> PL["plan.roles() → ('down',)"]
    REL --> EX["plan.expert_root() →<br/>mlp.experts.5"]
    EX --> GRP["device co-location group"]
    REL --> MOEQ["plan.is_moe() → walks moe_roots()"]
    style EX fill:#d9f7be
```

The expert family key used for device co-location is the declared family path, so
`gate`/`up`/`down` of one expert stay together on one device regardless of what
the projections are named.

## Registry-wide module-plan unit tests

`tests/module_tree/test_registered_module_plans.py` runs for every `MODEL_MAP`
entry without an opt-in flag, including aliases and each method override. The
committed `fixtures/registered_module_plans.json` records reviewed decoder
prefixes and ordered quantization/capture/exclusion blocks. A new registered
model without an explicit case fails the suite.

The tested registry contains 198 model-type entries, 173 definition classes,
and 175 definition/method cases. Original `dbrx` is a documented unsupported
sentinel and receives a rejection test; `dbrx_converted` receives a positive
contract test. This suite validates definitions and compiler/consumer agreement.
Real model layout and checkpoint weight ownership are verified separately by
model-specific tests such as `test_flash_model_tree_consumption.py`.

```mermaid
flowchart TD
    REG["MODEL_MAP entry"] --> CASE{"explicit inventory case?"}
    CASE -->|"missing"| FAIL["test failure: add model-plan case"]
    CASE -->|"present"| TREE["definition and method override"]
    TREE --> PLAN["ModulePlan.from_tree()"]
    TREE --> SUB["build_layer_modules() / shared-input metadata"]
    PLAN --> CHECK["compare paths, flags, norms, MoE and exact lookup"]
    SUB --> CHECK
    CHECK --> USE["ModuleLooper / WeightOnlyLooper via module_plan()"]
    TREE --> REAL["separate real-model tests: actual layout and checkpoint owners"]
    style FAIL fill:#ffa39e,stroke:#a8071a
```

New support must add both an explicit registry case and a real-model module-plan
unit test. This requirement is maintained in `AGENTS.md`,
`.agents/skills/gptqmodel-model-plan-tests/SKILL.md`, and the PR template.

### Quantizability (loader and writer)

```mermaid
flowchart TD
    N["named module<br/>model.layers.3.mlp.up_proj"] --> LP{"declared layer prefix?"}
    LP -->|no| SKIP["skip: not quantized"]
    LP -->|yes| REL["relative = mlp.up_proj"]
    REL --> DEC{"relative declared, or<br/>expanded expert leaf?"}
    DEC -->|yes| Q["quantize"]
    DEC -->|no| SKIP
    style Q fill:#d9f7be,stroke:#237804
    style SKIP fill:#ffa39e
```

### Checkpoint weight coverage

The tree is also the oracle for every checkpoint-owned weight inside a decoder
layer, not only for modules that the current quantization pipeline replaces.
This prevents a non-quantized projection, router, hyper-connection, compressor,
or norm from disappearing between the model definition and the live model.

```mermaid
flowchart TD
    SD["Checkpoint keys + conversion provenance<br/>state_dict for architecture fixtures"] --> OWN["deepest owning module<br/>for each entry"]
    OWN --> FIND["plan.find(owner)"]
    FIND -->|None| FAIL["test failure:<br/>undeclared weight owner"]
    FIND -->|spec| Q{"owner in<br/>simple_layer_modules?"}
    Q -->|yes| DECL["quantized by declaration"]
    Q -->|no| TAG{"spec.quantize == False?"}
    TAG -->|yes| EXCLUDED["explicit :! / :? exclusion"]
    TAG -->|no| FAIL2["test failure:<br/>untagged non-quantized weight"]
    style FAIL fill:#ffa39e,stroke:#a8071a
    style FAIL2 fill:#ffa39e,stroke:#a8071a
    style EXCLUDED fill:#d9f7be,stroke:#237804
```

`state_dict()` supplies parameters and persistent buffers for architecture
fixtures. Actual checkpoint ownership must be established from saved keys and
explicit conversion mappings. Post-load tensors can be persistent too; appearing
in `state_dict()` alone does not make them model-definition obligations.
The owning-module rule also covers direct checkpoint parameters and buffers on
containers, such as Qwen `linear_attn.A_log`, DeepSeek `self_attn.sinks`, and
Ripple `self_attn.value_res_lambda`.

The current plan compiles decoder-relative module owners, not a whole-model
per-tensor manifest. Base embeddings, vision towers, heads, and auxiliary
prediction modules outside declared decoder regions need separate coverage.
Its summary counts declaration flags, including structural parents; those counts
are not the number of live quantized tensors.

Measured Flash-family architecture-fixture coverage at this revision (these are
real model shells, not full checkpoint provenance audits):

| Model family | Decoder layers | Quantizable specs | Unique weight owners |
| --- | ---: | ---: | ---: |
| Qwen3.5 / Qwen3.8 | 32 | 10 | 19 |
| GLM-5.3-Flash | 45 | 873 | 37 |
| MiniMax-M3.1-Flash | 60 | 394 | 20 |
| DeepSeek-V4-Flash | 43 | 775 | 29 |
| Ripple | 5 (fixture) | 8 | 15 |

DeepSeek-V4.1-Flash is not included: its checkpoint declares
`model_type=deepseek_v41`, but the verified Transformers 5.22.1 environment and
GPT-QModel registry have no corresponding entry. Its checkpoint now provides
reference inference code; that is not a registered Transformers/GPT-QModel
loader. The test activates when upstream config support lands instead of
relabeling V4-Flash as V4.1. See the pinned
[V4.1 checkpoint inventory](deepseek-v41-module-inventory.md) for the current
evidence and remaining work.

### Rotation

Rotation needs the two layer norms, one attention parent, and one MLP parent.
It reads them from declared `:norm` and projection roles rather than assuming a
Llama layout:

```mermaid
flowchart TD
    API["model.module_plan()"] --> FUSE["fuse_layer_norms(plan=...)<br/>then rotate_model(plan=...)"]
    FUSE --> N1["Fusion: root specs with norm flag<br/>requires exactly 2 layer norms"]
    FUSE --> P1["_rotation_parents(plan)<br/>attention parent + MLP parent"]
    P1 --> Q1["_resolve_role()<br/>exactly one projection per requested role"]
    Q1 -->|"missing or multiple"| ERR["ValueError: incompatible declaration or layout"]
    N1 --> F["Fuse layer and final norms; reset norms"]
    Q1 -->|"resolved"| F
    F --> EMB["rotate_model: embeddings and head first<br/>then declared decoder projections"]
    PUBLIC["Public helper called without a plan"] --> DEF["_default_rotation_plan()<br/>Llama defaults"]
    DEF --> FUSE
    style DEF fill:#e6f7ff
    style P1 fill:#d9f7be
    style ERR fill:#ffa39e,stroke:#a8071a
```

`_resolve_role` requires exactly one match and raises with the offending path, so
a definition with two `:q` projections fails loudly instead of silently picking
one.

## Ambiguity in name lookup

Two helpers in `utils/model.py` used to return the first match in module
registration order. They now prefer an exact path and raise on multiple matches:

```mermaid
flowchart LR
    subgraph A["get_module_by_name_prefix"]
        A1["exact path in named_modules?"] -->|yes| AOK["return"]
        A1 -->|no| A2["dotted descendants"]
        A2 -->|1| AOK
        A2 -->|>1| AERR["ValueError:<br/>Ambiguous module prefix"]
        A2 -->|0| ANONE["None"]
    end
    subgraph B["get_module_by_name_suffix"]
        B1["exact path?"] -->|yes| BOK["return"]
        B1 -->|no| B2["ends with '.' + suffix"]
        B2 -->|1| BOK
        B2 -->|>1| BERR["ValueError:<br/>Ambiguous module suffix"]
        B2 -->|0| BNONE["None"]
    end
    style AERR fill:#ffa39e
    style BERR fill:#ffa39e
```

## Contracts

```mermaid
flowchart TD
    S["resolve effective tree"] --> I{"model.gptqmodel_effective_module_tree?"}
    I -->|yes| O["apply module_tree_overrides()"]
    I -->|no| C{"cls.module_tree?"}
    C -->|yes| O
    C -->|no| X["ValueError:<br/>register a model definition"]
    O --> F["freeze deep-immutable"]
    F --> P["ModulePlan.from_tree()"]
    P --> R["load / quantize / rotate"]
    style X fill:#ffa39e,stroke:#a8071a
    style P fill:#d9f7be,stroke:#237804
```

| Situation | Behavior |
| --- | --- |
| Unknown `model_type`, no definition | `ValueError`; never auto-guesses a tree |
| Multiple projections for a role in a single-role consumer | That consumer raises `ValueError`; the plan itself can represent multiple projections |
| Ambiguous prefix or suffix lookup | `ValueError` naming the candidates |
| Path not declared in the tree | treated as undeclared: not quantized, no role |
| Custom architecture | inject `model.gptqmodel_effective_module_tree` |
| Per-method difference | `module_tree_overrides[method]` on the definition |

## Extension points

- **New architecture:** add a model definition with an explicit `module_tree`.
  Never add a new name heuristic to make an architecture work.
- **Per-method divergence:** use `module_tree_overrides` keyed by quantization
  method instead of branching on names inside shared code.
- **Custom/injected models:** attach `gptqmodel_effective_module_tree` to the
  model object before loading.
- **Diagnostics only:** `_auto_detect_module_tree()` still exists to inspect an
  unknown model, but it is deliberately not on the loading path.

## Known limitation

`ModuleSpec.roles` falls back to a small name-to-role map (`q_proj`,
`k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `w1`, `w2`,
`w3`, `gate_up_proj`) when a declared leaf carries no explicit role tag. The
fallback applies at *declaration time*, not against live module names, so it
cannot select the wrong sibling — but it is still a naming convention. Fully
tagging every definition would allow the map to be deleted.

## Related tests

| Test file | What it pins |
| --- | --- |
| `tests/module_tree/test_explicit_module_plan.py` | roles, shared containers, renamed leaves, sibling rejection, fail-closed |
| `tests/module_tree/test_module_tree_variants.py` | tree variants produce identical declared paths |
| `tests/module_tree/test_auto_detect.py` | auto-detection stays diagnostic-only |
| `tests/test_afmoe_support.py`, `tests/test_k2_horizon_support.py` | real MoE definitions resolve without name overrides |
| `tests/module_tree/test_flash_model_tree_consumption.py` | real Flash-family meta models: quantizable specs, every checkpoint weight owner, roles, MoE hooks, looper helpers, Ripple custom code |
