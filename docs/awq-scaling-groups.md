# AWQ scaling groups

AWQ folds one smoothing scale per input channel into the module that produces a
quantized block's input. That is only output-preserving when the producer's
output really *is* the consumer's input and is not consumed by anything else.
The model tree encodes module order, not the residual graph, so tree order alone
cannot decide this.

This page documents the contracts consumed by
`BaseQModel.awq_get_modules_for_scaling` and
`gptqmodel.quantization.awq.quantize.scale.apply_scale`:

1. A layer with no usable scaling group is **quantized plainly**, never
   finalized unquantized (which previously failed later in packing with
   `KeyError: 'q_zeros'`).
2. A definition marks leaves whose output is not a direct scaling source with
   the `nosrc` tree flag. Groups whose `prev_op` carries it are dropped, so the
   consumer falls back to plain quantization instead of silently changing the
   decoder layer output.
3. **One gain norm has exactly one scaling group.** Routed and shared experts
   (and parallel attention/MLP branches) read the same normed tensor; folding
   them as two groups divides the norm twice and leaves one branch unscaled.
4. **Parallel branches share a source.** Sibling projections that read the same
   norm declare it with the same `:in=<tag>` flag, so the second branch folds
   into that norm rather than into the previous branch's output projection.
5. **A norm-like module that cannot absorb a fold is skipped.** A weightless
   norm, a norm with an input-dependent gate, a `Conv1D`, or a
   `HyperConnection` has no plain gain to divide; the group is dropped and the
   layer stays quantized plainly.
6. **An excluded projection is never a fold source.** A `:!`/`:?` Linear has
   no compensation step: dividing its output rows would change a tensor the
   checkpoint keeps in the source dtype. The group is dropped and the consumer
   is quantized plainly.

Introduced by the fixes for issues #4005, #4017, #4039, #4093 (items 1-6) and
#4121 (items 6-10). Rule 6 was added for #4651.

## Flow

```mermaid
flowchart TD
    A["awq_get_modules_for_scaling<br/>iterate full_layer_modules"] --> B{"non-quantized block?"}
    B -- "yes" --> C["resolve block[-1]<br/>track last_gain_norm<br/>and last_norm_candidate"]
    B -- "no" --> G["resolve feature_name and inp"]
    G --> H{"inp captured?"}
    H -- "no" --> I["skip block"]
    H -- "yes" --> J["_select_prev_op(input width,<br/>shared_source, allow_block_norm)"]
    J --> K{"prev_op found?"}
    K -- "no" --> I
    K -- "yes" --> L["generate_node_for_awq_scaling"]
    L --> M["update tracker to block[-1]"]
    M --> N["_merge_shared_source_nodes<br/>one group per gain norm"]
    N --> O["_filter_awq_scaling_sources"]
    O --> P{"prev_op nosrc,<br/>non-quantized Linear,<br/>or width mismatch?"}
    P -- "yes" --> Q["drop group<br/>consumer uses plain quantization"]
    P -- "no" --> R["keep group"]
    Q --> S["AWQProcessor._quantize_layer"]
    R --> S
    S --> T{"apply_scale can fold prev_op?"}
    T -- "yes" --> U["divide norm, scale consumers<br/>and declared scale consumers"]
    T -- "no" --> V["skip group<br/>layer stays quantized"]
    style Q fill:#fff3cd,stroke:#997404,stroke-width:2px
    style V fill:#fff3cd,stroke:#997404,stroke-width:2px
```

The tracker update runs for every block, including blocks skipped because
`prev_op is None`. That keeps the "previous linear in tree order" fallback
available for the next block instead of leaving `last_module` unset.

`_filter_awq_scaling_sources` therefore rejects a 2-D source that the plan
declares `:!`/`:?`. Tree order alone cannot tell a sequential producer from a
sibling declared inside a consumer's block: Jamba declares
`mamba: ("in_proj:0", "x_proj:1", "out_proj:2", "conv1d:!", "dt_proj:!")`, so
`block[-1]` was the excluded `mamba.dt_proj` and the next block (`mamba.x_proj`)
folded its scale there. `x_proj` reads `silu(conv1d(in_proj(x)))`; `dt_proj`
reads `x_proj`'s own output, so the width check passed while the data flow did
not. `mamba.x_proj` is now quantized with no fold, and `dt_proj` keeps its
checkpoint weights. A gain norm (1-D `weight`, divided by `scale_ln_fcs`) and a
quantized Linear remain valid sources.

## Per-module fallback policy

`AWQConfig.fallback` is the run-wide policy; a `dynamic` rule may replace it
for one projection, for example
`{"fallback": {"strategy": "stdclip", "threshold": 4096}}`. The merged
`Fallback` is resolved per module by `AWQProcessor._fallback_for_module` and
drives every selection gate, not only the strategy on a fallback path that was
already chosen (#5820):

- `_starved_group_members` compares each member's captured coverage with that
  member's own threshold.
- `_group_coverage_is_sufficient` accepts the group total only when every
  member's policy accepts it.
- `_quantize_layer` uses the group's merged policy for the fallback branch, the
  missing-feature guard and the replay-error fallback.
- `_fallback_strategy_for_module` still resolves the strategy on a chosen
  fallback path, including `apply_quant(..., scales_list=[])` (#5726).

A member with no policy at all (`fallback=None` and no override) still aborts a
layer with missing features; a dynamic override that enables the fallback for
that module is enough to fold it and plain-quantize it.

```mermaid
flowchart TD
    A["AWQProcessor._quantize_layer"] --> B["named_childs<br/>bound resolved_module_config per member"]
    B --> C["_fallback_for_member<br/>normalize_fallback(dynamic override, run-wide)"]
    C --> D{"group has any fallback policy?"}
    D -- "no" --> E["AWQ search only<br/>missing rows raise"]
    D -- "yes" --> F{"member starved?<br/>captured rows below own threshold"}
    F -- "no" --> G["AWQ scale search"]
    F -- "yes" --> H{"shared search input<br/>and a covered member?"}
    H -- "yes" --> I["search covered members<br/>fold all, plain-quantize starved"]
    H -- "no" --> J["plain-quantize group"]
    J --> K["_fallback_strategy_for_module<br/>merged strategy on the fallback path"]
    I --> K
    style E fill:#f8d7da,stroke:#842029,stroke-width:2px
    style J fill:#fff3cd,stroke:#997404,stroke-width:2px
    style K fill:#d1e7dd,stroke:#0f5132,stroke-width:2px
```

| Before (#5820) | Now |
| --- | --- |
| `_starved_group_members` passed `self.fallback` for every projection (`awq_processor.py:3952` at `078e7f51`). | Each member's merged policy decides its own coverage. |
| `_quantize_layer` gated the group decision and the replay-error fallback on the run-wide object (`awq_processor.py:4118`, `:4292` at `078e7f51`). | The group's merged policy decides both. |
| The dynamic override was merged only after the fallback path was already chosen (`awq_processor.py:966-983` at `078e7f51`). | One merged resolver feeds selection and strategy. |

On the tiny Llama fixture in
`tests/test_awq_dynamic_fallback_threshold_5820.py`, a dynamic STDCLIP
threshold of 4096 used to reproduce the disabled-threshold pack (255 of 256
packed words away from the run-wide STDCLIP control; maximum scale error
0.0055802 against 0.0000109645). It now matches the run-wide STDCLIP control
exactly, while a threshold below the captured coverage still keeps AWQ.

## Scale-search safety

The search-safety checks were introduced in [#4144](https://github.com/ModelCloud/GPTQModel/pull/4144).

The existing group dictionaries and tag syntax are unchanged. Search uses
the original captured activation `x` and candidate weights `Q(W * s) / s`.
Non-quantized routers and shared gates therefore keep their original weights
during search. Only final `apply_scale`, which changes the source to emit
`x / s`, multiplies those consumers by `s`. Scaling them during search would
change routing even with identity quantization.

```mermaid
flowchart TD
    A["AWQProcessor._quantize_layer"] --> B["_validate_scale_group_conflicts"]
    B --> C{"source or target overlap?"}
    C -- "yes" --> X["raise _AWQScaleGroupError<br/>no folds applied"]
    C -- "no" --> P["filter low-sample / missing-feature groups"]
    P --> D["_search_best_scale<br/>_validate_scale_group"]
    D --> E{"exact live bindings, widths,<br/>and targets inside replay boundary?"}
    E -- "no" --> F{"configured fallback enabled?"}
    E -- "yes" --> G0["re-enter capture-time MoE<br/>routing override on replay"]
    G0 --> G["reference forward<br/>temporary target observation hooks"]
    G --> H["finally remove observation hooks"]
    H --> R{"reference replay succeeded?"}
    R -- "no" --> Y["propagate evaluation exception"]
    R -- "yes" --> I{"every declared target executed?"}
    I -- "no" --> F
    I -- "yes" --> J["_compute_best_scale<br/>20 candidates: Q(W * s) / s<br/>routers and shared gates unchanged"]
    J --> K["finally restore original weights"]
    K --> K2["finally restore native MoE top-k"]
    K2 --> L{"candidate evaluation succeeded?"}
    L -- "yes" --> M["collect scale results<br/>apply_scale once per source<br/>apply_quant"]
    L -- "no" --> Y
    F -- "yes" --> N["warn with group and reason<br/>exclude group from folding and clipping<br/>apply_quant with empty scales"]
    F -- "no" --> X
    style X fill:#f8d7da,stroke:#842029,stroke-width:2px
    style Y fill:#f8d7da,stroke:#842029,stroke-width:2px
    style N fill:#fff3cd,stroke:#997404,stroke-width:2px
```

| Before (baseline `c9efdbacf`) | Now |
| --- | --- |
| Candidate search multiplies non-quantized consumer weights (`awq_processor.py:3099-3119`). | Search leaves them unchanged; final folding still compensates them. |
| Shared-source merge retains the first inspection module for cross-root groups (`models/base.py:5540-5546`). | Group construction remains unchanged, but search rejects boundaries missing any *scored* member rather than scoring only one branch. A group the tree declares across sibling branches of one norm (`_awq_allow_outside_fold`) folds the out-of-boundary members **and scale consumers** instead of scoring them: they read the same tensor, so the same scale is exact, and `apply_scale` compensates every declared consumer. Only members the search scores must be replayed (#4741, #4898). |
| A contained target need not execute during replay (`awq_processor.py:2551-2572`). | Reference-forward observation rejects unexecuted targets before candidate mutation; observation hooks are always removed. |
| Replay always routes with the model's native top-k, even when calibration captured features under `ExpertsRoutingOverride` (`awq_processor.py:2358-2661`). | Reference and candidate replays re-enter the capture-time override; a target that still cannot execute rejects the group. |
| Weight restoration occurs only after successful grid evaluation (`awq_processor.py:3267-3278`). | Restoration is in `finally`, covering quantizer, replay, and loss exceptions. |
| Search results are collected without a group-validation fallback (`awq_processor.py:2352-2359`). | Unsafe replay groups use the existing configured fallback with a warning, or raise with fallback disabled. Conflicting folds always raise. |

All paths above are layer-owned: `process` holds `_AWQLayerState.lock` while
calling `_quantize_layer`. Observation state, hooks and master weights are local
to a search; no process-global trace or persistent weight cache is installed.
The hook APIs are already used for calibration. This change does not qualify
new concurrent mutation of the same module or claim free-threaded validation.

Validation checks **both** containment and execution. An MoE expert not reached
by the retained reference samples makes the group unsupported for that replay;
fallback is conservative even if earlier calibration batches reached it. Code
that bypasses module calls and reads weights functionally also requires a
compatible existing Python group/replay override, or fallback. The checks do
not prove an entire dynamic graph or discover undeclared functional consumers.

Replay routing must match capture. Calibration forwards run under the configured
`MoEConfig(routing=ExpertsRoutingOverride(...))`, which can execute experts that
native top-k never selects. `_search_best_scale` therefore re-enters the same
override around both the reference and candidate replays before enforcing
coverage (#4151). Without that, an all-expert calibration was judged against a
native-top-k replay and a complete group fell back. The guard still rejects a
genuinely partial replay: a declared target that the block does not execute even
under the configured routing raises `_AWQScaleGroupError`. The separate
pre-existing fallback for experts that captured no features at all (for example
native routing that never reached them) is unchanged.

Cross-root groups are not split: independently folding their shared norm would
divide it repeatedly. Nor is inspection automatically widened to the decoder
layer: its input can differ from the captured norm output. A Python override
may supply an existing-format group with a compatible input, kwargs and full
objective. Until then, the complete shared-source group falls back together.
Default `AWQConfig.fallback` behavior is preserved; set `fallback=None` to make
these unsafe replay groups raise instead. Existing no-group/missing-feature
paths are not redesigned here.

Conflicts are checked **before** low-sample, width or missing-feature filtering.
Dropping one conflicting group first would let another group divide the shared
norm (or modify tied target weights) without compensating the filtered branch.

CPU regressions cover eager and chunked search, FP32/FP16/BF16 router neutrality,
an independent Torch folded-model quantization oracle, failure rollback,
complete/incomplete parallel objectives, capture-time routing overrides with a
genuinely unexecuted target, and real Qwen2/Qwen3 MoE and Cohere/Cohere2
topologies. Inference kernels, persistent quantized weight storage, architecture
declaration syntax, and replay mask/position reconstruction are unchanged. A
declarative graph/DSL and automatic cross-root replay are out of scope.

## Architecture relationship qualification

The source audit covered all 206 registered model types (182 definitions) at
`388f82f01`, without downloading checkpoint weights. It recorded 14 unsafe
model-type entries across 13 definition families. Those are counterexamples to
the inference rule, not a claim that every released checkpoint reaches the
same invalid fold. Matching channel widths and executing every **declared**
target cannot prove producer identity or discover an undeclared consumer.

The immediate repairs use existing declarations, not a graph compiler or DSL:

| Definition | Incorrect relationship before | Existing-format repair |
| --- | --- | --- |
| `cohere_compass` | The input norm also feeds the parallel MLP; `o_proj` is not its producer. | Shared `:in=normalized` on Q/K/V and gate/up, `o_proj:nosrc`; incomplete cross-root replay is rejected. |
| `granitemoe_swa` | Routed experts omitted the sibling shared MLP. | Shared input tags include `shared_mlp.input_linear`; incomplete cross-root replay is rejected. |
| `hunyuan_v1_moe` | The weightless gate container hid the actual router weight. | Declare the non-quantized affine consumer `mlp.gate.wg`. |
| `falcon_mamba` | Matching widths allowed a fold across SSM and nonlinear computation. | Mark mixer projections `:nosrc`; explicitly exclude convolution, time-step and state owners. |
| `moss` | Parallel attention/MLP and GELU were treated as sequential linear paths. | Mark the shared norm, attention output and activated MLP source `:nosrc`. |
| `internlm2` | The attention input and output projections shared stage zero. | Stage `wqkv:0`, `wo:1`. |
| `minicpm` | Parallel Q/K/V were sequential; MLP input and output stages were combined. | Group Q/K/V at stage zero and gate/up together; keep output projections in the next stage. |
| `gemma3n`, `gemma3n_text` | Folds omitted Laurel, crossed value normalization, or crossed AltUp/residual/activation boundaries. | Exclude the corresponding sources with `:nosrc`, declare AltUp/Laurel owners, and place per-layer output normalization after its projection. |
| `axk2`, `glm_moe_dsa` | An indexer projection was mistaken for a low-rank query producer; GLM's input fold omitted an indexer consumer. | Conservatively disable internal attention source folds and declare nested norm/indexer owners. FFN folds remain eligible. |
| `longcat_flash` | Indexed norms and dual residual sublayers were treated as one sequential path. | Declare indexed norm owners and nested router weights; disable the ambiguous attention/output sources. |
| `inkling_mm_model` | The norm fed unhandled native 3-D shared experts as well as routed experts. | Disable that norm's fold and explicitly exclude shared-expert, scale, convolution and relative-logit owners. |
| `qwen` | The inferred `w2` source crosses SiLU. | Mark `w2:nosrc`. |
| `granitemoehybrid` | Tree order folded `in_proj` into `out_proj` across the convolution and the SSM scan, and the post-attention norm was folded for the routed experts although `shared_mlp.input_linear` also reads it. | Mark `in_proj:nosrc`, `post_attention_layernorm:nosrc`; declare the router owner; `out_proj` folds the gated output `norm` (exact). |
| `lfm3`, `lfm3_vl`, `lfm3_omni` | The tree named the legacy `gate_proj`, so `operator_norm` was folded into Q/K/V **and** `o_proj`, while the live `g_proj` sigmoid gate was not compensated; `conv.conv` was undeclared. | Rename to `g_proj:!:nosrc` and keep it in the Q/K/V stage as a scale consumer; move `o_proj` to its own `:nosrc` stage; declare `conv.conv:!`. |

```mermaid
flowchart TD
    A["architecture module_tree<br/>parallel stages, :in, nested :!, :nosrc"] --> B["awq_get_modules_for_scaling"]
    B --> C["_merge_shared_source_nodes"]
    C --> D["_filter_awq_scaling_sources"]
    D --> E{"selected source is :nosrc?"}
    E -- "yes" --> P["drop smoothing group<br/>existing plain quantization path"]
    E -- "no" --> F["AWQProcessor._quantize_layer<br/>conflict, boundary and execution checks"]
    F --> G{"complete supported replay?"}
    G -- "yes" --> H["search and fold once"]
    G -- "no: _AWQScaleGroupError" --> I{"configured fallback enabled?"}
    I -- "yes" --> P
    I -- "no" --> X["raise; no unsafe fold"]
    F -- "raw forward exception" --> Y["propagate; not a safe fallback"]
    style P fill:#fff3cd,stroke:#997404
    style X fill:#f8d7da,stroke:#842029
    style Y fill:#f8d7da,stroke:#842029
```

Declaring a source `:nosrc` retains the pre-existing unconditional plain
quantization path for that group; this is different from rejecting a generated
group under `fallback=None`. The safety checks are not expanded to infer an
arbitrary graph. Raw replay exceptions (for example a native Bloom forward
missing its residual argument) remain errors, not configured fallback.

### `granitemoehybrid`: state-space path and cross-block consumer

`GraniteMoeHybridDecoderLayer` runs `mamba` on the Mamba layers, then gives the
post-attention norm to `block_sparse_moe` **and** `shared_mlp`
(`modeling_granitemoehybrid.py` on Transformers 5.22). The Mamba mixer splits
`in_proj` into a gate, the convolution input and `dt`; the convolution and the
SSM scan consume those channels before the gated `norm` feeds `out_proj`.

| Situation | Before | After |
| --- | --- | --- |
| `in_proj -> out_proj` group | emitted; a non-unit fold changed the full logits by 1.6e-1 | `in_proj:nosrc`, so the group is dropped |
| `mamba.norm -> out_proj` group | not reachable (`norm` was undeclared) | declared `norm:!`; the fold is exact (1.6e-7) |
| `post_attention_layernorm -> experts` group | emitted; the MoE replay boundary excludes `shared_mlp`, so logits drifted by 1.5e-2 | norm declared `nosrc`; routed and shared projections quantize plainly |
| Router owner | `plan.find("block_sparse_moe.router")` was `None` | `router:!` under the `:moe` root keeps its checkpoint weight and is a declared consumer of the MoE input norm |
| `conv1d`, `norm`, container parameters | undeclared / no explicit policy | `conv1d:!`, `norm:!`, container `mamba:!` |

```mermaid
flowchart LR
    N["post_attention_layernorm:!:nosrc"] --> R["block_sparse_moe.router:!"]
    N --> E["experts.N.gate_proj / up_proj"]
    N --> S["shared_mlp.input_linear"]
    E --> D["experts.N.down_proj"]
    I["input_layernorm"] --> P["mamba.in_proj:nosrc"]
    P --> C["mamba.conv1d:!"]
    C --> M["SSM scan / mamba.D / A_log / dt_bias"]
    M --> G["mamba.norm:!"]
    G --> O["mamba.out_proj"]
    style N fill:#fff3cd,stroke:#997404
    style P fill:#fff3cd,stroke:#997404
    style R fill:#e2e3e5,stroke:#41464b
```

`tests/module_tree/test_granitemoehybrid_model_tree.py` pins the owners, the
router policy, and the exact group set from the real tiny model; its
full-model fold invariant runs transformers' eager Torch Mamba-2 reference
(the installed `mamba_ssm` kernels are CUDA-only).

## Issue #4175: gated, lerped and hyper-connection consumers

These definitions emitted groups that passed `_validate_scale_group` but
changed the full-precision layer output when the fold was applied without
quantization:

| Definition | Incorrect relationship | Existing-format repair |
| --- | --- | --- |
| `intern_s2_preview`, `minicpmv4_6` | DeltaNet `in_proj_qkv`/`in_proj_z` read the same layer input norm, but the missing `:in=x` tags folded only `in_proj_qkv` and then folded the second branch into `in_proj_qkv`. | Restore Qwen3.5's `:in=x` tags: one group per input norm, with `in_proj_b`/`in_proj_a` compensated. |
| `minicpmv`, `minicpmo` | Sequential `q:0,k:1,v:2,o:3` folded the norm into `q`, then `q` into `k`; the ungrouped MLP put `down_proj` in the gate/up subset. | Q/K/V share stage 0 with `:in=x`, `o_proj:1`, gate/up `:in=x`, `down_proj:1`. |
| `diffusion_gemma` | The scale-free `v_norm` hides `v_proj` from `o_proj`, and the router was compensated as a consumer of `pre_feedforward_layernorm_2` although it reads the raw residual through its own scale-free norm. | `v_proj:...:nosrc`; declare the router after the expert block so it is not collected as a scale consumer. |
| `ricursive` | `g_proj` is a sigmoid gate on the attention output but was `o_proj`'s source; `v_proj` reads `lerp(value_residual, normed)` and was folded with the input norm. | `g_proj:nosrc`, `v_proj` in its own stage, and `input_layernorm:nosrc` because no single `v_proj` scale can undo a fold through the lerp. |
| `qwen4_exp`, `qwen4_exp_text` | No norms are declared, so attention `o_proj`/`out_proj` became the MLP input source across the gated hyper-connection mix. | Mark `o_proj`, `in_proj_a` and `out_proj` `:nosrc`; shared and routed experts fall back to plain quantization together. |

`tests/module_tree/test_awq_fold_safety_4175.py` applies every emitted fold
without quantization to a real tiny CPU layer and compares the full output
(~1e-7). That is the regression guard the width/binding validation could not
provide.

`tests/module_tree/test_awq_relationships.py` uses real tiny CPU model modules,
their `state_dict()` owners, nontrivial synthetic folds and full text outputs.
It covers shared/no-shared Granite, nested Hunyuan routing, a Falcon Mamba width
collision, AXK2/GLM indexers, LongCat residuals, dense/sparse Inkling text layers,
MHA/GQA CohereCompass and Gemma3n's AltUp/Laurel decoder. The legacy remote-code
families also have pinned-source audit probes; their CI declaration checks do
not claim that remote implementations were loaded by the CPU job. The
always-run registry contract covers every registered alias and method case.

## One group per norm

`_merge_shared_source_nodes` merges every node whose `prev_op` is the same gain
norm into a single scaling group. The merged group divides that norm once and
multiplies every quantized consumer (all routed experts plus `shared_experts`,
or both parallel branches) and every declared `_scale_consumers` entry (router,
`shared_expert_gate`, `g_proj`, `output_gate`) by the same scale. The merged
group re-inspects a common parent when every consumer shares one, so the scale
search still scores the combined block. If the retained inspection boundary
cannot cover the complete group, the search-safety checks above reject it.

## Parallel branches

When a parallel block reads one norm from two quantized branches, the second
branch has no intervening norm to fold into. Each consumer carries the same
`:in=<tag>` flag, and `_select_prev_op` reuses the first branch's resolved norm
for the tag. Example (Falcon):

```
"input_layernorm": ("input_layernorm:!",),
"self_attention": ("query_key_value:0:in=x", "dense:1"),
"post_attention_layernorm": ("post_attention_layernorm:!",),
"mlp": ("dense_h_to_4h:0:in=x:nosrc", "dense_4h_to_h:1"),
```

## Declared non-sources

Two flags declare data flow:

- `:postnorm` marks a sandwich post-norm whose output is consumed by a residual
  add. It is checked during `prev_op` selection.
- `:nosrc` marks any other leaf whose output is not the direct input of the
  following scaling consumer. It is checked after selection by
  `_filter_awq_scaling_sources`, so it also covers quantized linears.

A `nosrc` flag on a tree leaf means: *this module's output is not the direct
input of the following scaling consumer*. Folding a scale into it changes the
decoder layer output because of one of:

- a residual add between the producer and the consumer,
- a parallel branch that consumes the same activation,
- a non-homogeneous activation (GELU, SiLU, xIELU, sigmoid gate) between the two
  linears,
- a fused projection whose output does not match the consumer's input width.

Syntax:

```
"input_layernorm": ("input_layernorm:!:nosrc",)
"mlp": ("fc1:0:nosrc", "fc2:1")
```

`nosrc` is metadata only: it does not change quantization, subset grouping, or
the emitted block list (the compiler drops non-`!`/`?` flags from block names).

| Definition | Declared non-sources | Why |
| --- | --- | --- |
| `olmo2`, `olmo3`, `exaone4`, `exaone4_5`, `gemma4_text` | post-attention / post-FFN norms (`:postnorm`) | the norm output is added to the residual |
| `opt` (post-norm) | `self_attn_layer_norm` | the normed value is also the MLP residual |
| `phi`, `gptj`, `codegen` | input norm, attention out, MLP in | parallel branches, residual, GELU |
| `gpt_neox`, `gpt_neo`, `gpt_bigcode`, `bloom`, `rw`, `falcon` | MLP input projection | GELU is not positively homogeneous |
| `starcoder2`, `apertus`, `mpt` | MLP up projection | non-homogeneous activation |
| `gpt_oss` | expert `up_proj` | `(clamp(up) + 1) * glu` is not homogeneous in up |
| `afmoe`, `muse_glimmer`, `hrm_text` | attention `gate_proj` | `o_proj` reads `attn * sigmoid(gate)` |
| `gemma4_text` | `v_proj` | MHA configs pass v through a scale-free `v_norm` |
| `minimax` (original) | both block norms | the normalized value is the residual |

## MoE-root capture

The routed-expert gate/up group reads the shared pre-router input of the
declared `:moe` root, not the per-expert routed rows. `BaseQModel.awq_input_feature_aggregation`
returns `capture_root` for every `ModulePlan.moe_roots()` template, and
`AWQProcessor.preprocess` seeds an empty task for that root. The activation is
recorded by a forward-pre-hook registered in `run_subset_stage`.

The hook is resolved from the module plan, not from the optional
`moe_lifecycle_hooks` object: the root input is a plain pointwise capture and
does not need routing control. Before this, a definition without lifecycle
hooks (Granite-MoE, gpt-oss, ERNIE-4.5-MoE, Gemma-4, DBRX-converted,
Diffusion-Gemma, Kimi, Nemotron-H puzzle) left the root feature as `(0,)`.
`awq_get_modules_for_scaling` then read `input_width = 0`, treated every expert
as width-foreign, and dropped the whole gate/up group without a warning
(#4697).

```mermaid
flowchart TD
    A["awq_input_feature_aggregation<br/>plan.moe_roots()"] --> B["capture_root policy<br/>for the root template"]
    B --> C["AWQProcessor.preprocess<br/>seed tasks root"]
    D["run_subset_stage<br/>_model_module_plan(looper.gptq_model)"] --> E{"AWQ and layer has<br/>plan.is_expert?"}
    E -- "yes" --> F["resolve layer-level root<br/>module.get_submodule(template)"]
    F --> G["register_moe_root_capture_hook<br/>forward_pre_hook on root"]
    E -- "no" --> H["no root hook"]
    G --> I["record_moe_root_input_feature<br/>bounded token rows"]
    I --> J["_layer_input_features<br/>aggregate root feature"]
    J --> K{"awq_get_modules_for_scaling<br/>MoE gate/up block"}
    K -- "root feature non-empty" --> L["group: norm -> experts gate/up<br/>plus declared consumers"]
    K -- "root feature empty" --> M["warn and skip group<br/>experts stay on plain quantization"]
    L --> N["AWQProcessor._quantize_layer<br/>apply_scale"]
    style M fill:#fff3cd,stroke:#997404,stroke-width:2px
```

| Mechanism | Before | After |
| --- | --- | --- |
| Root hook source | `moe_lifecycle_hooks.get_moe_block_for_subset` only (`gptqmodel/looper/stage_subset.py`) | module-plan `moe_roots()` for AWQ, deduplicated against the lifecycle block |
| Definitions without lifecycle hooks | root feature `(0,)`, gate/up group dropped silently | root feature captured, group emitted |
| Empty root feature | width `0` treated every consumer as foreign; only `log.debug` | explicit `log.warning` names the root and the skipped group |
| Nested roots (`mixer.experts`) | group builder reads a layer-level root feature | unchanged: the existing shared-expert feature fallback supplies the input |

## Before and after

| Mechanism | Before | After |
| --- | --- | --- |
| Routed + shared experts | two groups from one norm; each fold left the other unscaled (0.049 / 0.080 on DeepSeek-V3) | one merged group (`_merge_shared_source_nodes`) |
| Parallel branches (Falcon, Cohere) | second branch folded into the first branch's output projection (0.25) | both branches carry `:in=<tag>`; `_select_prev_op` reuses the shared source |
| Non-homogeneous fc1→fc2 | folded across GELU (0.02-0.05) | `:nosrc` on fc1 |
| Gated attention | `gate_proj` became block 0's last module, folded into `o_proj` (0.28) | `:nosrc` on `gate_proj` |
| Weightless / gated norms | `apply_scale` raised (`NotImplementedError`, `AttributeError`) | `is_foldable_source` drops the group; layer is quantized plainly |
| MoE block with a non-foldable norm | fell back to the last expert linear | `last_norm_candidate` returns the block norm; group is dropped |
| `Cohere2` / `Cohere2-MoE` norms | crash (`not in allowed_norms`) | added to `allowed_norms` |
| `Cohere2-MoE` shared experts | undeclared, unscaled (0.01) | declared and merged into the routed group |
| `olmoe` `q_norm`/`k_norm` | full-width norms picked as `prev_op` (0.29) | group index `0` keeps them inside the q/k/v block |
| `laguna` `g_proj` | omitted consumer (0.006) | declared `:!` sibling; collected as a scale consumer |

## Remote-code family trees

The tree is the only declaration of which module feeds a scaling group. Three
remote-code families shipped a tree that did not describe their forward graph,
and two of them quantized to garbage without raising:

| Definition | Before | After | Effect |
| --- | --- | --- | --- |
| `instella` | inherited `LlamaQModel.module_tree` (`input_layernorm` / `post_attention_layernorm`) | own tree with `pre_attention_layernorm` / `pre_feedforward_layernorm` | q/k/v get a norm source; `mlp.gate/up` no longer fold into `self_attn.o_proj` across the residual (wikitext-2 PPL 27997 -> 27.7 on the 3B checkpoint) |
| `k2_horizon` | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `o_proj` as separate dict nodes | one flat `""` tuple `q_proj:0`, `k_proj:0`, `v_proj:0`, `gate_proj:0:nosrc`, `o_proj:1` | one group per attention input instead of chained single-module groups (PPL 1100 -> sane) |
| `telechat2` | `{"dense": ("dense",)}` only | `("query:0", "key_value:0", "dense:1")` | `query`/`key_value` are quantized instead of silently staying fp16 |

Issue #4121 added four more definitions whose tree did not match the forward
graph. All of them are now in `FAMILY_MODEL_TYPES`:

| Definition | Before | After | Effect (per-group fold error) |
| --- | --- | --- | --- |
| `gpt_pangu` | `mlp: ("c_fc:0", "c_proj:1")`; the last block's `ln_1` group contained `attn.q_proj` | `c_fc:0:nosrc`; `awq_get_modules_for_scaling` drops the top-query block's `q_proj` | GELU fold 4.5e-2 -> dropped; `ln_1 -> q_proj` 3.4e-3 -> `q_proj` excluded, remaining k/v group 1.2e-8 |
| `olmo_hybrid` | one static tree; `post_attention_layernorm` was a gain source on every layer | `awq_post_norm_modules(layer)` adds it on the full-attention layer only | 3.6e-1 -> group dropped on attention layers; linear-attention fold stays exact (1.6e-8) |
| `gemma4`, `gemma4_unified` | composite/unified trees kept `v_proj:0` and `post_feedforward_layernorm:!` | copied `v_proj:0:in=x:nosrc` and `:postnorm` from the text tree; `post_per_layer_input_norm` is `:postnorm` too | `v_proj -> o_proj` 1.9e-1 -> dropped; `post_feedforward_layernorm -> per_layer_input_gate` 2.4e-1 -> dropped; `post_per_layer_input_norm -> per_layer_projection` 2.2e-1 -> dropped |
| `nemotron_h`, `nemotronh_nano_omni_reasoning_v3` | `mixer.up_proj` fed `down_proj`; the routed and shared expert MLPs repeated the same fold | `up_proj:3:nosrc`, plus `mixer.shared_experts.up_proj:nosrc` and `mixer.experts.#.up_proj:nosrc` | `relu(x)^2` folds measured up to 8.7e-3 per expert -> dropped; the block input norm remains the only source, and only one gain norm feeds routed + shared experts |

Every family listed in `FAMILY_MODEL_TYPES` is covered by the
output-preservation invariant: folding a random scale into each group must
leave the decoder layer's logits unchanged to `1e-5` relative.

## Trees that alternate layer shapes

`ModulePlan` is built from one `module_tree` per model, but OLMo Hybrid's
decoder layers do not share a shape. `OlmoHybridAttentionDecoderLayer`
post-norms the attention output and *adds it to the residual*:

```
residual = h
h = post_attention_layernorm(self_attn(h))
h = residual + h
```

`OlmoHybridLinearAttentionDecoderLayer` feeds the same-named norm straight into
the MLP, where a fold is exact. A static `:postnorm` flag would therefore drop
a valid group on half the layers, and no flag would fold across a residual on
the other half.

```mermaid
flowchart TD
    A["awq_get_modules_for_scaling"] --> B["awq_post_norm_modules(layer)"]
    B --> C["set(get_post_norm_modules())"]
    C --> D{"layer has self_attn?"}
    D -- "yes, full attention" --> E["add post_attention_layernorm"]
    D -- "no, linear attention" --> F["static set only"]
    E --> G["_is_gain_norm returns False<br/>group dropped, consumer quantized plainly"]
    F --> H["fold into post_attention_layernorm"]
    style G fill:#fff3cd,stroke:#997404,stroke-width:2px
    style H fill:#d1e7dd,stroke:#0f5132,stroke-width:2px
```

`get_post_norm_modules` stays a classmethod for the definitions that declare one
static set; `awq_post_norm_modules` is the per-layer extension point.

## Coverage: declared specs versus undeclared leaves

`LayerModuleCoverageTracker` used to count only *declared* specs. A quantizable
`nn.Linear` that no spec named was invisible: every declared spec matched while
the leaf stayed in the source dtype. TeleChat2's `query`/`key_value` were
exactly this, and the report said "all 4 specs matched".

```mermaid
flowchart TD
    A["ModuleLooper.create_named_modules"] --> B["plan = _model_module_plan(gptq_model)"]
    B --> C{"candidate is nn.Linear / Conv1d / Conv2d in full?"}
    C -- "no" --> D["ignore"]
    C -- "yes" --> E{"plan.find(name) is None?"}
    E -- "no" --> F["declared: hit/miss accounting"]
    E -- "yes" --> G["record_undeclared(name, layer_index)"]
    G --> H["report(): NOT declared by the definition tree"]
    style G fill:#fff3cd,stroke:#997404,stroke-width:2px
    style H fill:#fff3cd,stroke:#997404,stroke-width:2px
```

The LM head is excluded because it is quantized through its own path, not the
layer tree. A leaf that is intentionally not quantized must still be declared
with `:!` or `:?`; "no spec at all" is now reported.

## AWQ replay scaffolding

AWQ scores a multi-module scaling group by replaying its common parent. Two
inputs to that replay are captured per linear, not per parent, so they can be
stale or incomplete.

### Position kwargs

`AWQProcessor._prepare_module_forward` rebuilds position scaffolding whenever
it does not match the sample being replayed. Before, `position_ids` was only
added when the key was *absent*, so a 430-token capture was replayed with a
522-token activation (EXAONE-3.0) or 48 vs 20 (tiny DeciLM/Nanbeige), crashing
in `apply_rotary_pos_emb`.

```mermaid
flowchart TD
    A["AWQProcessor._prepare_module_forward"] --> B{"definition overrides<br/>awq_module_replay_inputs?"}
    B -- "yes" --> C["TeleChat2: promote x to 3-D,<br/>add residual and causal bool mask"]
    B -- "no" --> D["module_kwargs = dict(module_kwargs)"]
    C --> D
    D --> E{"position_ids length == seq_len?"}
    E -- "no" --> F["stale_position_ids = True"]
    E -- "yes" --> G["keep captured position_ids"]
    F --> H{"root rotary found?"}
    G --> H
    H -- "no" --> I["module.rotary_emb / module.rotary"]
    H -- "yes" --> J["_get_rotary_for_device"]
    I --> K{"rotary available?"}
    J --> K
    K -- "yes" --> L["rebuild position_ids and position_embeddings<br/>for seq_len"]
    K -- "no" --> M["rebuild position_ids with arange;<br/>drop mismatched position_embeddings"]
    style M fill:#fff3cd,stroke:#997404,stroke-width:2px
```

`exaone` also declares `rotary_embedding = "transformer.rotary"`, so its
model-level rotary is found directly. Nanbeige/DeciLM keep the rotary on the
attention module and take the `module.rotary_emb` path.

### Multimodal rotary resolution

A multimodal wrapper nests the decoder one level deeper than a text model.
Qwen2-VL, Qwen2.5-VL, Gemma 3 and Mistral 3 keep the decoder rotary on
`model.language_model.rotary_emb`, and their definitions do not declare
`rotary_embedding`. `AWQProcessor._get_root_rotary` looked only at
`model.rotary_emb`, returned `None`, and the replay then dropped the captured
`position_embeddings`; the attention unpacked `None`:

```
AWQ Qwen2-VL-2B-Instruct -> TypeError: cannot unpack non-iterable NoneType object
```

```mermaid
flowchart TD
    A["AWQProcessor._get_root_rotary"] --> B{"gptq_model.rotary_embedding declared?"}
    B -- "yes" --> C["get_module_by_name_prefix"]
    B -- "no" --> D{"model.rotary_emb or model.rotary?"}
    D -- "yes" --> E["return it"]
    D -- "no" --> F["_scan_root_rotary over named_modules"]
    F --> G{"segment in vision/audio tower?"}
    G -- "yes" --> H["skip: model.visual.rotary_pos_emb"]
    G -- "no" --> I{"numbered segment? per-layer rotary"}
    I -- "yes" --> J["skip: handled by module-local fallback"]
    I -- "no" --> K{"looks like a rotary embedding?"}
    K -- "yes" --> L["shallowest match wins"]
    K -- "no" --> M["not a candidate"]
    style H fill:#f8d7da,stroke:#842029,stroke-width:2px
    style J fill:#f8d7da,stroke:#842029,stroke-width:2px
    style L fill:#d1e7dd,stroke:#0f5132,stroke-width:2px
```

Two exclusions keep the scan from picking the wrong module:

- **Vision/audio towers.** `model.visual.rotary_pos_emb` is a rotary embedding
  too, and by name it is the shallowest match on Qwen2-VL. Its frequency width
  is the vision head width, so rebuilding the decoder's cos/sin from it would
  silently produce a wrong-length tensor.
- **Per-layer rotaries.** A rotary under a numbered container
  (`layers.0.self_attn.rotary_emb` on Nanbeige/DeciLM) belongs to one layer and
  is usually an old-style `(x, seq_len)` module. Those families are handled by
  the module-local fallback in `_prepare_module_forward`, and `_get_root_rotary`
  keeps returning `None` for them so the other replay paths behave as before.

The resolved module is cached on the processor, so the scan runs once.

### Module-owned replay inputs

`BaseQModel.awq_module_replay_inputs` lets a definition add arguments the
per-linear capture cannot record. TeleChat2's `TelechatAttention.forward`
requires `(hidden_states, residual, attention_mask, ...)`; the default
implementation returns `(x, module_kwargs)` unchanged, and TeleChat2 promotes
the 2-D capture to 3-D, passes the block input as `residual` (an additive
constant shared by reference and candidate), and builds a causal bool mask.

### Module replay failures and the scale-group fallback

A replay can fail for reasons the processor cannot pre-check. Two contracts keep
such a failure from aborting `quantize()`:

1. A tensor-geometry `RuntimeError` from the replayed module is converted to
   `_AWQScaleGroupError` (`_is_replay_geometry_error`), so the configured
   fallback quantizes the consumer plainly (#5085).
2. A bare `AssertionError` raised **by the replayed module** is converted the
   same way (`AWQProcessor._invoke_replay_module`). Hymba's `reuse_kv` mixer
   asserts `kv_last_layer is not None`; that value is threaded between layers by
   the forward executor, is not part of the per-linear capture, and cannot be
   reconstructed from the retained token rows (#5098).

The conversion is scoped to the replay call: the processor's own assertions and
non-`AssertionError` exceptions still propagate, so a numerical or logical
failure is never silently downgraded to plain quantization.

```mermaid
flowchart TD
    A["AWQProcessor._search_best_scale"] --> B["replay boundary forward<br/>_iter_prepared_forward_outputs"]
    B --> C{"module forward raised?"}
    C -- "no" --> D["score candidates"]
    C -- "RuntimeError naming<br/>broadcast or matmul shapes" --> E["_AWQScaleGroupError"]
    C -- "AssertionError from the module" --> E
    C -- "any other exception" --> F["propagate; abort quantize()"]
    E --> G{"AWQConfig.fallback set?"}
    G -- "yes" --> H["plain quantization for the group<br/>log the configured fallback"]
    G -- "no" --> I["raise; no unsafe fold"]
    style F fill:#f8d7da,stroke:#842029,stroke-width:2px
    style I fill:#f8d7da,stroke:#842029,stroke-width:2px
    style H fill:#fff3cd,stroke:#997404,stroke-width:2px
```

### `hymba`: KV-reuse layers

Every Hymba checkpoint shares KV across layers (`kv_reuse_group`, 15 consumer
layers of 32 on `nvidia/Hymba-1.5B-Base`). A consumer's `HymbaBlock` projects no
K/V of its own: it reads the producer layer's `kv_last_layer`, which
`ForwardExecutor` injects for the real forward only
(`looper/forward_executor.py`, enabled by `HymbaQModel.write_shared_kv_cache`).

`input_layernorm -> mamba.in_proj` is a single-member group whose declared
inspection module was the whole mixer, so the scale search replayed
`HymbaBlock(x, **captured_kwargs)` without that value and asserted. Block replay
was also the wrong objective: the group has one Linear member, and the
projection's own reconstruction error is exactly what the search ranks.

`HymbaQModel.awq_get_modules_for_scaling` re-anchors that group's
`module2inspect` onto the `mamba.in_proj` `nn.Linear`. The fold itself is
unchanged: `input_layernorm` is still divided and `in_proj`'s input rows are
still scaled (`tests/test_hymba_awq_4891.py` pins that folding the resulting
groups still reproduces the logits).

```mermaid
flowchart LR
    subgraph OLD["before: boundary is the mixer"]
        N1["input_layernorm"] --> M1["HymbaBlock.forward"]
        M1 --> R1["needs kv_last_layer<br/>and packed causal mask"]
        R1 --> X1["AssertionError<br/>aborts quantize()"]
    end
    subgraph NEW["after: boundary is the projection"]
        N2["input_layernorm"] --> P2["mamba.in_proj Linear"]
        P2 --> S2["score scale on the projection"]
        S2 --> F2["fold the norm; consumer stays quantized"]
    end
    style X1 fill:#f8d7da,stroke:#842029,stroke-width:2px
    style F2 fill:#d1e7dd,stroke:#0f5132,stroke-width:2px
```

| Situation | Before | After |
| --- | --- | --- |
| `input_layernorm -> mamba.in_proj` on a KV-reuse layer | replayed `HymbaBlock`; `assert kv_last_layer is not None` aborted `quantize()` | replayed `mamba.in_proj`; the projection is the whole group and owns no cross-layer state |
| `input_layernorm -> mamba.in_proj` on a producer layer | replayed `HymbaBlock`; group scored through the mixer | same re-anchor: one Linear member, so the projection alone is the objective |
| Replay raises `AssertionError` in the module | propagated and aborted the run | `_AWQScaleGroupError`; configured fallback or a raise when `fallback=None` |
| Processor-side assertion or non-assertion failure | propagated | unchanged: still propagates |
| Multi-consumer norm or non-Linear target | replayed the block | unchanged: the mixed replay is kept so the existing validation decides |

## Looped decoder calibration

HRM-Text, Ouro (`total_ut_steps=4`), Nanbeige 4.2 (`num_loops=2`) and
SenseNova-Loop (`loop_settings = {start, end, eval_count}`) run the same
physical layers several times per forward. The default capture aborts at the
first layer, so each processor's Hessian only saw loop 0. Ouro's steps 1-3 have
cosine -0.03..0.42 against the single captured input.

A definition opts in by deriving from `LoopedLayerCaptureMixin`. When the mixin
is missing the run still succeeds, so the guard is a definition-level contract:
`tests/test_looped_decoder_capture.py` asserts the mixin is present for every
looped family **and** that the captured invocation counts equal the model's own
`execution_plan()`. SenseNova-Loop regressed exactly this way in #5081: its
docstring claimed the looper already handled the loop, but the definition
derived only from `Qwen3_5QModel`, so layers inside the looped block were
calibrated on one visit and the post-loop layers on one-loop inputs.

```mermaid
flowchart TD
    A["ModuleLooper.cache_inputs"] --> B["lazy build_quantization_input_cache"]
    B --> C{"definition uses<br/>LoopedLayerCaptureMixin?"}
    C -- "no" --> D["StageInputsCapture:<br/>abort at the first layer"]
    C -- "yes" --> E["_materialize_looped_capture_model"]
    E --> F["one full forward per sample;<br/>pre-hook on every physical layer"]
    F --> G["per-layer bucket: every loop invocation"]
    G --> H["cache.captured_batch_count =<br/>max bucket length"]
    H --> I["ModuleLooper sets processor.num_batches"]
    I --> J["pre_quantize serves the<br/>current layer's invocations"]
    style D fill:#f8d7da,stroke:#842029,stroke-width:2px
```

`_resolve_batch_total` caps the replay at `processor.num_batches`, so publishing
`captured_batch_count` is what keeps loops 1..N-1 from being truncated.

## Top-query layer replay

`GPTPanguModel.forward` applies `ln_f` before the last decoder block and passes
it `custom_query = wqe(position_ids)`, so that block's `q_proj` reads query
embeddings rather than hidden states. `PanguAlphaQModel.prepare_layer_replay_kwargs`
now mirrors that for the last block only; the measured last-layer `q_proj`
relative error on query embeddings dropped from GPTQ 0.130 (worse than RTN
0.101) to GPTQ 0.0558 (better than RTN 0.0933).

```mermaid
flowchart TD
    A["BaseQModel.prepare_layer_replay_kwargs"] --> B{"last block is layer?"}
    B -- "no" --> C["replay kwargs unchanged"]
    B -- "yes" --> D["layer input = ln_f(hidden_states)"]
    D --> E{"position_ids tensor?"}
    E -- "no" --> F["arange over seq_len"]
    E -- "yes" --> G["use captured position_ids"]
    F --> H["custom_query = wqe(position_ids)"]
    G --> H
```

## Failure and ambiguity contract

| Situation | Before | After |
| --- | --- | --- |
| No norm in the tree and the first block is quantized | every block skipped, empty map, pack crash | first block skipped, later blocks use the previous linear; invalid ones are filtered |
| `prev_op` is a declared non-source | folded anyway (silent output change) | group dropped; consumer is quantized plainly |
| `prev_op` gain width differs from the consumer input | `apply_scale` shape error | group dropped |
| `prev_op` is a weightless / gated norm / Conv1D / HyperConnection | `NotImplementedError` or `AttributeError` | group skipped by `is_foldable_source` |
| No group survives | layer finalized unquantized; pack crash | layer quantized plainly |

## Validation

`tests/test_awq_scaling_groups.py` builds a tiny model for every family in
`FAMILIES`, folds a random scale in `[0.5, 1.5]` with the library's own
`apply_scale`, and requires the decoder-layer logits to change by less than
`1e-5` relative (Llama measures ~1.5e-7). `tests/test_awq_empty_groups.py` pins
both empty-group paths to `apply_quant(..., scales_list=[])`.

Issue #4093 item coverage:

| Test | Invariant |
| --- | --- |
| `tests/test_awq_scaling_groups.py` (`instella`, `k2_horizon`) | fold preserves the decoder-layer output |
| `tests/test_instella_support.py` | the tree declares the real Instella norm names and resolves on the checkpoint's own layer |
| `tests/test_k2_horizon_support.py` | q/k/v/gate form one flat group; `o_proj` is the separate consumer |
| `tests/test_telechat2_coverage.py` | `query`/`key_value`/`dense` are quantized and no leaf is undeclared; replay supplies `residual` + mask |
| `tests/test_awq_position_kwargs.py` | stale `position_ids` are rebuilt, mismatched `position_embeddings` dropped, module-local rotary rebuilds cos/sin |
| `tests/test_pangu_alpha_replay.py` | last layer replays `ln_f` + `custom_query`; GPTQ beats RTN on the top-query `q_proj` |
| `tests/test_looped_decoder_capture.py` | Ouro captures 4 invocations, Nanbeige 2, per-loop cosine < 0.99; SenseNova-Loop captures the published `execution_plan()` exactly and its post-loop replay input equals the real model's |
| `tests/test_layer_module_coverage.py` | undeclared quantizable leaves are recorded and warned about |
| `tests/test_awq_moe_root_capture_4697.py` | a real tiny Granite-MoE/gpt-oss quantize applies the routed-expert gate/up group from the declared root feature; an empty root warns instead of silently dropping the group |

Issue #4121 item coverage:

| Test | Invariant |
| --- | --- |
| `tests/test_awq_scaling_groups.py` (`gpt_pangu`, `olmo_hybrid`, `gemma4`, `gemma4_unified`, `nemotron_h`) | fold preserves the decoder layer's logits to `1e-5` relative |
| `tests/module_tree/test_nemotron_h_model_tree.py` | the Omni definition shares the Nemotron-H declaration, and its fold preserves the real remote decoder layer to `1e-5` relative |
| `tests/test_awq_vl_rotary_resolution.py` | `_get_root_rotary` finds `model.language_model.rotary_emb`, never the vision tower, ignores per-layer rotaries, and `_prepare_module_forward` rebuilds cos/sin for a nested rotary; tiny Qwen2-VL quantizes with AWQ and GPTQ |

### On-demand CPU test suites

From the repository root, use the [CPU test runner](../tests/run_awq_cpu.sh)
with the project dependencies (including CPU-capable PyTorch), `pytest`,
`pytest-timeout`, and `parameterized` installed:

```bash
bash tests/run_awq_cpu.sh unit   # Offline regressions
bash tests/run_awq_cpu.sh dense  # SmolLM2-135M
bash tests/run_awq_cpu.sh moe    # TinyMoE-100M
```

These suites do not run automatically on pull requests. The optional
checkpoint tests in `tests/test_awq_hf_cpu.py` use revision-pinned models from
Hugging Face. They download real
safetensors, compare held-out logits and next-token loss before/after 4-bit AWQ,
check all expected packed projections and unquantized MoE routers, then save,
reload with the same activation dtype, and check logits and greedy generation.
Network or download errors fail the test rather than silently skipping it.
The MoE test permits configured fallback when calibration does not execute all
experts and records how many groups used it. These short deterministic checks
guard against catastrophic drift; they are not a corpus-level perplexity or
model-quality qualification.

## Not in scope

- MLA attention groups (`deepseek_v2/v3/v32`, `deepseek_v4`, `axk2`) whose
  `q_b_proj`/`kv_b_proj` read different low-rank tensors are filtered by the
  existing width check; the MoE routed+shared merge is covered by the structural
  test.
- `nosrc` is a static declaration. A new family whose forward graph has a
  residual, parallel branch, or non-homogeneous activation between a producer
  and its consumer must declare it; the invariant test catches regressions for
  the families listed above.
