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

Introduced by the fixes for issues #4005, #4017, and #4039.

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
    O --> P{"prev_op nosrc<br/>or width mismatch?"}
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

## One group per norm

`_merge_shared_source_nodes` merges every node whose `prev_op` is the same gain
norm into a single scaling group. The merged group divides that norm once and
multiplies every quantized consumer (all routed experts plus `shared_experts`,
or both parallel branches) and every declared `_scale_consumers` entry (router,
`shared_expert_gate`, `g_proj`, `output_gate`) by the same scale. The merged
group re-inspects a common parent when every consumer shares one, so the scale
search still scores the combined block.

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

## Not in scope

- MLA attention groups (`deepseek_v2/v3/v32`, `deepseek_v4`, `axk2`) whose
  `q_b_proj`/`kv_b_proj` read different low-rank tensors are filtered by the
  existing width check; the MoE routed+shared merge is covered by the structural
  test.
- `nosrc` is a static declaration. A new family whose forward graph has a
  residual, parallel branch, or non-homogeneous activation between a producer
  and its consumer must declare it; the invariant test catches regressions for
  the families listed above.
