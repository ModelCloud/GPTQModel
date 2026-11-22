---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-awq-model-onboarding
description: Qualify AWQ source/consumer relationships, scale-search objectives, native replay context, and safe fallback when adding a model or alias, changing a definition, or modifying AWQ grouping, search, replay, or folding.
---

# AWQ model-onboarding safety gate

## At a glance

For every new model/alias, definition variant, or grouping/search/replay/fold
change, qualify each layout against native forward code or use tested plain
quantization. A passing [module plan](../gptqmodel-model-plan-tests/SKILL.md),
matching widths, live bindings and observing all declared targets cannot reveal
an undeclared consumer. State the AWQ policy even for non-AWQ onboarding.

Start in `gptqmodel/models/definitions/` and `gptqmodel/looper/awq_processor.py`;
use the evidence matrix below, not architecture names or generation quality.

## 1. Establish the native execution relationships

Read the actual modeling implementation for each supported layout/config
variant, including Defuser conversions and forward helpers. Record the source
version/revision and forward locations in the PR. Use real tiny CPU models for
execution tests; meta models only establish structural coverage. A tree-generated
fake model or mocked forward cannot qualify a real architecture.

For every proposed group, record review evidence (a table or prose, not a new
runtime declaration language):

- The exact source output and each projection input it produces.
- **All** consumers of that output: parallel attention/MLP branches, routed
  and shared experts, routers/shared gates, residual paths, and nested affine
  children such as `gate.wg`. Excluding a module from quantization does not
  exclude it from final scale compensation.
- Intervening transformations: nonlinearities, normalization, positional or
  attention operations, state-space mixing, residual addition, concatenation,
  gating, and packed QKV/head rearrangements. Matching widths across such a
  boundary do not authorize a fold.
- The source's actual gain convention (`weight` versus `1 + weight`), bias,
  foldable dimensions, and packed/GQA/MQA row layout. A weightless or
  input-dependent norm is not a plain affine gain.
- The scoring boundary, its captured input/kwargs, and why replay evaluates
  the complete computation affected by that one scale.

Inspect conditional modes explicitly: pre/post-layernorm residuals, parallel
decoders, dense/MoE/hybrid layers, shared experts, and relevant head/width
variants. Treat aliases with different layouts as separate cases. Do not infer
execution order from adjacent leaves or architecture names.

## 2. Express only justified groups with existing APIs

Use the existing module tree and ordinary model overrides; do not invent a DSL
or add increasingly broad global name/width guesses to admit a new model.

- Projections reading the same tensor are parallel, not sequential stages.
  Shared-source branches use existing `:in=<tag>` semantics and one merged
  scaling group; fold a source norm **once**, never once per branch/expert.
- Include every non-quantized affine consumer in final compensation, resolving
  the actual weight-bearing child rather than just its container.
- Use `:nosrc` for an output that cannot safely be a fold source. It prevents
  using that producer; it must not remove the linear from quantization.
- If existing replay cannot score every affected branch together, do not split
  the source into independently folded groups or force an unrelated boundary.
  Reject the group or conservatively disable that optimization.
- Validate conflicting sources, overlapping/tied targets, and bindings **before**
  filtering low-sample/missing-feature groups or choosing fallback. Filtering
  one side must not hide an unsafe surviving fold.

Keep `BaseQModel.awq_get_modules_for_scaling`,
`AWQProcessor._validate_scale_group_conflicts`, and
`AWQProcessor._validate_scale_group` consistent. Passing these guards is not a
substitute for the native execution evidence above.

## 3. Preserve candidate coordinates and native replay

The existing search evaluates `Q(W * s) / s` on original-coordinate inputs.
Keep non-quantized routers/shared gates unchanged during candidate replay;
compensate their input columns during the final fold, when the norm output
actually becomes `x / s`. Do not also divide replay input in this simulation
or rescale only one branch of the scoring objective.

Use `awq_module_replay_inputs` for model-specific replay preparation and
`awq_module_replay_chunk_kwargs` for metadata with non-leading batch layouts.
The chunk hook receives **full prepared kwargs**; its returned keys override
generic slicing. Slice by the real layout, not just `tensor.shape[0] == batch`.
For example Bloom ALiBi has `[batch * heads, 1, sequence]` rows; handle one head,
multiple heads, and the final partial chunk without slicing twice.

Reuse native causal/padding masks, positions/rotary data, ALiBi, and required
kwargs. Do not fabricate all-zero additive masks or all-valid padding merely
to satisfy a signature. Reject missing/stale/incompatible context through the
existing unsupported-replay contract; do not swallow arbitrary exceptions.
Any reconstruction must derive from sufficient captured native context and
be independently tested against native execution, including padding/causality.

A candidate-independent additive residual may cancel from reconstruction
error; this does **not** justify a final fold through that residual branch.
Bloom post-layernorm residual mode is the counterexample: disable unsafe
scaling groups unless all affected paths can actually be compensated.

Temporary candidate weights and observation hooks must be restored in
`try/finally`, including reference, quantizer, and candidate-forward failures.

## 4. Required regression evidence

Add model-specific tests in the same PR; do not just rerun generic tests.
Cover each applicable row below, or explain precisely why it is inapplicable.

| Invariant | Required evidence |
| --- | --- |
| Native ownership and grouping | Real model class plus normal conversion, live `state_dict()` owners, paths/tags and aliases/variants per the model-plan skill; compare inferred groups with native forward relationships. |
| Neutral final folds | For every usable group, apply non-unit, non-uniform channel scales **without quantization**, compare full decoder/model outputs with a fresh native baseline, and restore state between groups. A local target-only comparison misses omitted siblings. Assert groups were actually exercised; zero groups is not a preservation pass. |
| Neutral candidate search | With identity quantization, every candidate preserves outputs and router logits/selected experts. Check intermediate candidates, not only the selected scale or final fold. |
| Correct search objective | Compare candidate losses/selected scales with a separately transformed computation using actual quantization and all affected branches. The oracle must not call the search implementation to generate expected values. |
| Complete replay | Verify every declared target executes. Include a negative incomplete-branch/unobserved-expert case and assert configured fallback/error; do not disable observation guards to make MoE tests pass. |
| Context and chunk parity | Compare replay with native forwards and candidate loss/selected scale across chunked and legacy search, unchunked and micro-batched replay, distinct per-sample padding, head layouts and partial final chunks. Perturb future tokens and check earlier causal outputs remain unchanged. |
| Rollback | Inject reference, quantizer and candidate-forward failures; assert exact weight restoration and hook cleanup on success and failure. |
| Unsafe/ambiguous cases | Regress wrong-source width collisions, nonlinear/residual boundaries, conflicting sources, overlapping/tied targets, and filtering that could hide conflicts where relevant to the model/change. |
| Safe degradation | Test permitted fallback and `fallback=None`, plus architecture-level no-group modes. Confirm **all** supported linears still quantize and pack; empty groups must not finalize an unquantized layer. |

Use non-degenerate inputs, non-identity scales, deterministic seeds, native
evaluation mode, and dtype-appropriate measured tolerances. Include FP32 and
supported low-precision replay dtypes where applicable; use the numerical
oracle skill for quantization-math changes. Tiny native configs need no trained
checkpoint download for these invariants. Generation or low perplexity alone
does not replace them; checkpoint smoke tests are complementary evidence.

Useful working references (extend the relevant patterns, not their expected
outputs):

- `tests/module_tree/test_awq_relationships.py`: native full-output neutral
  folds, parallel branches, nonlinear boundaries, nested/shared consumers.
- `tests/test_awq_scale_search_safety.py`: identity candidates, independent
  folded-model objective, rollback, observation, and pre-filter conflicts.
- `tests/test_awq_moe_router_scaling.py`: real MoE routing and compensation.
- `tests/test_bloom_awq_replay.py` and
  `tests/module_tree/test_bloom_awq_relationships.py`: native causal/padding
  context, packed-head chunking, residual variants, and CPU quantization.

Run the new/updated model-specific tests **and** this CPU regression baseline:

```bash
python -m pytest -q \
  tests/test_awq_scale_search_safety.py \
  tests/test_awq_moe_router_scaling.py \
  tests/test_awq_empty_groups.py \
  tests/test_awq_position_kwargs.py \
  tests/test_awq_scale_search_memory.py \
  tests/test_bloom_awq_replay.py \
  tests/module_tree/test_awq_relationships.py \
  tests/module_tree/test_bloom_awq_relationships.py \
  tests/module_tree/test_registered_module_plans.py \
  tests/module_tree/test_definition_subset_ordering.py
```

Run additional affected folding, norm, scaling-group, and math tests as needed.
Record interpreter/dependency versions, exact commands, pass/skip counts, test
nodes, and coverage limits. CPU is sufficient for these structural/native
replay invariants; it does not qualify a GPU kernel. An unavailable modeling
implementation may skip with a precise reason, but registry checks still run
and the skipped variant is **unverified**, not AWQ-qualified.

## 5. Qualification and failure contract

State one outcome per supported layout/config in the PR:

- **AWQ scaling qualified:** source/consumer evidence and applicable tests pass.
- **Plain quantization only:** unsafe groups are deliberately omitted and a
  test proves every supported linear still quantizes, without a norm fold.
- **Unsupported/unverified:** disable/reject the unproven scale optimization
  and document the exact limitation; do not claim a skipped variant is qualified.
  A comment describing uncertainty does not justify emitting guessed groups.

For emitted groups, `_AWQScaleGroupError` uses configured plain-quantization
fallback; `fallback=None` propagates the error. An architecture returning no
scaling groups uses the existing plain-quantization path even with
`fallback=None`. These are different contracts; test the one intended.
Never catch every exception, drop a consumer, weaken a guard, or replace a
failed numerical assertion with a skip to advertise support.

References: [scaling-group contract](../../../docs/awq-scaling-groups.md) and
[Bloom replay safety](../../../docs/awq-bloom-replay.md), with implementation
diagrams. If runtime flow changes, also follow `gptqmodel-logic-flow-diagrams`.
