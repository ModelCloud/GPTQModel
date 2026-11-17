---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-model-plan-tests
description: Add and validate module-plan unit tests and fusion eligibility when registering a model type or alias, changing a model definition, or changing module-tree consumption in GPT-QModel.
---

# Model support requires module-plan tests

## At a glance

- **What:** every model type and alias has a registry plan case, a model-specific module-plan test, and a fusion-eligibility entry.
- **Why:** `module_tree` is the oracle for which checkpoint weights exist and how each is quantized. A wrong tree silently skips, misgroups, or drops weights.
- **When:** adding to `MODEL_MAP`, changing a definition's paths, tags, variants, or method overrides, or changing a module-plan consumer.
- **Where:** `gptqmodel/models/definitions/`, `gptqmodel/models/module_plan.py`, `tests/module_tree/` (fixtures `registered_module_plans.json`, `fusable_models.json`).
- **How:** build the real model class tiny on CPU or meta, resolve every checkpoint-owned weight through the plan, and run the always-on registry contract.
- **Also:** every new model or definition change needs `gptqmodel-awq-model-onboarding`.

## Registry and model-specific tests

Every new `MODEL_MAP` entry, including an alias, must add an explicit case to
`tests/module_tree/fixtures/registered_module_plans.json` and a model-specific
module-plan unit test under `tests/module_tree/`. Changes to a definition's
paths, tags, variants, or method overrides must update its tests in the same PR.
The always-run registry contract fails for an unlisted model or method case.

A correct weight inventory is necessary but does not qualify AWQ: scale
sources, consumers, and search objectives are proven separately under
`gptqmodel-awq-model-onboarding`.

Use `tests/module_tree/test_registered_module_plans.py` for the fast compiler
and consumer contract. Its inventory pins decoder prefixes and ordered blocks
from the independent quantization subset planner. Review expectation changes
against the architecture; do not blindly refresh snapshots from `ModulePlan`.
Cover each model-type alias, layout variant, and quantization-method override.

For model-specific coverage, build the real model class with a tiny config on
CPU or meta, run the normal Defuser conversion where required, and obtain the
plan through `model.module_plan()` when testing a loaded wrapper. Verify:

- Declared quantizable paths resolve on the real decoder layers, including
  dense/MoE and hybrid attention branches used by the model.
- Declared roles and routed/shared expert families reach the actual consumers,
  including looper grouping and lifecycle hooks where applicable.
- Similar prefixes/suffixes and undeclared descendants are rejected.

Every checkpoint-owned decoder-layer weight, quantized or not, must satisfy
all three:

- Its deepest owning module resolves through `ModulePlan.find()`.
- A quantized owner is declared by `simple_layer_modules()`.
- A non-quantized owner carries an explicit `:!` (not quantized) or `:?`
  (capture-only) tag.

The registry contract runs without an opt-in flag and must not skip supported
definitions.

Use `state_dict()` (parameters and persistent buffers) as the architecture
fixture, not `named_modules()`, `named_parameters()`, or `named_buffers()`
alone. For an actual checkpoint audit, use checkpoint keys and declared
conversion mappings to distinguish checkpoint weights from tensors created
after loading; runtime-only tensors do not create model-definition obligations.
Do not substitute a tree-generated fake model for real model validation.

The Flash-family reference is
`tests/module_tree/test_flash_model_tree_consumption.py`. A missing optional
remote implementation may skip its real-model test with a precise reason,
but must still run the registry contract. Do not blanket-skip exceptions or
report a skipped case as validated. Unsupported registry sentinels such as
original `dbrx` require an explicit rejection test, not positive coverage.

Run the registry contract, the new model-specific test, and affected consumer
tests. Record the exact commands, pass/skip counts, dependency limitations,
and model-test nodes in the PR. New model support is not complete with only an
E2E generation test or a registry entry.

## Fusion eligibility

Every model type/alias also needs an entry in
`tests/module_tree/fixtures/fusable_models.json`: either an explicitly qualified
definition-owned `fusable` config or a precise reason the current executor is
not applicable or remains unqualified. Keep `docs/fusion-model-coverage.md` in
sync. Eligibility never enables fusion by default.

Compose immutable role profiles through `FusableConfig("target").merge(...)`;
merges preserve target identity and source immutability; use `profiles=("qkv",)` for a family subset.
Concrete sibling paths and live parents must resolve from `model.module_plan()`.
Do not add HF parent-class import paths, class allowlists, or name heuristics to
fusion declarations. Inventory class names are independently observed test
facts and never runtime admission rules. Keep native-forward/shared-input
checks and verify preamble work runs once on both fused and fallback paths.
Test multiple-source/diamond deduplication, explicit profile replacement, policy
conflicts, scoped overrides and failure before changing an installed plan.
Printed/extracted decisions must include origins and winning policy. Definitions
may use `model_profiles` for per-model role policies. MoE expert
paths remain excluded by the current planner.

Before adding a qualification, run the real-model topology/planner test and
the native parent tests in `tests/module_tree/test_fusable_models.py`. Cover
GPTQ/AWQ and FP16/BF16 against separate native forwards and independent packed
integer-weight oracles. A skipped native test or a missing implementation is
not a validated profile. Full-model performance claims additionally need
measurements on that architecture; shared parent qualification does not imply
the Llama benchmark numbers apply to it.
