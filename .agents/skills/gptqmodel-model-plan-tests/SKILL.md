---
name: gptqmodel-model-plan-tests
description: Add and validate module-plan unit tests when registering a model type, changing a model definition, or changing module-tree consumption in GPTQModel.
---

# Model support requires module-plan tests

Every new `MODEL_MAP` entry, including an alias, must add an explicit case to
`tests/module_tree/fixtures/registered_module_plans.json` and a model-specific
module-plan unit test under `tests/module_tree/`. Changes to a definition's
paths, tags, variants, or method overrides must update its tests in the same PR.
The always-run registry contract fails for an unlisted model or method case.

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
- Checkpoint-owned weight owners resolve through `plan.find()` and carry
  quantization, exclusion, or capture tags. Include non-quantized weights.
- Declared roles and routed/shared expert families reach the actual consumers,
  including looper grouping and lifecycle hooks where applicable.
- Similar prefixes/suffixes and undeclared descendants are rejected.

`state_dict()` is a useful architecture fixture for parameters and persistent
buffers. For an actual checkpoint audit, use checkpoint keys and declared
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
