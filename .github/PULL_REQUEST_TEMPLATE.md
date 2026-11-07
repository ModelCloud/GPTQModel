## Summary

Describe the bug, fix, or feature clearly and briefly.

## What Changed

- List the main code changes.
- List any API or behavior changes.
- List any follow-up work that is intentionally out of scope.

## Tests

Every working PR must include at least one new simple, fast, targeted unit test when the change affects behavior, a bug fix, or a regression path.

- [ ] Unit tests have been executed and passed, or the reason they are not applicable is documented.
- [ ] I added a new simple/fast unit test for this change, or documented why that is not applicable.

Paste the exact test commands and results here:

```bash
```

## Model Support / Module Plan

Required for new model support, model-type aliases, and definition changes.
See `.agents/skills/gptqmodel-model-plan-tests/SKILL.md`.

- [ ] Every new `MODEL_MAP` entry has an explicit registry plan case.
- [ ] A new model-specific module-plan unit test builds the real model class
      and verifies declared paths, checkpoint weight ownership/tags, roles,
      and applicable consumers, including layout variants and method overrides.
- [ ] Runtime-created tensors are distinguished from checkpoint-owned weights.
- [ ] Fusion eligibility is recorded in `fusable_models.json`: a validated
      definition-owned `fusable` config composes role profiles with target-first `merge(...)`, or
      an explicit non-applicability/qualification-gap reason is given.
- [ ] Fusion modules/parents resolve through `model.module_plan()` with no HF
      class-path allowlist; preamble/fallback ordering is tested.
- [ ] Composed fusion configs preserve target identity and source immutability;
      merge conflicts, explicit resolution, scoped policy and printed provenance
      have unit coverage.
- [ ] New fusion qualifications pass real-model planner coverage and native
      GPTQ/AWQ FP16/BF16 forward/oracle tests; skipped tests are not counted as validation.
- [ ] Registry and model-specific tests passed; any dependency skips and their
      precise coverage limits are listed below.

Model test node IDs and results (or "not applicable"):

## Logic Flow Documentation

Required for changes that alter control flow, replace a decision mechanism
(for example name matching replaced by declared metadata), change resolution or
selection order, change data or tensor lifecycle, or introduce a fail-closed
contract. See `$gptqmodel-logic-flow-diagrams` in
`.agents/skills/gptqmodel-logic-flow-diagrams/SKILL.md`.

- [ ] This change does not alter a major logic flow, or the required diagrams
      and docs page are included below.
- [ ] New-flow Mermaid diagram(s) describe the shipped code using real symbol names.
- [ ] The removed/replaced mechanism is shown or tabulated with `file:line` evidence.
- [ ] Failure and ambiguity paths are marked and visually distinct from success.
- [ ] A reference page under `docs/` repeats the diagrams, titled for the
      mechanism, and is linked from `README.md`.

Diagram summary (or "not applicable" with one sentence of justification):

## Review Requirements

AI-assisted code is welcome. The PR must still be reviewed before it is opened as ready for review.

I or an AI agent (such as Codex or Claude) have:

- [ ] Reviewed this PR.
- [ ] Checked that the code matches existing project structure, APIs, and conventions.
- [ ] Avoided unnecessary monkeypatching and used the project's normal extension points where possible.
- [ ] Minimized and compacted the impacted code surface.
- [ ] Considered and eliminated potential regressions.

## Kernel Accuracy Requirements

- [ ] For kernel-related changes, accuracy drift was measured against an applicable independent Torch FP32/FP64 oracle, or the reason this is not applicable is documented.
- [ ] Fusion support changes preserve AUTO's native class and add actual-device
      FP16/BF16 tests where supported, native shape/workspace/descriptor checks,
      mutation, direct source calls, exact restoration and packed-storage checks.
- [ ] The complete kernel inventory/device report lists unavailable dependencies,
      architecture restrictions and unqualified formats; skips are not validation.

## Notes

Add any migration notes, risks, compatibility concerns, or reviewer guidance here.
