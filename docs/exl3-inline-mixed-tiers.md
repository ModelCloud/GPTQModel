# Persist deterministic adjacent mixed-tier plans

Expose validated inline mixed-tier policies, projection scoring, deterministic per-layer tier selection, and durable readable plan publication. Execution and requantization wiring follows in the processor layer.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["inline_mixed_policy"]
    S1["build_layer_tier_plan"]
    S2["InlineMixedTierPlanStore"]
    M2_0["InlineMixedTierPlanStore.__init__"]
    M2_1["InlineMixedTierPlanStore.load"]
    M2_2["InlineMixedTierPlanStore.commit"]
    E["ValueError"]
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    S2 -. "exposes" .-> M2_0
    S2 -. "exposes" .-> M2_1
    S2 -. "exposes" .-> M2_2
    S2 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_inline_mixed.py:177`: `inline_mixed_policy` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_inline_mixed.py:298`: `build_layer_tier_plan` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_inline_mixed.py:373`: `InlineMixedTierPlanStore` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
