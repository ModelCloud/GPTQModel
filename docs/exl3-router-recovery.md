# Expose ranked router candidates for route recovery

Preserve native live top-k choices at tied boundaries while exposing ranked softmax and grouped-sigmoid candidates for bounded recovery. Model bindings are introduced with the model integrations.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["learned_router_ranked_choices"]
    S1["bind_softmax_router_recovery"]
    S2["bind_sigmoid_grouped_router_recovery"]
    E["RuntimeError, ValueError"]
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    S2 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_router_candidates.py:79`: `learned_router_ranked_choices` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_router_candidates.py:225`: `bind_softmax_router_recovery` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_router_candidates.py:238`: `bind_sigmoid_grouped_router_recovery` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
