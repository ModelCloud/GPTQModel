# Carry native full-mtp alignment and source loading

Carry teacher-forced token/frontier alignment, the native MTP block and RMSNorm, optional expert loading, and shared embedding/head behavior through the explicit source adapter.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["aligned_inputs"]
    S1["GlmMTP"]
    M1_0["GlmMTP.__init__"]
    M1_1["GlmMTP.forward_aligned"]
    M1_2["GlmMTP.forward"]
    S2["load_mtp"]
    E["ValueError"]
    S0 -->|"raises on rejected input/state"| E
    S1 -. "exposes" .-> M1_0
    S1 -. "exposes" .-> M1_1
    S1 -. "exposes" .-> M1_2
    S1 -->|"calls where selected"| S0
    S1 -->|"raises on rejected input/state"| E
    S2 -->|"calls where selected"| S1
    S2 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_mtp.py:57`: `aligned_inputs` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_mtp.py:120`: `GlmMTP` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_mtp.py:167`: `load_mtp` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
