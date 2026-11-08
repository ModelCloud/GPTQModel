# Carry v4.1 main-input and dspark replay adapters

Carry native text input preparation without decoder allocation, grouped teacher-forced draft anchors, shared main-prefix projection, dSpark continuation and explicit joint-ring state.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["V41DSparkInput"]
    M0_0["V41DSparkInput.__init__"]
    M0_1["V41DSparkInput.forward"]
    M0_2["V41DSparkInput.prepare_joint"]
    S1["V41MainInput"]
    M1_0["V41MainInput.__init__"]
    M1_1["V41MainInput.prepare"]
    E["ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -. "exposes" .-> M0_2
    S0 -->|"raises on rejected input/state"| E
    S1 -. "exposes" .-> M1_0
    S1 -. "exposes" .-> M1_1
    S1 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_dspark.py:9`: `V41DSparkInput` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_inputs.py:9`: `V41MainInput` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
