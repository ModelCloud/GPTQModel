# Add bounded native core calibration and exl3 sources

Carry the native BF16 GLM source, explicit attention/MoE stage boundaries, independent raw-Hessian ownership, padded projection quantization, and bounded source reads. This exposes the existing runner API without changing the default loader.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["GlmBF16Source"]
    M0_0["GlmBF16Source.__init__"]
    M0_1["GlmBF16Source.materialize_block"]
    M0_2["GlmBF16Source.load_block"]
    S1["quantize_projection"]
    S2["prepare_context"]
    S3["attention_forward"]
    S4["complete_moe"]
    E["ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -. "exposes" .-> M0_2
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    S2 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_core.py:98`: `GlmBF16Source` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_core.py:31`: `quantize_projection` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_core.py:253`: `prepare_context` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_core.py:290`: `attention_forward` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
