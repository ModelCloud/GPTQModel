# Carry flash core and mtp calibration adapters

Carry hybrid KDA/sparse-attention stage boundaries, bounded KDA execution, hyperconnection continuation, original native checkpoint names, and aligned Flash MTP source loading.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["GlmFlashBF16Source"]
    M0_0["GlmFlashBF16Source.__init__"]
    M0_1["GlmFlashBF16Source.materialize_block"]
    M0_2["GlmFlashBF16Source.materialize_routed_experts"]
    S1["bounded_kda"]
    S2["staged_forward"]
    S3["GlmFlashMTP"]
    M3_0["GlmFlashMTP.__init__"]
    M3_1["GlmFlashMTP.forward_aligned"]
    E["ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -. "exposes" .-> M0_2
    S0 -->|"raises on rejected input/state"| E
    S3 -. "exposes" .-> M3_0
    S3 -. "exposes" .-> M3_1
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_flash_core.py:76`: `GlmFlashBF16Source` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_flash_core.py:49`: `bounded_kda` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_flash_core.py:267`: `staged_forward` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/glm5_flash_mtp.py:70`: `GlmFlashMTP` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
