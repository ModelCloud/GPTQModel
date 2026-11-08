# Carry causal core calibration and padded exl3 source storage

Carry explicit attention/MoE/head calibration boundaries, bounded source-shard ownership, padded core projection storage validation and reconstruction, and calibrated EXL3 source replacement. Keep the existing independent format and CUDA oracle tests as carried evidence.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["quantize_core_projection"]
    S1["reconstruct_core_projection"]
    S2["KimiK3EXL3Source"]
    M2_0["KimiK3EXL3Source.load_block"]
    M2_1["KimiK3EXL3Source.materialize_routed_experts"]
    S3["capture_linear_input"]
    E["KeyError, ValueError"]
    S0 -->|"raises on rejected input/state"| E
    S2 -. "exposes" .-> M2_0
    S2 -. "exposes" .-> M2_1
    S2 -->|"calls where selected"| S1
    S2 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k3_core_format.py:64`: `quantize_core_projection` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k3_core_format.py:56`: `reconstruct_core_projection` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k3_core_format.py:126`: `KimiK3EXL3Source` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k3_core_stream.py:87`: `capture_linear_input` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
