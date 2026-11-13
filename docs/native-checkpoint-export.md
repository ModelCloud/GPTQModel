# Compose native checkpoint passthrough and save overlays

Preserve source-native checkpoint names and untouched tensors, compose validated auxiliary overlays, stream exact passthrough exports, reject incomplete publication trees, and publish EXL3 ledgers without embedding the tensor manifest in model config. Retain early auxiliary-source validation and the earlier atomic shard writer.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["_validated_save_state_overlay"]
    S1["_exllamav3_checkpoint_passthrough_plan"]
    S2["_build_exllamav3_checkpoint_passthrough_state_dict"]
    S3["ModelWriter"]
    E["RuntimeError, TypeError, ValueError"]
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    S2 -->|"raises on rejected input/state"| E
    S3 -->|"calls where selected"| S0
    S3 -->|"calls where selected"| S1
    S3 -->|"calls where selected"| S2
    S3 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/models/writer.py:658`: `_validated_save_state_overlay` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/models/writer.py:792`: `_exllamav3_checkpoint_passthrough_plan` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/models/writer.py:936`: `_build_exllamav3_checkpoint_passthrough_state_dict` |
| Previous stack layer: `gptqmodel/models/writer.py:799` | `gptqmodel/models/writer.py:1387`: `ModelWriter` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
