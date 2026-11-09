# Carry bounded v4 target and mtp calibration replay

Carry the existing V4 floatx/router precision handling, explicit auxiliary traversal, causal target/MTP alignment, bounded prefix runtime, checkpoint mappings, route recovery, vision/native tensor retention, and integrated quantized MTP save behavior.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["DeepSeekV4QModel"]
    M0_0["DeepSeekV4QModel.prepare_input_capture_layer"]
    M0_1["DeepSeekV4QModel.save_state_overlay"]
    M0_2["DeepSeekV4QModel.capture_first_layer_input_kwargs"]
    S1["DeepSeekV4MTPReplay"]
    M1_0["DeepSeekV4MTPReplay.__init__"]
    M1_1["DeepSeekV4MTPReplay.prepare_batch"]
    S2["DeepSeekV4MTPQuantizationModel"]
    M2_0["DeepSeekV4MTPQuantizationModel.prepare_input_capture_layer"]
    M2_1["DeepSeekV4MTPQuantizationModel.prepare_dataset"]
    E["IndexError, RuntimeError, TypeError, ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -. "exposes" .-> M0_2
    S0 -->|"raises on rejected input/state"| E
    S1 -. "exposes" .-> M1_0
    S1 -. "exposes" .-> M1_1
    S1 -->|"raises on rejected input/state"| E
    S2 -. "exposes" .-> M2_0
    S2 -. "exposes" .-> M2_1
    S2 -->|"calls where selected"| S1
    S2 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| Previous stack layer: `gptqmodel/models/definitions/deepseek_v4.py:9` | `gptqmodel/models/definitions/deepseek_v4.py:1410`: `DeepSeekV4QModel` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/models/definitions/deepseek_v4.py:740`: `DeepSeekV4MTPReplay` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/models/definitions/deepseek_v4.py:2349`: `DeepSeekV4MTPQuantizationModel` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
