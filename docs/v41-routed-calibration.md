# Carry v4.1 routed capture and adjacent recovery

Carry bounded expert Hessian and natural-route capture, validated routed calibration batches, adjacent-router recovery, phase-specific frontier reuse, and packed selected-projection replay.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["V41Capture"]
    M0_0["V41Capture.__init__"]
    M0_1["V41Capture.__enter__"]
    M0_2["V41Capture.capture_routed"]
    S1["V41Recovery"]
    M1_0["V41Recovery.__init__"]
    M1_1["V41Recovery.__enter__"]
    S2["V41RoutedBatch"]
    M2_0["V41RoutedBatch.validate"]
    S3["install_projection"]
    E["Captured, RuntimeError, ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -. "exposes" .-> M0_2
    S0 -->|"raises on rejected input/state"| E
    S1 -. "exposes" .-> M1_0
    S1 -. "exposes" .-> M1_1
    S1 -->|"raises on rejected input/state"| E
    S2 -. "exposes" .-> M2_0
    S2 -->|"raises on rejected input/state"| E
    S3 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_capture.py:11`: `V41Capture` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_recovery.py:15`: `V41Recovery` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_routed_batch.py:10`: `V41RoutedBatch` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_mixed_replay.py:10`: `install_projection` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
