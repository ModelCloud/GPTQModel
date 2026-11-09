# Resume capture at calibration-batch boundaries

Carry the existing batch spool with provenance and tensor-geometry validation, atomic batch commits, and exact committed-batch recovery. Looper callbacks follow in the replay layer.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["EXL3CaptureBatchSpool"]
    M0_0["EXL3CaptureBatchSpool.__init__"]
    M0_1["EXL3CaptureBatchSpool.committed_indices"]
    M0_2["EXL3CaptureBatchSpool.commit"]
    S1["canonical_json_bytes"]
    E["EXL3CaptureBatchSpoolError, KeyError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -. "exposes" .-> M0_2
    S0 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_capture_batch_spool.py:101`: `EXL3CaptureBatchSpool` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_capture_batch_spool.py:44`: `canonical_json_bytes` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
