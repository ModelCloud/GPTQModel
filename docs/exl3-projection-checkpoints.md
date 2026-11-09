# Checkpoint completed projections atomically

Expose content-bound projection requests, atomic safetensors publication, validated restore, and readable checkpoint artifacts. Keep checkpoint identity and completion boundaries unchanged.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["build_projection_request"]
    S1["EXL3ProjectionCheckpointStore"]
    M1_0["EXL3ProjectionCheckpointStore.__init__"]
    M1_1["EXL3ProjectionCheckpointStore.committed_tensor_path"]
    M1_2["EXL3ProjectionCheckpointStore.load_committed"]
    E["RuntimeError, ValueError"]
    S1 -. "exposes" .-> M1_0
    S1 -. "exposes" .-> M1_1
    S1 -. "exposes" .-> M1_2
    S1 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_projection_checkpoint.py:89`: `build_projection_request` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_projection_checkpoint.py:179`: `EXL3ProjectionCheckpointStore` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
