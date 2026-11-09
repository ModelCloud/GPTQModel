# Dispatch authenticated projections to pipelined workers

Carry authenticated tensor transport, dynamic coordinator/remote slot scheduling, projection checkpoint reuse, bounded scratch retention, per-device serialization, and remote Hessian failure diagnostics. This preserves the existing explicit NVIDIA execution contract; it does not newly qualify ROCm.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["encode_tensor_envelope"]
    S1["decode_tensor_envelope"]
    S2["execute_remote_projection"]
    S3["EXL3RemoteClient"]
    M3_0["EXL3RemoteClient.__init__"]
    E["AssertionError, RuntimeError, ValueError"]
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    S2 -->|"raises on rejected input/state"| E
    S3 -. "exposes" .-> M3_0
    S3 -->|"calls where selected"| S0
    S3 -->|"calls where selected"| S1
    S3 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_remote.py:243`: `encode_tensor_envelope` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_remote.py:266`: `decode_tensor_envelope` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_remote.py:473`: `execute_remote_projection` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_remote.py:627`: `EXL3RemoteClient` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
