# Carry v4.1 native source and mapped ple replay

Carry the V4.1-specific architecture contract, packed source execution, one-tensor source reads, memory-mapped PLE gathers, native attention/index selection, RMS normalization and explicit mHC/window replay state. This family remains an explicit calibration API, without a new automatic MODEL_MAP registration.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["MappedPLETable"]
    M0_0["MappedPLETable.__init__"]
    M0_1["MappedPLETable.__enter__"]
    S1["V41Source"]
    M1_0["V41Source.__init__"]
    M1_1["V41Source.load_dspark_input"]
    M1_2["V41Source.load_main_input"]
    S2["V41NativeLinear"]
    M2_0["V41NativeLinear.__init__"]
    M2_1["V41NativeLinear.forward"]
    S3["V41ReplayBatch"]
    M3_0["V41ReplayBatch.prepare"]
    E["Captured, IndexError, RuntimeError, ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -->|"raises on rejected input/state"| E
    S1 -. "exposes" .-> M1_0
    S1 -. "exposes" .-> M1_1
    S1 -. "exposes" .-> M1_2
    S1 -->|"calls where selected"| S2
    S1 -->|"raises on rejected input/state"| E
    S2 -. "exposes" .-> M2_0
    S2 -. "exposes" .-> M2_1
    S2 -->|"raises on rejected input/state"| E
    S3 -. "exposes" .-> M3_0
    S3 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/ple_mmap.py:15`: `MappedPLETable` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_source.py:45`: `V41Source` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_native.py:212`: `V41NativeLinear` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_replay.py:26`: `V41ReplayBatch` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
