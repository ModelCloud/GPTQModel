# Persist authenticated capture frontiers

Preserve durable capture descriptors, tensor/route evidence, bounded frontier hydration, and namespace-scoped cleanup as a standalone store for the following looper integration.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["EXL3CaptureDescriptor"]
    S1["EXL3CaptureFrontierStore"]
    M1_0["EXL3CaptureFrontierStore.__init__"]
    M1_1["EXL3CaptureFrontierStore.restore_index"]
    M1_2["EXL3CaptureFrontierStore.load_record_hessian"]
    E["EXL3CaptureFrontierError"]
    S1 -. "exposes" .-> M1_0
    S1 -. "exposes" .-> M1_1
    S1 -. "exposes" .-> M1_2
    S1 -->|"calls where selected"| S0
    S1 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_capture_frontier.py:56`: `EXL3CaptureDescriptor` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_capture_frontier.py:142`: `EXL3CaptureFrontierStore` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
