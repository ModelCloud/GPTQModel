# Carry k3.1 source and bounded calibration adapters

Carry the K3.1 source adapter, native KDA A_log padding validation, bounded FP64-sum Hessian capture, functional indexer input handling, and owned routed source reconstruction.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["KimiK31Source"]
    M0_0["KimiK31Source.__init__"]
    M0_1["KimiK31Source.load_block"]
    S1["capture_routed_hessian"]
    S2["DenseRawHessian"]
    M2_0["DenseRawHessian.__init__"]
    E["ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    S2 -. "exposes" .-> M2_0
    S2 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k31.py:38`: `KimiK31Source` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k31.py:112`: `capture_routed_hessian` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k31.py:92`: `DenseRawHessian` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
