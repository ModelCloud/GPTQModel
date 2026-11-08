# Carry bounded native core and mtp calibration

Carry native Step-5 source/reference loading, bounded fused-expert reads, actual-row covariance products, functional indexer W input observation without extra GEMMs, explicit stage replay, and MTP alignment. Preserve the reviewed source config instead of mutating its document.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["Step5BF16Source"]
    M0_0["Step5BF16Source.__init__"]
    M0_1["Step5BF16Source.materialize_block"]
    M0_2["Step5BF16Source.load_block"]
    S1["capture_inputs"]
    S2["staged_forward"]
    S3["load_mtp"]
    E["ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -. "exposes" .-> M0_2
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    S3 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/step5_core.py:148`: `Step5BF16Source` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/step5_core.py:380`: `capture_inputs` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/step5_core.py:372`: `staged_forward` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/step5_core.py:498`: `load_mtp` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
