# Persist authenticated typed v4.1 frontiers

Carry atomic content-bound V4.1 frontier and routed-batch storage, typed tree encoding, exact provenance validation, and nested quantizer metric preservation.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["save_frontier"]
    S1["load_frontier"]
    S2["save_routed_batch"]
    S3["load_routed_batch"]
    E["ValueError"]
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    S2 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_checkpoint.py:131`: `save_frontier` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_checkpoint.py:138`: `load_frontier` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_checkpoint.py:152`: `save_routed_batch` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/v41_checkpoint.py:161`: `load_routed_batch` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
