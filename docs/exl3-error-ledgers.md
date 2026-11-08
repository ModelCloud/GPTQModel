# Retain restart-idempotent projection and route ledgers

Expose projection/family error records, natural-route evidence, provenance compaction, atomic ledger publication, and restart-idempotent journals. Processor and writer wiring follows in later layers.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["build_projection_record"]
    S1["append_exl3_error_journal"]
    S2["write_exl3_error_ledger"]
    E["OSError, ValueError"]
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_error_ledger.py:642`: `build_projection_record` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_error_ledger.py:929`: `append_exl3_error_journal` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/exl3_error_ledger.py:995`: `write_exl3_error_ledger` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
