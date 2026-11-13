# Integrate resumable distributed and mixed-tier execution

Wire the preceding stores and policies into EXL3Processor: natural-route capture, batch/frontier restore, deterministic recovery, projection checkpoint hits, remote slots, inline tier upgrades, bounded runtime replay, and completed checkpoint-tree publication. This is the larger integration layer; the preceding feature APIs remain individually reviewable.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["EXL3Processor"]
    M0_0["EXL3Processor.__init__"]
    M0_1["EXL3Processor.restore_subset_capture_batches"]
    M0_2["EXL3Processor.forward_committed_batch_indices"]
    S1["_EXL3NaturalRouteCapture"]
    M1_0["_EXL3NaturalRouteCapture.__init__"]
    M1_1["_EXL3NaturalRouteCapture.__enter__"]
    M1_2["_EXL3NaturalRouteCapture.capture_expert_input"]
    S2["clone_exllamav3_config_for_module"]
    E["NotImplementedError, RuntimeError, ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -. "exposes" .-> M0_2
    S0 -->|"calls where selected"| S1
    S0 -->|"calls where selected"| S2
    S0 -->|"raises on rejected input/state"| E
    S1 -. "exposes" .-> M1_0
    S1 -. "exposes" .-> M1_1
    S1 -. "exposes" .-> M1_2
    S1 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| Previous stack layer: `gptqmodel/looper/exllamav3_processor.py:91` | `gptqmodel/looper/exllamav3_processor.py:704`: `EXL3Processor` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/looper/exllamav3_processor.py:268`: `_EXL3NaturalRouteCapture` |
| Previous stack layer: `gptqmodel/looper/exllamav3_processor.py:57` | `gptqmodel/looper/exllamav3_processor.py:668`: `clone_exllamav3_config_for_module` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
