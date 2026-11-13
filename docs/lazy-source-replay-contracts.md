# Preserve lazy native sources across replay and export

Carry the model extension hooks, floatx source selection/reload, source tensor mappings, active-shard staging, concrete device materialization, safe replay cloning, and source config selectors used by the native model runners. Keep Dots3/Kimi registration in their later model layers.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["_uses_auto_module_decoder"]
    S1["_convert_model_with_defuser"]
    S2["LazyTurtle"]
    M2_0["LazyTurtle.__init__"]
    M2_1["LazyTurtle.materialize_submodule"]
    M2_2["LazyTurtle.materialize_direct_meta_tensors"]
    S3["BaseQModel"]
    M3_0["BaseQModel.__init__"]
    M3_1["BaseQModel.prepare_dataset"]
    M3_2["BaseQModel.forward"]
    E["AttributeError, EnvironmentError, NotImplementedError, RuntimeError, TypeError, ValueError"]
    S2 -. "exposes" .-> M2_0
    S2 -. "exposes" .-> M2_1
    S2 -. "exposes" .-> M2_2
    S2 -->|"raises on rejected input/state"| E
    S3 -. "exposes" .-> M3_0
    S3 -. "exposes" .-> M3_1
    S3 -. "exposes" .-> M3_2
    S3 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/models/auto.py:538`: `_uses_auto_module_decoder` |
| Previous stack layer: `gptqmodel/models/loader.py:535` | `gptqmodel/models/loader.py:535`: `_convert_model_with_defuser` |
| Previous stack layer: `gptqmodel/utils/structure.py:845` | `gptqmodel/utils/structure.py:846`: `LazyTurtle` |
| Previous stack layer: `gptqmodel/models/base.py:312` | `gptqmodel/models/base.py:312`: `BaseQModel` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
