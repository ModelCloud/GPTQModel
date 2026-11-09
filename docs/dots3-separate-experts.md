# Expose dots3 note routed experts individually

Register Dots3 Note with separate per-expert checkpoint names, native top-k scatter, auxiliary tensor passthrough, and grouped-router recovery. Preserve the existing model adapter; do not refactor the expert implementation in this extraction.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["Dots3NoteQModel"]
    M0_0["Dots3NoteQModel.before_model_load"]
    M0_1["Dots3NoteQModel.after_model_load"]
    M0_2["Dots3NoteQModel.zero_route_recovery_context"]
    S1["_SeparateDots3Experts"]
    M1_0["_SeparateDots3Experts.__init__"]
    M1_1["_SeparateDots3Experts.forward"]
    E["IndexError, RuntimeError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -. "exposes" .-> M0_2
    S0 -->|"raises on rejected input/state"| E
    S1 -. "exposes" .-> M1_0
    S1 -. "exposes" .-> M1_1
    S1 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/models/definitions/dots3_note.py:44`: `Dots3NoteQModel` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/models/definitions/dots3_note.py:13`: `_SeparateDots3Experts` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
