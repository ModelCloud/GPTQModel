# Add bounded k3 source and explicit attnres model contracts

Register Kimi K3 and Kimi Linear routed-expert definitions and carry the bounded native source adapter, MXFP4 decode, AttnRes prefix/residual boundaries, and checkpoint reference loading. Ordinary looper entry points continue to reject unsupported AttnRes continuation explicitly.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["KimiK3Source"]
    M0_0["KimiK3Source.__init__"]
    M0_1["KimiK3Source.load_block"]
    S1["prepare_moe"]
    S2["finish_moe"]
    S3["KimiLinearQModel"]
    M3_0["KimiLinearQModel.quantize"]
    M3_1["KimiLinearQModel.pre_quantize"]
    S4["KimiK3QModel"]
    E["KeyError, NotImplementedError, ValueError"]
    S0 -. "exposes" .-> M0_0
    S0 -. "exposes" .-> M0_1
    S0 -->|"raises on rejected input/state"| E
    S3 -. "exposes" .-> M3_0
    S3 -. "exposes" .-> M3_1
    S3 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k3.py:82`: `KimiK3Source` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k3.py:138`: `prepare_moe` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/utils/kimi_k3.py:160`: `finish_moe` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/models/definitions/kimi_k3.py:18`: `KimiLinearQModel` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
