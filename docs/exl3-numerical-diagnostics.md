# Preserve hessian conditioning and projection diagnostics

Carry the fork quantizer changes for FP64 Hessian congruence, symmetry restoration, per-projection RNG isolation, and structured reconstruction/Hessian error metrics. Preserve the existing arithmetic and failure evidence rather than redesigning it.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    S0["regularize_and_transform_hessian_"]
    S1["restore_hessian_symmetry_"]
    S2["quantize_exl3"]
    S3["reconstruction_error_metrics"]
    E["RuntimeError, ValueError"]
    S0 -->|"calls where selected"| S1
    S0 -->|"raises on rejected input/state"| E
    S1 -->|"raises on rejected input/state"| E
    S3 -->|"raises on rejected input/state"| E
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py:688`: `regularize_and_transform_hessian_` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py:666`: `restore_hessian_symmetry_` |
| Previous stack layer: `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py:1014` | `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py:1266`: `quantize_exl3` |
| API absent in the previous stack layer (path/symbol checked) | `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py:1053`: `reconstruction_error_metrics` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.
