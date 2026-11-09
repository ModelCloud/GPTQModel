# Integrate flash target and mtp recovery and export

Carry the existing GLM-5 Next and GLM MoE DSA model bindings, original token-ID capture, bounded replay frontiers, transformed mask/position handling, native expert conversion, and target/MTP route recovery. Keep upstream module_tree verbatim, including role metadata and dense low-rank/shared-expert policy. The native expert container and activation hook stay owned by Transformers/Defuser. The fork-only converter and expert forward implementation are removed; coverage validation uses the existing after_defuser_conversion hook.

This page describes the shipped API and conditional calls carried from the working fork. It is a structural call map, with no invented execution ordering between public entry points. Solid edges name calls present in the source; dashed edges expose class entry points. Rejected states and propagated failures are marked separately.

```mermaid
flowchart TD
    L["Glm5NextQuantizationLoader.from_config / from_pretrained"] --> A["_attach_glm5_next_mtp"]
    A --> M["Glm5NextMTPDecoderLayer: checkpoint-only layer 45"]
    M --> C["_convert_model_with_defuser: existing upstream flow"]
    C --> D["defuser.convert_model: native Glm5NextTextExperts"]
    D --> V["Glm5NextQModel.after_defuser_conversion"]
    V --> Q{"Target / MTP expert coverage valid?"}
    Q -->|"yes"| N["Native experts, activation and upstream module_tree retained"]
    Q -->|"no"| E["Raise: incomplete expert conversion"]
    N --> R["Glm5NextQModel.after_model_load"]
    R --> B["bind_sigmoid_grouped_router_recovery"]
    style E fill:#fee,stroke:#b22
```

| Before this extraction layer | Carried implementation |
|---|---|
| Previous stack layer: `gptqmodel/models/definitions/glm5_next.py:13` | `gptqmodel/models/definitions/glm5_next.py:254`: `Glm5NextQModel` |
| Previous stack layer: `gptqmodel/models/definitions/glm_moe_dsa.py:11` | `gptqmodel/models/definitions/glm_moe_dsa.py:19`: `GlmMoeDsaQModel` |

The previous stack layer supplies dependencies. Standalone store/policy APIs are connected by the later processor, looper and model layers rather than being enabled implicitly here.

Validation for this extraction is limited to Python parsing, dependency inspection and whitespace checks. Existing regression/oracle tests are carried with the feature; no new model quantization, accelerator work, hardware benchmark or broad runtime revalidation was performed. The author reports successful prior use in the fork. This draft seeks initial feedback on value and scope before further refactoring or validation.

This is an extension of existing GLM support. The upstream module tree, role metadata, dense shared experts, processor-loading policy and Defuser dispatch remain authoritative. The added loader exposes MTP layer 45; the added post-conversion hook checks that ordinary Defuser conversion covered both target and MTP experts.

Two tiny CPU integration checks passed in 6.47 seconds: native expert object/forward preservation and lazy source materialization. No model quantization was run. The previous custom converter/container is deliberately retained only in the fork remainder.
