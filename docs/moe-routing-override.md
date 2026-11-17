# MoE routing override and bypass contract

This page describes how `MoEConfig` routing modes reach a model's router and
expert lifecycle, and why an unsupported request must fail closed instead of
falling back to native routing.

Introduced by PR for issue #4636.

## 1. The two routing modes

`MoEConfig` selects one of three behaviors:

| Mode | `MoEConfig.routing` | Calibration behavior |
| --- | --- | --- |
| native | `None` | The model's own router selects experts per token. |
| override | `ExpertsRoutingOverride(...)` | Temporarily set the router's top-k so each token reaches that many experts. |
| bypass | `ExpertsRoutingBypass(...)` | Feed the MoE block input to every expert, then run the model's routed forward for the output. |

`ModuleLooper` resolves the mode once in `__init__`
(`gptqmodel/looper/module_looper.py:431`) and `ForwardExecutor` applies it per
forward. `AWQProcessor` re-enters the same mode for its scale-search replay.

## 2. Old flow: two silent fallbacks

```mermaid
flowchart TD
    A["MoEConfig routing"] --> B{"mode"}
    B -- override --> C["MoETopKState = set_moe_topk(module, top_k)"]
    C --> D["PhimoeTopKRouter.top_k is written"]
    D --> E["sparsemixer(..., top_k)"]
    E --> F["sparsemixer ignores top_k and selects 2 experts"]
    F --> G["calibration silently uses native top-2"]
    B -- bypass --> H["_should_use_moe_lifecycle"]
    H --> I{"moe_lifecycle_hooks present"}
    I -- no --> J["WARN once per forward"]
    J --> K["native routed forward"]
    K --> G
    I -- yes --> L["feed every expert"]
    style F fill:#f8d7da,stroke:#b02a37,color:#842029
    style G fill:#f8d7da,stroke:#b02a37,color:#842029
    style J fill:#f8d7da,stroke:#b02a37,color:#842029
```

The saved config still recorded the requested mode, so the checkpoint looked
like it had been calibrated with override or bypass when it had not.

## 3. New flow: honoured or rejected

```mermaid
flowchart TD
    A["MoEConfig routing"] --> B{"mode"}
    B -- override --> C["set_moe_topk(module, top_k)"]
    C --> D{"router class ignores top_k"}
    D -- yes --> E["install softmax top-k forward"]
    D -- no --> F["router reads its top_k field"]
    E --> G["selected width = min(top_k, local experts)"]
    F --> G
    B -- bypass --> H["validate_moe_routing_support"]
    H --> I{"moe_lifecycle_hooks present"}
    I -- no --> J["ValueError before the first forward"]
    I -- yes --> K["MoELifecycleContext feeds every expert"]
    K --> L["routed forward produces the output"]
    style J fill:#f8d7da,stroke:#b02a37,color:#842029
    style G fill:#d1e7dd,stroke:#0f5132,color:#0f5132
    style L fill:#d1e7dd,stroke:#0f5132,color:#0f5132
```

The override installer is `_install_phimoe_topk_override`
(`gptqmodel/utils/model.py:4064`). It keeps the native `sparsemixer` path while
`router.top_k` equals the model's configured value, and only replaces the
forward when the override changes it. `restore_moe_topk` restores the original
value, so the next forward uses the native path again.

The bypass gate is `validate_moe_routing_support`
(`gptqmodel/looper/module_looper.py:169`), called from `ModuleLooper.__init__`.
`_should_use_moe_lifecycle` keeps the same guard for a config mutated after
construction.

## 4. Before and after

| Situation | Before | After |
| --- | --- | --- |
| `ExpertsRoutingOverride` on PhiMoE | `set_moe_topk` wrote `top_k`; `sparsemixer` kept selecting 2 experts; no warning | PhiMoE router forward selects `min(top_k, num_experts)` experts; restore returns to native top-2 |
| `ExpertsRoutingBypass` on PhiMoE | No `moe_lifecycle_hooks`; one WARN per forward; native top-2 calibration | `GateUpDownMoELifecycleHooks` declared; every defused expert receives the block input |
| `ExpertsRoutingBypass` on a definition without hooks | One WARN per forward; native calibration; config still recorded bypass | `ValueError` in `ModuleLooper.__init__` before calibration |
| Router that ignores `top_k` | Silent no-op | Model-specific installer or explicit rejection |

## 5. Definitions that now reject bypass

These definitions declare `dynamic_expert_index` but no
`moe_lifecycle_hooks`. Their override path is unchanged because their routers
honour `top_k`; their bypass path now raises the `ModuleLooper` error instead of
falling back:

`gpt_oss`, `granitemoe`, `granitemoehybrid`, `ernie4_5_moe`,
`ernie4_5_vl_moe`, `gemma4`, `nemotron_h`, `nemotron_h_puzzle`,
`dbrx_converted`, `diffusion_gemma`, `kimi_k3`, `deepseek_v4`, `deepseek_v41`.

PhiMoE is implemented rather than rejected: its defused experts expose
`gate_proj`/`up_proj`/`down_proj` with SiLU, so `GateUpDownMoELifecycleHooks`
can feed every expert.

## 6. Failure and ambiguity contract

| Request | Result |
| --- | --- |
| override on a router that honours `top_k` | router selects the requested width |
| override on PhiMoE | router selects the requested width through the installed forward |
| override on an unsupported router | model-specific installer, otherwise the router's own behavior is used |
| bypass with `moe_lifecycle_hooks` | every expert sees the block input |
| bypass without `moe_lifecycle_hooks` | `ValueError` before the first forward |
| dense model with an explicit bypass | `ValueError`; the option cannot be honoured |

## 7. Not in scope

Implementing lifecycle hooks for the thirteen definitions listed above is not
part of this change. They fail closed until a definition supplies tested
`moe_lifecycle_hooks`.
