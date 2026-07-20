---
name: gptqmodel-model-support
description: Add, review, or debug GPT-QModel support for a transformer model family, including architecture diagnosis, explicit module trees, attention/MLP grouping, MoE expert expansion and lifecycle hooks, model registration, quantization compatibility, save/load, and evaluation tests.
---

# GPT-QModel model support

Diagnose the instantiated model architecture before writing an adapter. Model names and config inheritance are hints; the runtime module tree and forward order are the contract.

Read [references/model-adapter-map.md](references/model-adapter-map.md) before adding a definition.

## Diagnose first

1. Record the model's `model_type`, architecture class, Transformers version, dtype, trust-remote-code requirement, and relevant nested configs.
2. Inspect a tiny or meta-initialized instance. Locate embeddings, repeating decoder layers, attention projections, MLP projections, normalization, final norm, and LM head.
3. Read the layer's forward method to capture execution order and shared inputs. For MoE, identify router, expert count/config path, expert projections, shared experts, and execution order.
4. Find the closest definition in `gptqmodel/models/definitions/`. Reuse an existing class only when paths, grouping, flags, and lifecycle behavior are truly identical.

Do not rely on automatic module-tree detection as the sole support path for a model intended to work across GPTQ and AWQ-family algorithms. Define the structure explicitly when robust support matters.

## Implement the adapter

1. Add a focused class derived from `BaseQModel` under `gptqmodel/models/definitions/`.
2. Define `module_tree` in actual forward order. Group projections that share an input/statistics boundary, and split projections that consume a later activation.
3. Mark inference-only and capture-only nodes correctly. Do not quantize norms, routers, or other control modules merely because they contain weights.
4. For MoE models, set `dynamic_expert_index` to the config field that holds the expert count, use nested `#` expert placeholders, mark the MoE scope, and select lifecycle hooks that match the projection pattern. Preserve shared-expert ordering.
5. Set only evidenced class hooks such as `pre_lm_head_norm_module`, AWQ shape-dependent modules, required dtype/packages/processor/loader, module-tree overrides, or conversion maps.
6. Import the definition and register every supported `model_type` in `MODEL_MAP` in `gptqmodel/models/auto.py`. Gate definitions that require a particular Transformers version.

If method-specific execution differs, use `module_tree_overrides` instead of weakening the default tree. Keep aliases explicit when runtime shell and checkpoint names differ.

## Validate end to end

1. Unit-test module-tree expansion: exact ordered blocks, non-quantized flags, capture-only behavior, aliases, and expert expansion.
2. Instantiate through the public auto-model path and verify the expected adapter is selected.
3. Quantize a tiny checkpoint with GPTQ and AWQ when both are claimed. Exercise dynamic layer settings if supported.
4. Save, reload, and generate through a real compatible backend; confirm config and tensor names survive.
5. Add or extend a `ModelTest` using the nearest model-family fixture. Run the fast path first, then the slow/full evaluation when math, grouping, or lifecycle changes could affect quality.
6. Test optional configs and variants independently: dense versus MoE, text-only versus multimodal, tied weights, GQA/MQA, shared experts, or remote code as applicable.

Report unsupported variants explicitly. A model that merely loads, or one that quantizes only because all target modules were skipped, is not supported.
