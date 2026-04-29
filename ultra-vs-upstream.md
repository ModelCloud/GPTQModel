# GPTQModel Ultra vs Upstream GPTQModel

Comparison baseline:

| Branch | Commit |
|---|---|
| Ultra `main` | `26841b63` |
| Upstream `upstream/main` | `1002e69d` |

At this baseline, Ultra is 37 commits ahead of upstream and 1 upstream commit behind. The content diff spans 23 files with about 747 added lines and 81 removed lines.

## Summary

GPTQModel Ultra currently differs from upstream GPTQModel mainly by adding embedding requantization support. Ultra adds a `requantize()` API for already-quantized models, a `QuantizeEmbed` mode selector, GPTQ support for `nn.Embedding`, quantized embedding save/load handling, and embedding-aware looper execution.

Ultra is also missing one latest upstream CI-fix commit at this baseline.

## Change Inventory

| Area / file | Direction | Change |
|---|---:|---|
| `gptqmodel/__init__.py` | Ultra-only | Exports `QuantizeEmbed` from the public package API. |
| `gptqmodel/quantization/config.py` | Ultra-only | Adds `QuantizeEmbed` enum values: `input`, `output`, and `both`. |
| `gptqmodel/quantization/__init__.py` | Ultra-only | Exports `QuantizeEmbed` from the quantization package. |
| `gptqmodel/models/base.py` | Ultra-only | Adds `requantize()` for already-quantized models. |
| `gptqmodel/models/base.py` | Ultra-only | Adds `quantize(..., embed_quant_mode=...)` and allows quantization re-entry only for embedding requantization. |
| `gptqmodel/models/base.py` | Ultra-only | Adds input/output embedding getters and embedding module-name helpers. |
| `gptqmodel/models/base.py` | Ultra-only | Passes `embed_quant_mode` into `ModuleLooper`. |
| `gptqmodel/models/_const.py` | Ultra-only | Adds `nn.Embedding` to supported quantizable module types. |
| `gptqmodel/looper/input_cache.py` | Ultra-only | Adds `src_inputs` to cache raw `input_ids`, defaulting to `layer_inputs` for compatibility. |
| `gptqmodel/looper/loop_processor.py` | Ultra-only | Updates empty `InputCache` construction for the new `src_inputs` field. |
| `gptqmodel/looper/stage_inputs_capture.py` | Ultra-only | Captures raw `input_ids` when input embedding quantization is active. |
| `gptqmodel/looper/module_looper.py` | Ultra-only | Tracks `embed_quant_mode`, input/output embedding modules, and embedding module names. |
| `gptqmodel/looper/module_looper.py` | Ultra-only | Unties tied word embeddings before embedding requantization. |
| `gptqmodel/looper/module_looper.py` | Ultra-only | Validates selected embedding modules and injects default embedding dynamic config: `bits=8`, `group_size=32`, `sym=True`, `desc_act=False`, `mse=2.4`. |
| `gptqmodel/looper/module_looper.py` | Ultra-only | Adds progress steps for input/output embedding quantization and hooks output embeddings or `lm_head` where needed. |
| `gptqmodel/looper/forward_executor.py` | Ultra-only | Adds `is_embeddings_module`, skips layer replay kwargs and `use_cache` for embeddings, and calls embeddings without keyword replay inputs. |
| `gptqmodel/utils/looper_helpers.py` | Ultra-only | Mirrors embedding-aware forward behavior in the parallel batch worker. |
| `gptqmodel/looper/stage_layer.py` | Ultra-only | Treats input embeddings, output embeddings, and `lm_head` as quantization stage units. |
| `gptqmodel/looper/stage_layer.py` | Ultra-only | Uses `src_inputs` for input embeddings and `lm_head_pre_quantize_generate_hook()` for output embeddings. |
| `gptqmodel/looper/stage_subset.py` | Ultra-only | Builds subset plans for standalone embedding modules. |
| `gptqmodel/looper/stage_subset.py` | Ultra-only | In embedding-only mode, skips non-embedding quant tasks while still forwarding layers for activation propagation. |
| `gptqmodel/quantization/gptq.py` | Ultra-only | Adds GPTQ support for `nn.Embedding`: token-count diagonal Hessian, transposed embedding weights, diagonal inverse, and an embedding-specialized quantization path. |
| `gptqmodel/quantization/gptq.py` | Ultra-only | Recognizes existing `QuantLinear`-style modules and frees embedding Hessian state. |
| `gptqmodel/nn_modules/qlinear/torch.py` | Ultra-only | Adds `TorchQuantEmbeddings`, backed by Torch GPTQ dequantization and `torch.nn.functional.embedding`. |
| `gptqmodel/nn_modules/qlinear/__init__.py` | Ultra-only | Updates packing to handle embeddings with no bias and transposed embedding weights. |
| `gptqmodel/utils/model.py` | Ultra-only | Updates `find_modules()` to recognize embeddings, `BaseQuantLinear`, and `*QuantLinear` classes. |
| `gptqmodel/utils/model.py` | Ultra-only | Updates `create_quant_module()` and `create_quant_layer()` to support `nn.Embedding` and select `TorchQuantEmbeddings`. |
| `gptqmodel/utils/model.py` | Ultra-only | Adds `untie_word_embeddings()`, `get_module_name()`, and checkpoint inspection for quantized input/output embeddings. |
| `gptqmodel/models/loader.py` | Ultra-only | Detects quantized embedding modules in saved safetensors and preserves them during load. |
| `gptqmodel/models/loader.py` | Ultra-only | Forces loaded quant config runtime state with `qcfg.device = device` and `qcfg.offload_to_disk = False`. |
| `gptqmodel/models/writer.py` | Ultra-only | Preserves quantized input/output embeddings when saving and regenerating model shells. |
| `tests/test_stage_modules.py` | Ultra-only | Updates `cache_inputs()` test call for the new `embed_quant_mode` argument. |
| `gptqmodel/looper/exllamav3_processor.py` | Upstream-only, not yet in Ultra | Upstream adds `act_group_aware=False` to EXL3 capture config. |
| `tests/models/test_falcon.py` | Upstream-only, not yet in Ultra | Upstream imports `BACKEND` and sets `LOAD_BACKEND = BACKEND.AUTO`. |
| `tests/models/test_gpt_oss.py` | Upstream-only, not yet in Ultra | Upstream imports `BACKEND` and sets `LOAD_BACKEND = BACKEND.AUTO`. |

## Ultra-Only Commit Themes

| Theme | Representative commits |
|---|---|
| Requantization API and lm_head support | `a63c87d9`, `b8c68157`, `6a32298d`, `5c6e6176` |
| Embedding quantization support | `479c321f`, `1c11366e`, `c83f14ee`, `3a31de31`, `98ffbf36` |
| Input/output embedding modes | `323dc96a`, `01402b92`, `9cf74f8a`, `4e05a097` |
| Upstream sync and dependency updates | `26841b63`, `39006a0d`, `5c531e3d` |

## Practical Impact

| Capability | Upstream GPTQModel | GPTQModel Ultra |
|---|---|---|
| Quantize transformer layers | Supported | Supported |
| Quantize `lm_head` | Supported | Supported with additional requantization fixes |
| Requantize an already-quantized model | Not present at this baseline | Added through `requantize()` |
| Quantize input embeddings | Not present at this baseline | Added through `QuantizeEmbed.INPUT` or `QuantizeEmbed.BOTH` |
| Quantize output embeddings | Not present at this baseline | Added through `QuantizeEmbed.OUTPUT` or `QuantizeEmbed.BOTH` |
| Save/load quantized embeddings | Not present at this baseline | Added safetensors detection and preservation |
| Embedding GPTQ Hessian handling | Dense module path only | Adds embedding-specific diagonal Hessian path |
| Embedding inference module | Not present at this baseline | Adds `TorchQuantEmbeddings` |

