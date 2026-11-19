# MobileLLM: license scope

Upstream: facebookresearch/MobileLLM at revision `6cb80c4064183622a64105c875cdd8a35fc8c56b`
(`utils/modeling_llama.py`), which is licensed under the **FAIR Noncommercial Research License**
and its Acceptable Use Policy.

Port history: commits `cda891d7a` (initial port), `bf9c9b664` (cache/rotary rollback),
`1e85d8ec9`, `b030e1535`, and `45fdcc995`.

## Restricted-derived: replace or remove

| file | evidence |
|---|---|
| `gptqmodel/models/ports/mobilellm/modeling_mobilellm.py` (entire file) | Header: "Copyright (c) Meta Platforms ... Licensed under the FAIR Noncommercial Research License"; "Derived from facebookresearch/MobileLLM utils/modeling_llama.py". Every class and function in it falls under that header: `_restore_forward_state_on_error`, `MobileLLMRMSNorm`, `MobileLLMRotaryEmbedding`, `_rotate`, `_column_parallel`, `_row_parallel`, `eager_attention_forward`, `MobileLLMAttention`, `MobileLLMMLP`, `MobileLLMDecoderLayer`, `MobileLLMPreTrainedModel`, `MobileLLMModel`, `MobileLLMForCausalLM`. |
| `gptqmodel/models/ports/mobilellm/configuration_mobilellm.py` (entire file, `MobileLLMConfig`) | Same Meta/FAIR header. |
| `gptqmodel/models/ports/mobilellm/__init__.py` | Same Meta/FAIR header. It is a trivial re-export, but it should be rewritten with an Apache header. |
| `gptqmodel/models/ports/mobilellm/LICENSE` | Byte-for-byte FAIR Noncommercial Research License and AUP. Delete it once the code has been replaced. |
| `gptqmodel/models/ports/mobilellm/NOTICE` | FAIR attribution and derivation notice for the files above. Replace it, or drop it if no third-party code remains. |

`modeling_mobilellm.py` also carries an Apache-2.0 attribution to EleutherAI and HuggingFace,
because upstream MobileLLM is itself based on HF Llama. That attribution does **not** make the
file permissive, since the FAIR terms apply to the derivative as a whole. The Apache building
blocks the file used can instead be taken directly from the vendored
`gptqmodel/models/ports/native/llama` port (marked "Vendored from huggingface/transformers v5.23.0
(Apache-2.0)"), or from Transformers itself.

## Permissive: keep (Apache-2.0, ModelCloud)

| file | evidence |
|---|---|
| `gptqmodel/models/definitions/mobilellm.py` (`MobileLLMQModel`) | SPDX-License-Identifier: Apache-2.0, ModelCloud. It is a GPT-QModel definition: a `LlamaQModel` subclass with a loader that points at the port, plus a guard that rejects `pretraining_tp > 1`. It contains no upstream code. |
| `tests/test_mobilellm_local_port.py` | SPDX Apache-2.0, ModelCloud. It imports the port only through its public names. |
| `tests/module_tree/test_mobilellm_local_port.py` | SPDX Apache-2.0, ModelCloud. |
| registry lines in `gptqmodel/models/auto.py` (`"mobilellm": MobileLLMQModel`) and `gptqmodel/models/definitions/__init__.py` | Apache, repository-wide. |
| documentation mentions in `README.md`, `docs/supported-models.md`, `docs/transformers5-model-ports.md`, `docs/fusion-model-coverage.md`, `docs/rotation-model-support.md` and `docs/transformers-5-compat.md` | Apache. The license line in `docs/transformers5-model-ports.md` row `MobileLLMQModel` ("FAIR Noncommercial Research License and AUP") should be updated after the reimplementation. |

## Interface facts

The class, config field and state-dict names in SPEC.md section 1 are interface facts needed for
checkpoint compatibility. They are not copyrightable expression, and the reimplementation must
keep them.
