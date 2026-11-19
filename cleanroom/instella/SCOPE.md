# Instella port: license scope

Port directory: `gptqmodel/models/ports/instella/`

## Restricted (to be replaced by the clean-room implementation)

| file | contents | evidence |
|---|---|---|
| `configuration_instella.py` | all of it (`InstellaConfig`) | header `SPDX-License-Identifier: LicenseRef-Instella-Research-Only-RAIL-MS`; `Modifications copyright(c) 2025 Advanced Micro Devices,Inc.`; docstring says it is adapted from the pinned AMD checkpoint implementation |
| `modeling_instella.py` | all of it: the rotary module, the eager attention function, attention, decoder layer, pretrained base (incl. `_init_weights`), `InstellaModel`, `InstellaForCausalLM` | the same `LicenseRef-Instella-Research-Only-RAIL-MS` SPDX header plus the AMD copyright; "adapted from OLMo2 (Transformers v4.47.1) by AMD" |
| `LICENSE` | AMD "Instella-3B [RESEARCH-ONLY RAIL-MS]". Its Licensed Artifacts list both "Model" and "Source Code", and its definition of Source Code includes derivatives | delete it together with the two files above (it should go only once nothing in the directory is derived from AMD code) |
| `NOTICES` | AMD's third-party notices: dependencies on allenai OLMo (Apache-2.0) and Qwen2.5-72B-Instruct (Qwen license / Apache-2.0) | this belongs to the AMD artifact. Delete it with LICENSE, because the Apache-2.0 reimplementation builds only on the vendored HF code, whose attribution is `ports/native/NOTICE` |

Some parts of `modeling_instella.py` are imported, not authored there, and are already Apache-2.0:
`LlamaRMSNorm`, `apply_rotary_pos_emb` and `repeat_kv` come from `ports/native/llama/modeling_llama.py`, and
`Qwen2MLP` comes from `ports/native/qwen2/modeling_qwen2.py`. Both are vendored from HF Transformers v5.23.0 and
covered by `ports/native/NOTICE`. The new implementation should keep importing them.

## Permissive (keep)

| file | evidence |
|---|---|
| `gptqmodel/models/ports/instella/__init__.py` | `SPDX-License-Identifier: Apache-2.0`, ModelCloud. It only re-exports three class names. Its docstring says "see LICENSE and NOTICES", so update that once those files are deleted. |
| `gptqmodel/models/definitions/instella.py` (`InstellaQModel`, module_tree, loader) | Apache-2.0 header, ModelCloud. It contains no model math. A comment mentions `modeling_instella.py:663-664`, the upstream line numbers, as a fact about checkpoint attribute names. |
| `gptqmodel/models/auto.py` registry entry, `definitions/__init__.py` import | Apache-2.0 |
| tests: `tests/test_instella_local_port.py`, `tests/module_tree/test_instella_local_port.py`, `tests/test_instella_support.py` | Apache-2.0 headers, ModelCloud. Note: `reference_logits` in `test_instella_local_port.py` describes itself as an "independent transcription of AMD's eager equations". It is test-only, re-derived from the equations, and calls no port code. The implementation team must not read it, because it is a structural paraphrase of the restricted forward. |

## Docs to update after replacement (not code)

- `docs/transformers5-model-ports.md:119` says "Research-only RAIL-MS source terms retained". Update it after the
  replacement.
- `docs/fusion-model-coverage.md:245` and `docs/awq-scaling-groups.md` are neutral and need no change.

## Model weights

The AMD weights (`amd/Instella-3B*`) stay under the research-only RAIL-MS license whatever the code license is. A
clean-room code port changes only the license of the Python source. Users loading the official checkpoints are still
bound by AMD's model license.
