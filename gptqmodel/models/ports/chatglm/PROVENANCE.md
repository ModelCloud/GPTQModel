<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ChatGLM modeling provenance

GPT-QModel owns the causal-language-model implementation for the existing
`chatglm` definition. Transformers 5.23.0 has no `chatglm` config or causal-LM
mapping; its native `glm` and `glm4` classes have different checkpoint trees.

The following public sources were read as text, without importing their Python.
Revisions are immutable Hugging Face git commits.

| Source | Revision | Inputs |
| --- | --- | --- |
| [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b/tree/d2e2d91789248536a747d9ce60642a336444186c) | `d2e2d91789248536a747d9ce60642a336444186c` | config, modeling, legacy weight index |
| [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b/tree/e9e0406d062cdb887444fe5bd546833920abd4ac) | `e9e0406d062cdb887444fe5bd546833920abd4ac` | config/configuration, modeling, legacy weight index |
| [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat/tree/bd8234fe5e0c09c48637a92abb0c797cb5fa0e73) | `bd8234fe5e0c09c48637a92abb0c797cb5fa0e73` | config, modeling, safetensors index |
| [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b/tree/bf0f5cfb575eebebf9b655c5861177acfee03f16) | `bf0f5cfb575eebebf9b655c5861177acfee03f16` | incompatible original ChatGLM1 config, for rejection |

`core.py` adapts ChatGLM3's encoder math, shared with ChatGLM2.
`core_glm4.py` adapts the GLM4 batch-first attention layout and extended RoPE
base. `configuration_chatglm.py` adapts the upstream configuration.
`modeling_chatglm.py` supplies current Transformers initialization, outputs,
generation and the atomic `DynamicCache` boundary. Checkpoint keys, including
the persistent rotary buffer, remain unchanged. `checkpoint_contracts.json`
records the exact public configurations and sorted checkpoint keys used by the
full-size meta-model audit (200, 200 and 284 keys respectively).

The upstream ChatGLM implementation repositories license code under Apache 2.0;
their license is retained in `LICENSE`. The rotary helper also identifies
MIT-licensed labml.ai derivation; that notice is retained in `LICENSE.rope`.
Model-weight licenses are separate; this package contains no trained weights.
The retained Apache `LICENSE` was verified byte-for-byte against ChatGLM3 git
`732294f42e2625509000daa87a9c3748d0bbb957`. `LICENSE.glm4` preserves GLM-4's
Apache notice at git `024d8fd48e62cf53785b68d657f1e018f87b5674`;
the retained MIT notice matches
labml.ai git `db59a04a8a2c6af922f1656f8c0fd113dd9a51b8`.

## Adaptations and limits

- GLM2/3 retain sequence/batch/head/cache axes internally; GLM4 retains
  batch/head/sequence cache axes. Public caches use Transformers'
  batch/head/sequence convention, translating only at the model boundary.
- GLM4 is selected by the checkpoint's explicit `rope_ratio`, including its
  published value of 500. An unrelated native GLM architecture is never used.
- Rotary frequencies are computed in FP32, following pinned ChatGLM3/GLM4.
  Older ChatGLM2 computed frequencies in activation precision; its FP16/BF16
  long-context rounding is intentionally not reproduced. Original ChatGLM2
  low-precision bitwise parity is not claimed.
- Removed upstream global Torch JIT changes, obsolete Transformers-version
  checks, remote quantizer imports and checkpoint-code generation hooks.
  HF meta/no-init contexts replace `skip_init`. Generation uses a real cache.
- Standard `generate()` is supported. Upstream tokenizer-specific `chat()`,
  `stream_chat()`, response parsing and classification heads are outside this
  causal-LM definition. No tokenizer correction or chat template is embedded.
- Original ChatGLM1's `transformer.layers`/2D-position architecture is rejected
  by config markers before weights load. Upstream CUDA `quantization_bit`
  checkpoints are also rejected; use GPT-QModel quantization.
- Prefix tuning, MQA/non-MQA, LayerNorm/RMSNorm, residual and bias variants are
  covered by tiny-model tests. GPU, trained-checkpoint quality, very long
  contexts, static/sliding cache implementations and optimized kernels need
  separate validation. Current cache support is `DynamicCache`, with legacy
  tuples accepted for direct forward continuation only.
- AWQ explicitly uses plain quantization for both layouts, with an empty
  scaling-group list. Packed QKV producer/consumer folding and scale-search
  replay are unqualified; no safety claim is inferred from matching widths.
  Real CPU AWQ quantize/save/reload tests cover both layouts and MQA modes.

See [coverage and loading flow](COVERAGE.md) for runnable validation.
