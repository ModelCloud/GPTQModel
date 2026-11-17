<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ChatGLM model and tokenizer provenance

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
- Standard `generate()` is supported. Upstream model `chat()`, `stream_chat()`,
  response parsing and classification heads are outside this causal-LM
  definition. Package-owned tokenizers preserve source prompt construction;
  released chat-template JSON remains a data asset, with normalization owned
  by Tokenicer.
- Original ChatGLM1's `transformer.layers`/2D-position architecture is rejected
  by config markers before weights load. Upstream CUDA `quantization_bit`
  checkpoints are also rejected; use GPT-QModel quantization.
- Prefix tuning, MQA/non-MQA, LayerNorm/RMSNorm, residual and bias variants are
  covered by tiny-model tests. GPU, trained-checkpoint quality, very long
  contexts, static/sliding cache implementations and optimized kernels need
  separate validation. Current cache support is `DynamicCache`, with legacy
  tuples accepted for direct forward continuation only.
- The HF `vocab_size` attribute aliases physical `padded_vocab_size`, keeping
  Transformers embedding/head resizing and subsequent checkpoint reconstruction
  synchronized. Explicit padded rows remain authoritative when older metadata
  also records a smaller logical vocabulary.
- With `fp32_residual_connection=True`, residual additions retain FP32 state.
  Each normalization consumes its parameter dtype, followed by attention/MLP
  compute in that dtype; the language head consumes the embedding/model dtype
  when final normalization is disabled. These casts follow actual parameters
  after `.half()`, `.bfloat16()` or dtype-selected loading, rather than stale
  configuration dtype. CPU tests cover both norms and residual placement modes.
- AWQ explicitly uses plain quantization for both layouts, with an empty
  scaling-group list. Packed QKV producer/consumer folding and scale-search
  replay are unqualified; no safety claim is inferred from matching widths.
  Real CPU AWQ quantize/save/reload tests cover both layouts and MQA modes.

See [coverage and loading flow](COVERAGE.md) for runnable validation.

## Released tokenizer ownership

The same immutable release revisions above supply `tokenization_chatglm.py`.
The derived `tokenization_glm2.py`, `tokenization_glm3.py` and
`tokenization_glm4.py` retain the source SentencePiece / role-token / TikToken
algorithms. Their code uses the corresponding retained Apache licenses and
attribution in `NOTICE`; no remote Python is imported by the shipped loader.

| Release | Source SHA-256 | Tokenizer asset SHA-256 |
| --- | --- | --- |
| GLM2 | `a223e318392c2a065b96ddfd371e780873cc12365719ceec86d4a610fa476f3c` | `e7dc4c393423b76e4373e5157ddc34803a0189ba96b21ddbb40269d31468a6f2` |
| GLM3 | `aa92d89aaa950bb3339ffe515ce6e8da7631fffb11319574f42093457baef660` | `e7dc4c393423b76e4373e5157ddc34803a0189ba96b21ddbb40269d31468a6f2` |
| GLM4 | `90a9a1b739f834cdf7a2f40959f9eeff2a705b80acefb2cea9c19ff3958a9279` | `5a493598071550244b2ee7f26118f3edec2150b9dfa967929a99052ac83fe716` |

`tokenizer_contracts.json` additionally records exact tokenizer-config hashes,
GLM3's special-token-map hash, source-derived IDs for five text cases per
release, padded batches, pair construction, GLM2 round prompts, GLM3 tool/history
construction and literal role commands, and GLM3/4 rendered chats. Tests download
only those data files at the specified immutable revisions and verify SHA-256.
They never fetch checkpoint Python. GLM2/4 publish no special-token-map file at
these revisions; GLM3's is included in the pinned inventory.

The independent oracle ran the inspected source under Transformers 4.44.2 on
Python 3.12.16, SentencePiece 0.2.1, TikToken 0.12.0 and Jinja2 3.1.6. Only
GLM2's unused Torch import and GLM3's final tensor-container conversion were
removed in that isolated no-Torch oracle; integer tokenization and templates
were unchanged. Runtime qualification uses Transformers 5.23.0,
SentencePiece 0.2.3 and TikToken 0.12.0. GLM4 source pair input raises
`KeyError('<eos>')`; pair input is not qualified for GLM4, and its source behavior
is recorded rather than advertised as supported.

### Transformers 5 tokenizer adaptations

- One package-owned `ChatGLMTokenizer.from_pretrained` dispatcher reads only
  tokenizer JSON metadata. Released GLM2/3 share the class name
  `ChatGLMTokenizer`; GLM3's role-token metadata/chat template selects its source
  class, while GLM2's metadata selects its source class. GLM4's explicit
  `ChatGLM4Tokenizer` selects TikToken. Saved package classes use distinct
  `ChatGLM2Tokenizer`/`ChatGLM3Tokenizer`/`ChatGLM4Tokenizer` names. Unknown class
  names fail closed, even with trust and bypass. `auto_map` is never a dispatch
  authority, and all imports are package-relative.
- TF5 passes `padding_side` into `_pad`; the GLM2 signature accepts it while
  retaining source left-padding semantics. GLM3 replaces removed
  `batch_encode_plus` with `prepare_for_model` for already encoded chat IDs.
  Its source-specific `encode_special_tokens` constructor flag is passed to
  the base initializer so save/reload preserves literal role-command encoding.
- Source SP display conversion hides BOS/EOS/UNK as empty strings. Enumerating
  that display conversion as a vocabulary made TF5 register EOS/UNK over
  GLM3's fixed assistant/observation IDs 64796/64797. Both SP ports enumerate
  actual SentencePiece pieces plus fixed commands in `get_vocab`; encoding,
  decoding and prompt IDs retain source semantics. This corrects vocabulary
  ownership in the source port, without adding a generic normalization shim.
- The loader constructs the selected tokenizer before the shared outer
  Tokenicer normalization. Reusable metadata/special-token normalization is
  owned by [Tokenicer](https://github.com/ModelCloud/Tokenicer/pull/65), not this
  dispatcher. Parent owns the shared once-only wrapper adapter and dependency
  pins. The released source asset matrix works with released Tokenicer 0.0.17
  and the unreleased 0.0.19 preload candidate at
  `c14e1c8c062c7b7cc9a981054366f98bde015fa1`.

No trained-weight generation-quality, evaluation-score, arbitrary tokenizer
revision, original ChatGLM1 tokenizer, multi-threaded tokenizer, or GPU claim
is made. Model cache/dtype/AWQ limits above still apply.
