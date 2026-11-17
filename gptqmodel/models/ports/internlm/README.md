# InternLM 1 Transformers 5 port

Adapted from the InternLM and Hugging Face teams' Apache-2.0 implementation:

- Repository: [`internlm/internlm-7b`](https://huggingface.co/internlm/internlm-7b/tree/96e127d08d851a88cac736a9b091dd953ae1b873).
- Revision: `96e127d08d851a88cac736a9b091dd953ae1b873`.
- Sources: `configuration_internlm.py`, `modeling_internlm.py` and
  `pytorch_model.bin.index.json`. No checkpoint Python is imported at runtime.
- Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
  The upstream code is based on Transformers' Llama implementation and licensed
  under the Apache License, Version 2.0; see the repository's [LICENSE](../../../../../LICENSE).
- GPT-QModel modifications: current config/model/generation APIs, SDPA attention,
  raw-key `DynamicCache` integration with position metadata, call-local dynamic
  NTK rotary tables, safetensors reload, and quantization capture via keyword arguments.

This is InternLM **1**, with full-head Q/K/V, optional attention biases, RMSNorm,
gated SiLU MLP, and `origin` or dynamic NTK rotary. InternLM 2 and later are
separate architectures. The original checkpoint index owns
`self_attn.rotary_emb.inv_freq`; it remains a persistent, explicitly excluded
module-plan buffer. No checkpoint-key conversion is required.

Keys are cached before rotary, as in the upstream forward: crossing the trained
context length changes the dynamic rotary base for *all* keys. Position metadata
is retained for padded batches and follows beam reorder, crop, repeat, select and
reset. Non-empty foreign caches, static caches and quantized caches are rejected.
The upstream streaming chat/thread helpers and sequence-classification head are
outside GPT-QModel's causal-LM interface and are not ported.

AWQ policy: **plain quantization only**, with no scale-search/folding groups.
The tests quantize all seven linears per decoder layer (with and without
attention bias) and save/reload them through the public API. Fusion is unqualified.

Qualification uses tiny real CPU models and independent upstream-style tensor
computations in FP32, FP16 and BF16, plus real meta-model module plans. It covers
generation, raw-key continuation across dynamic NTK boundaries, beam operations,
causality, padding, malicious `auto_map`, public dense loading, and safetensors
dense/AWQ round trips. GPU kernels and trained 7B quality are not qualified by
these tests; the checkpoint index was inspected without downloading its weights.

## Package-owned release tokenizer

`InternLMQModel.loader.tokenizer_class` binds `InternLMTokenizer` directly.
It reads the release's SentencePiece `tokenizer.model` without executing
checkpoint Python, including with explicit trust and the security bypass enabled.
See [InternLM tokenizer selection and serialization](TOKENIZERS.md) for exact
release/source hashes, shared InternLM2 algorithm, prompt/ID qualification,
before/after flow diagrams, failure/retry behavior and validation commands.
