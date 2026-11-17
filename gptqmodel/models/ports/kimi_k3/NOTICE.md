# Kimi K3 source and license notices

This directory adapts Moonshot AI's Kimi K3 inference implementation from
[`moonshotai/Kimi-K3` at `f831ab66814297da540d832a5235f8e904f29d06`](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06).
The unmodified checkpoint configuration is retained in `checkpoint_config.json`.
No checkpoint Python is imported or downloaded by this package at runtime.

| Shipped file | Original file and attribution |
| --- | --- |
| `configuration_kimi_k3.py` | Moonshot AI's `configuration_kimi_k3.py` |
| `modeling_kimi_linear.py` | Moonshot AI's `modeling_kimi_linear.py`; DeepSeek-AI and Hugging Face Apache-2.0-derived MLA/MoE sections retain their header |
| `modeling_kimi_k3.py` | Moonshot AI's `modeling_kimi_k3.py`; original LLaVA/Hugging Face Apache-2.0 attribution retained |
| `kimi_k3_processor.py`, `kimi_k3_vision_processing.py`, `media_utils.py` | Identically named Moonshot AI media implementation files |
| `tokenization_kimi.py`, `encoding_k3.py` | Identically named Moonshot AI architecture tokenizer and XTML encoder files |

The upstream custom **Kimi K3 License** is reproduced in `LICENSE`. It is not
replaced by GPT-QModel's Apache-2.0 license; review its conditions before use or
redistribution. The Apache-2.0 text for derived sections is in `LICENSE.apache`.
The canonical OpenAI/Hugging Face GPT-2 byte-to-Unicode alphabet is copied into
the tokenizer because Transformers 5 removed its former public import path.

`eager_kda.py` adapts the MIT-licensed Torch recurrence, causal depthwise
convolution and normalization equations from
[`fla-org/flash-linear-attention` at `7edb579d636139df2b31df580d497928121d35d4`](https://github.com/fla-org/flash-linear-attention/tree/7edb579d636139df2b31df580d497928121d35d4):
`fla/ops/kda/naive.py`, `fla/ops/kda/gate.py`,
`fla/modules/causal_conv1d/ops.py`, and `fla/modules/norm/layernorm_gated.py`.
Copyright 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li; see `LICENSE.fla`.
The checkpoint's sigmoid output gate is retained rather than substituting the
modern FLA module's default SiLU gate. This is a portable reference execution
path, not a qualification of FLA/Triton fused kernels.

## Pinned tokenizer assets

Tokenizer vocabulary/configuration remain checkpoint data, not bundled weights
or executable code. They are loaded through the package-selected tokenizer
class. All hashes below describe upstream bytes before formatting/adaptation.

| Upstream asset | SHA-256 |
| --- | --- |
| `tokenization_kimi.py` | `f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944` |
| `encoding_k3.py` | `49ff03305fdc4be26867972788d36150b67f8a9e852e62bb7959d87482223676` |
| `tiktoken.model` | `b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103` |

K2, K2.5 and K3 share vocabulary asset bytes but **not** their architecture chat
encoding contract. This implementation intentionally does not replace the K2
or K2.5 tokenizers.
