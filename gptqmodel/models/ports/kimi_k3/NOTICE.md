# Kimi K3 source and license notices

This directory adapts Moonshot AI's Kimi K3 inference implementation from
[`moonshotai/Kimi-K3` at `f831ab66814297da540d832a5235f8e904f29d06`](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06).
A compacted copy of the checkpoint configuration is kept as a test fixture in
`tests/fixtures/ports/kimi_k3/checkpoint_config.json`.
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
redistribution. The Apache-2.0 text for derived sections is in the repository root `LICENSE`.
The GPT-2 byte-to-Unicode alphabet is imported from
`transformers.convert_slow_tokenizer`.

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

## Ordinary Kimi-Linear tokenizer and padded K3 tensor provenance

The ordinary source is
[`moonshotai/Kimi-Linear-48B-A3B-Instruct@e1df551a447157d4658b573f9a695d57658590e9`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/tree/e1df551a447157d4658b573f9a695d57658590e9).
Its pinned model card declares MIT, without a separate LICENSE file or tokenizer
copyright header. The shared K3 implementation retains the K3 license above;
it is not relabelled MIT. Ordinary sorted-tools/Jinja semantics are distinct
from K3 XTML. Source/asset SHA-256 values:

| Asset | SHA-256 |
| --- | --- |
| `tokenization_kimi.py` | `b6b304f63d355f29f5f5aa27694337c824c90e1ac80559ca537909824729761f` |
| `tokenizer_config.json` | `5ba77d5822137ac3cb9337d2b84561d10425562cf8b8f2c74c91a832516a778d` |
| `chat_template.jinja` | `75e3cd76bd26b77c02507eb35073b66c8ffd0c88eab185ebc6f8c6e425161809` |
| `tiktoken.model` | `b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103` |

`tests/fixtures/ports/kimi_k3/a_log_reference.json` reproduces 512 tensor bytes from K3 at the same pinned
revision/license above: `model-00013-of-000096.safetensors`, header length824104,
tensor offsets[3584,4096), file bytes[827696,828208), SHA-256
`13f756b245c0f444a480166b5398f1ba788de4e4926f04b351edc0f4378643dd`.
These are small source-reference data, not a complete trained checkpoint.
