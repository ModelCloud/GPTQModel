# GPT-QModel Instella source provenance

The implementation derives from AMD's Instella checkpoint source, adapted to
Transformers 5.23.0. It is not an OLMo/Llama alias. Transformers 5.23.0 has no
`instella` entry in `CONFIG_MAPPING` or `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`.

Authoritative repository: [amd/Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-Instruct/tree/f83f548d8e20eccc8e403998970f42fb0acba657).
Pinned revision: `f83f548d8e20eccc8e403998970f42fb0acba657`.

| Upstream file | SHA-256 |
| --- | --- |
| `modeling_instella.py` | `65c3f7c1a7464e5db0e3e2e12c5968dc9c52fdab6b1d6cff6742fcd8f56ce8d0` |
| `config.json` | `869e73fa380578c84ac26d8a79c7e11ab8403ec831080740742887391a0d4060` |
| `model.safetensors.index.json` | `c1b11b0c0588866ec43cae0991a40f3706e716e898f883327490d1518f66cf4a` |
| `LICENSE` | `d5acdfe7de057b6e40b88df130e054bbfc79ddd5ffee07929301d52aedc81d68` |
| `NOTICES` | `7c0322cf06d97bff45be6ca1ed12ad591a777470711ca94378a0a1be0bc4577a` |

Source references in `modeling_instella.py` at that revision:

| Lines | Preserved contract |
| --- | --- |
| 31–188 | Config defaults, MHA/GQA/MQA and scaling fields |
| 191–319 | FP32 RMSNorm reduction, split-half RoPE, original/linear/dynamic NTK equations, KV repetition |
| 321–447, 557–630 | Full projected-width Q/K norms before head reshape; Q/K/V/O ownership; eager/SDPA |
| 633–750 | Bias-free gated MLP, pre-attention/pre-feedforward norms and residual ordering |
| 856–1008 | Embeddings, decoder traversal, final norm, cache positions and outputs |
| 1132–1236 | Causal LM head, logits selection and next-token loss |

The adaptation splits config/model modules, replaces legacy mask helpers with
`create_causal_mask`, and uses Transformers 5 `Cache`, `DynamicCache`,
`StaticCache`, generation, output and safetensors APIs. Config reload accepts
Transformers 5's `rope_parameters` serialization of legacy `rope_scaling`.
Nonpersistent rotary frequencies are rebuilt after meta initialization. Each
dynamic rotary call keeps its own frequency tensor reference while publishing
the warm buffer for subsequent calls; no in-place shared frequency mutation or
global RoPE registration is needed. Checkpoint code is never downloaded or
imported at runtime.

The pinned index contains 399 weight keys: 36 layers × 11 tensors plus the
embedding, final norm and LM head. SHA-256 of UTF-8
`"\n".join(sorted(index["weight_map"]))` (no trailing newline) is
`f21aaded620da6bf22f2a85f3328c144644250e0975533cef7d1a56ea5185a1a`.
The meta-model regression checks this independently extracted digest and
complete owner/quantization tags. No checkpoint weights were downloaded.

## License and attribution

`LICENSE` and `NOTICES` preserve the upstream text with trailing whitespace
removed to satisfy repository whitespace checks. The hashes above identify
the unmodified upstream files. AMD's LICENSE explicitly covers
both **Model and Source Code** under **Instella-3B RESEARCH-ONLY RAIL-MS**;
the permitted purpose is academic or research use. It must not be represented
as Apache-2.0 or unrestricted commercial source. `NOTICES` separately preserves
Allen Institute for AI OLMo attribution and Apache-2.0 license text. Modified
files carry AMD copyright and ModelCloud modification notices. Distribution in
GPT-QModel requires the maintainer's license decision; preserving the license
does not itself resolve that compatibility question.

See [coverage and loading flow](COVERAGE.md) for tested behavior and limits.
