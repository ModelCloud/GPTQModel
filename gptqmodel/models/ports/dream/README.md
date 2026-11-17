<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dream local model in GPT-QModel

This package owns Dream configuration, bidirectional decoder layers, and diffusion
generation. `DreamQModel.loader` selects these classes directly through
`LocalModelLoader`; checkpoint `auto_map` and API trust flags cannot replace them.
Transformers 5.23.0 has no native Dream config or AutoModel mapping.

## Source and license

Adapted from [Dream-org/Dream-v0-Instruct-7B at revision
05334cb9faaf763692dcf9d8737c642be2b2a6ae](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/tree/05334cb9faaf763692dcf9d8737c642be2b2a6ae).
The inspected files are Apache-2.0, copyright 2024 The Dream team, HKUNLP Group
and the HuggingFace Inc. team. Their copyright and license notices remain in the
adapted files. The modeling source also credits GPT-NeoX, OPT and Qwen ancestry;
that notice is preserved. See [LICENSE](LICENSE) for the license text.

SHA-256 of the original inspected files (before adaptation):

| File at pinned revision | SHA-256 |
| --- | --- |
| `configuration_dream.py` | `24d038962ccf16361595494461000ec0c1b043653c7494d7fc835c96f121f0b9` |
| `modeling_dream.py` | `3166e789f0d69beb1f4fbdee2317953d1c1c3b9bc21479b109afc5dd059de5b3` |
| `generation_utils.py` | `7f8ad01484898946b3c9c9d5ebc85e5ac726061d0fa7b4a522c4b8110c13a921` |
| `config.json` | `11afc7fe7a1881e73dcd70cca6868aaaf8eb89ad45ea5fc6426c3fb82dec9340` |
| `model.safetensors.index.json` | `998a078123ffc97763690de7f2a677eb89168af5eaf8a5e12e6bc24d18e25bdb` |

The published checkpoint has 28 layers, hidden size 3584, 28 query heads and
4 KV heads; its 339 state-dict keys are embeddings, final norm, LM head, and
12 tensors per layer. The local model preserves these names, biased Q/K/V,
unbiased output/MLP projections, RMSNorm, SiLU gating, and bidirectional attention.

## Transformers 5 adaptation

The port normalizes legacy RoPE metadata without mutating caller dictionaries,
uses current config/cache/output APIs, and supports Torch eager and SDPA attention.
The old optional FlashAttention import is absent. Rank-2 padding masks and rank-4
boolean/additive masks are supported. Nonpersistent rotary buffers are explicitly
rebuilt when Transformers loads from meta storage; they are absent from safetensors.

`generate()` invokes Dream's full-sequence denoising sampler, including its shifted
logits, timestep schedule and `origin`, `maskgit_plus`, `topk_margin`, and `entropy`
algorithms. It always forwards with `use_cache=False`. Explicit caller-supplied
`DynamicCache` continuation preserves upstream incremental projection semantics;
it is **not** equivalent to recomputing a bidirectional prefix with future context.

See [local loading and diffusion generation](docs/local-loading-and-diffusion-generation.md)
for the decision flow, failure paths and old/new mechanism comparison.

## Quantization policy and limits

Dream declares seven quantized projection linears per layer and two non-quantized
norms. MHA, GQA and MQA variants have real meta-model ownership/role tests.
AWQ scale search, norm folding and replay are **unqualified** for diffusion
attention. `DreamQModel.awq_get_modules_for_scaling()` returns no groups so the
existing AWQ plain-quantization path still quantizes and packs all seven linears.
CPU tests capture actual projection inputs, quantize and pack both layers, verify
norms are unchanged and execute the resulting `AwqTorchLinear` model.

Batch quantization remains disabled. Sliding-window attention retains the
upstream eager/SDPA warning and has not been qualified. GPU kernels, accelerator
FP16/BF16 behavior, FlashAttention, concurrent GIL=0 accelerator execution,
trained 7B weights, uncached Hub/tokenizer loading, full quantization/evaluation,
and task-quality scores have not been validated by this port's CPU suite.

## CPU qualification

Validated on Python 3.14t, Transformers 5.23.0 and Torch 2.15.0+cpu:

```bash
python -m pytest -q tests/test_dream_local_port.py tests/module_tree/test_dream_local_port.py
```

The suite covers direct and public model selection under all API trust/existing
security-bypass combinations with malicious checkpoint `auto_map`, FP32/FP16/BF16
forward comparison against an independent functional oracle, eager/SDPA and
MHA/GQA/MQA layouts, masks, cache reset/continuation, all four diffusion samplers,
failure after a completed denoising step followed by retry, and tied/untied
safetensors save/reload equivalence. It uses tiny real CPU/meta models and local
fixtures, without downloading trained weights or executing checkpoint Python.

The public-load test is red on the assigned base when trust and the existing
bypass are both enabled: tokenizer discovery re-enters remote AutoConfig. The
parent's shared fix (`5423ad629`, supplied for integration) forwards the already
selected config to AutoTokenizer and makes all four cases pass. This package
does not patch shared tokenizer logic. The rotary-buffer reload regression also
failed numerically before `_init_weights` rebuilt the nonpersistent buffers.
