# Ovis2.6 MoE / Next: Transformers 5 local ownership

GPT-QModel owns the Ovis wrapper, configs, NaViT/SigLIP2 vision, soft visual
vocabulary and processor. `ovis2_6_moe` selects native `Qwen3MoeForCausalLM`;
`ovis2_6_next` selects native `Qwen3NextForCausalLM`. Neither exact wrapper has
a native Transformers 5.23 AutoConfig/AutoModel mapping. Native Ovis2 is not
a replacement for either. Checkpoint Python is never selected for these aliases.

## Sources and attribution

| Release | Pinned revision | Actual tensor-header inventory |
| --- | --- | --- |
| [30B](https://huggingface.co/AIDC-AI/Ovis2.6-30B-A3B/tree/44beb8e150b28c7ca537a1673582c64f57d342e3) | `44beb8e150b28c7ca537a1673582c64f57d342e3` | 19,308 keys/shapes |
| [80B](https://huggingface.co/AIDC-AI/Ovis2.6-80B-A3B/tree/96dd72344e17f098ba7cbe108e6872d12ba7cbfb) | `96dd72344e17f098ba7cbe108e6872d12ba7cbfb` | 74,832 keys/shapes |

Inspected source: each release's `configuration_ovis2_6.py`,
`modeling_ovis2_6.py`, config, model card, checkpoint index and every shard
header. `provenance.json` records SHA-256 hashes of the inspected data and
checked-in fixtures. Gzipped shape inventories contain names/shapes only, not
trained tensors. The 30B index omits the two vision post-LayerNorm tensors
present in its actual shard headers; both present and absent cases are tested.
The 80B index and headers agree. Fixtures also include actual pinned tokenizer
and image-processor data; both releases share the identical serialized Qwen BPE.

Both pinned cards declare Apache-2.0. See `LICENSE` and `NOTICE`; retained
vision attribution identifies Qwen2.5-VL rotary/windowing and SigLIP-style
encoder origins. No checkpoint Python is executed to discover behavior.

## Resolution and failure boundaries

```mermaid
flowchart TD
    A[GPTQModel.load: model_type] --> B{Definition-owned LocalModelLoader}
    B -->|ovis2_6_moe| C[Ovis2_6_Moe_Config + Qwen3MoeForCausalLM]
    B -->|ovis2_6_next| D[Ovis2_6_Next_Config + Qwen3NextForCausalLM]
    C --> E[Packaged Ovis wrapper + Siglip2NavitModel]
    D --> E
    E --> F[Tokenicer-normalized tokenizer + Ovis26Processor]
    B -->|foreign decoder or vision type| X[ValueError: fail closed]
    T[Checkpoint auto_map / trust / bypass] -. cannot replace package classes .-> B
    style X fill:#fdd
```

```mermaid
flowchart TD
    A[PIL image or frame-list video] --> B[Source resize / normalize / patch order]
    B --> C[NaViT 2-D RoPE + window/full frame attention]
    C --> D[Pre-post-LN hidden state: spatial merge]
    D --> E[Visual vocabulary softmax + four zero channels]
    E --> F[vte + indicator embeddings replace negative IDs]
    F --> G[llm: native causal or hybrid recurrent cache]
    F --> H[Actual first-decoder AWQ capture]
    H --> I[No scale groups: plain quantization]
    I --> J[26 / 30 tiny decoder linears packed]
    J --> K[Safetensors + ProcessorMixin save]
    K --> L[Two reloads: exact packed/dense state and eager logits]
    F -->|unknown ID / missing media / atom mismatch| X[Fail before decoder mutation]
    style X fill:#fdd
```

| Before | After |
| --- | --- |
| Definition inherited checkpoint AutoModel selection | Explicit lazy package paths for both configs/models/processors |
| Source constructor downloaded tokenizer/image processor | Native decoder construction is offline; normalized assets injected |
| Transformers 4 checkpoint class registration | Explicit native Qwen3 MoE/Next nested configs, no global registration |
| FlashAttention-only vision assumption | Real eager and SDPA frame/window attention, no optional CUDA import |
| Indicator/rotary reload initialization could drift | Computed indicator range and deterministic nonpersistent RoPE reset |
| Requested eager lost on quantized nested reload | Propagate selected root backend only into unset child configs |
| Next AWQ override allowed the router | Explicit plain policy; router/shared gate/linear-attention BA remain dense |
| Generic token sanitization clipped visual IDs | Definition preserves IDs; wrapper rejects unknown negatives |

Original `llm.*`, `visual_tokenizer.*`, `vte.*` tensor names and source
pre-post-LN vision feature selection remain intact. Native caches are not
converted into an invented shared format. The source thinking-budget path
is retained, but trained thinking quality and budget-boundary semantics have
not been independently qualified. Frame-list video is supported; no container
decoder is added. Source temporal patch size is one; other values fail closed.

## Validation and reproduction

Validated with Python 3.14.6t (`PYTHON_GIL=0`), Transformers 5.23.0,
Torch 2.15.0+cpu, released Tokenicer 0.0.17. No tokenizer shim is present.

```bash
export PATH=/home/ubuntu/.venvs/gptqmodel-tf5/bin:$PATH
PYTHON_GIL=0 python -m pytest -q tests/test_ovis2_6_moe_local_port.py tests/module_tree/test_ovis2_6_moe_local_port.py
bash format/check.sh
python -m compileall -q gptqmodel/models/ports/ovis2_6_moe gptqmodel/models/definitions/ovis2_6_moe.py
git diff --check
```

Tests cover real CPU/meta owners, every released header shape, MHA/GQA/MQA
and bias, FP32/FP16/BF16 eager/SDPA media forward, native cache continuation
and causality, greedy/beam parity to an independently instantiated decoder,
assignment and two dense safetensors cycles. Independent float64 equations
cover patchification, interpolated positions, coordinate RoPE, ragged multi-frame
full/window attention, MLP, spatial merge, soft visual vocabulary and embedding
insertion. Processor tests independently check actual pixels/patch order,
image/video IDs, serialization and retry. Public loading tests exercise hostile
model/config/tokenizer/processor maps with trust false/true and bypass absent/0/1.
Pinned released tokenizer cases check multilingual IDs/decode/chat and reload.

Four public image-calibrated AWQ cases cover both aliases, offload false/true,
`fallback=None`, actual first-block capture, all 26/30 tiny decoder linears,
every excluded dense tensor, two exact packed-state/logit reloads and processor
persistence. No scale-search/fold safety is claimed. Mutation evidence: multiplying
eager vision attention scale by 1.1 makes five independent-oracle tests fail;
five SDPA controls stay green. Earlier regressions detect lost nested eager,
missing ProcessorMixin persistence and missing Next container exclusion.

## Integration ownership

`central_fixture_proposals.json` provides exact replacement registry cases and
unqualified fusion reasons. Parent owns central fixture edits, main README,
central docs/reference-page integration and PR diagrams; this port-local page
supplies their diagrams/table. Neither alias is in the generic AWQ numerical
`FAMILIES`, so no generic plain-family set change is proposed. Parent must package
LICENSE/NOTICE/provenance plus JSON/gzip fixtures. Shared qualification baseline
copies are separate commits and must not be integrated as owned port changes.

## Limits

No trained multi-GB payload execution or generation quality claim. No GPU,
FlashAttention, optimized FLA, multi-GPU/device-map qualification, concurrent
PyTorch execution, compile, training, fusion executor, extended-context or
foreign/static/offloaded/quantized-cache claim. CPU disk offload here means
LazyTurtle quantization, not a claim about inference-cache offload. Existing
trained model tests use uncached `/monster/data/model` paths and are not run.
