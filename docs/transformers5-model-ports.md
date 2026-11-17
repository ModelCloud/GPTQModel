# GPT-QModel Transformers 5 model ports

Definitions select package-owned implementations through `LocalModelLoader`.
Checkpoint `auto_map`, API trust, and the safety bypass cannot replace these
config/model classes. See [implementation selection](trust-remote-code-isolation.md)
for the complete selection and failure flow.

Definitions with exact native Transformers 5 support retain their native
implementation. Their rows record checkpoint/layout qualification and any
definition ownership corrections alongside the package-owned ports.

| Definition / model type | Implementation and provenance | CPU validation | AWQ policy | Limits |
| --- | --- | --- | --- | --- |
| `ApertusQModel`, `Apertus1p5TextQModel`, `Apertus1p5QModel` / `apertus`, `apertus1p5_text`, `apertus1p5` | [Native Apertus and package-owned 1.5 text/multimodal qualification](../gptqmodel/models/ports/apertus/README.md); pinned Apache-2.0 Swiss AI fork | Native text oracle; full/pruned output vocabularies, bias/activation variants, live CPU/meta ownership, eager/decoder-SDPA cache/generation and safetensors reload, actual image/audio processor and embedding splice, public hostile metadata matrix | Actual CPU plain quantization of every decoder projection across layouts, including failure/retry; no scale folding/search | Official gated checkpoint returned HTTP 401: no trained weights, official tokenizer/chat template or checkpoint key audit; no GPU/FlashAttention/CUDA xIELU, packed GPTQ/AWQ roundtrip or fusion qualification; vision attention remains the source's fixed implementation |
| `AgnesQModel` / `agnes` | [Agnes 3.0 Flash multimodal port and pinned Apache-2.0 provenance](../gptqmodel/models/ports/agnes/README.md) | Full 1506-key checkpoint schema; independent FP64 delta recurrence and image/video patch oracles; mixed/homogeneous attention and parallel FFN layouts, media/cache/beam/failure-reset behavior, real local processor save/reload, public malicious model/processor metadata matrix | Actual plain AWQ quantization for both FFN layouts; delta state/norm/convolution tensors stay dense; no scale groups | No GPU/FP16, trained-checkpoint load/quality, video file decoder or fused FLA/CUDA qualification; recurrent crop/assisted rollback explicitly unsupported |
| `BaiChuanQModel` / `baichuan` | [Baichuan 1/2 source and layout provenance](../gptqmodel/models/ports/baichuan/PROVENANCE.md); pinned Apache-2.0 sources for 7B/13B variants | Independent FP32/FP16/BF16 oracles for all four variants; fused QKV, RoPE/ALiBi and normalized heads, cache/generation, malicious metadata matrix, safetensors round trips and live CPU/meta ownership; both 13B index schemas audited | Actual plain AWQ processor quantizes all five decoder linears; norms and Baichuan 1 persistent rotary buffer stay dense; retry idempotence | No GPU, trained-checkpoint or monolithic 7B payload audit; static/tuple/offloaded caches, AWQ scale search/replay, upstream chat/quantizer helpers and fusion unqualified |
| `BrumbyQModel` / `brumby` | [Manifest AI 14B/4B and retention 1.0.7 provenance](../gptqmodel/models/ports/brumby/PROVENANCE.md); Apache-2.0 | Exact full-size meta/index schemas; portable degree-2 retention, GQA, Q/K norms, dynamic/linear/default RoPE, independent CPU recurrence oracle, cache continuation/compression/rollback, padding, generation, repeated safetensors reload and public trust/bypass isolation | Actual CPU plain AWQ quantization and packing of all seven linears; gates and norms stay dense; no scale groups | No publisher CUDA/Triton-kernel parity, GPU packing or trained-checkpoint quality qualification; KV-only/static/offloaded caches and speculative rollback rejected |
| `AfMoeQModel` / `afmoe` | [Native Afmoe / Trinity qualification](../gptqmodel/models/ports/afmoe/README.md); Transformers 5.23.0 Apache-2.0 implementation; pinned Mini/Nano/Large index schemas and checkpoint license recorded locally | Full meta inventories match all three checkpoint schemas; unequal attention/residual widths; dense/MoE ownership, native conversion and routing oracle, cache/generation, public trust/bypass matrix, safetensors retry/round trips | Tested plain quantization for all declared linears; norms, router and expert bias stay dense | CPU pseudoquantization; no GPU packing/kernel or trained-checkpoint quality qualification; no scaling/folding claim |
| `InternLMQModel` / `internlm` | [InternLM 1 port](../gptqmodel/models/ports/internlm/README.md); Apache-2.0; upstream revision `96e127d08d851a88cac736a9b091dd953ae1b873` | Independent FP32/FP16/BF16 numerical oracle; biased/unbiased origin/dynamic rotary; eager/SDPA; generation/cache/beam lifecycle; public dense load and safetensors reload; live meta-model weight ownership; all 14 linears in two-layer AWQ quantization/reload | Tested plain quantization, zero scale-fold groups | No GPU or trained 7B numerical/quality qualification; fusion and foreign/static/quantized KV caches unsupported |

Only integrated implementations are listed here. This table does not qualify
other definitions, architectures, kernels, or trained checkpoints by analogy.

Run the InternLM checks from the repository root:

```shell
python -m pytest tests/test_internlm_local_port.py tests/module_tree/test_internlm_local_port.py tests/module_tree/test_registered_module_plans.py -q --tb=short
```
