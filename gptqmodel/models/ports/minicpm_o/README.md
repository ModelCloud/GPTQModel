# MiniCPM-o local Transformers 5 port

The port includes modeling code and checkpoint metadata. The [NOTICE](NOTICE)
and [third-party provenance](third_party_provenance.json) record the OpenBMB,
ChatTTS, Vocos and Transformers sources and their license texts.

This GPT-QModel port owns the exact `minicpmo` Omni architecture, not the
unrelated native MiniCPM-V-4.6 architecture. Transformers 5.23.0 has no
native `minicpmo` config/model mapping.

| Snapshot | Language decoder | Preserved Omni owners | Pinned inventory |
| --- | --- | --- | --- |
| [4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5/tree/fb09628e98b1f55d8d5b863e0eccfcdd11bff499) | Native Qwen3 | NaViT SigLIP, resampler, Whisper, projection, MiniCPMTTS | 1,414 keys/shapes |
| [2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6/tree/cf0b69a6118cd3b2c90bb37f3f4366e2fba802ab) | Native Qwen2 | NaViT SigLIP, resampler, Whisper, projection, ConditionalChatTTS/DVAE | 1,599 keys/shapes |

The complete inventories were independently read from all eight released
safetensors shard headers using byte-range requests. No trained tensor
payloads were downloaded. Tests instantiate the actual complete released
architectures on meta and require exact key/shape equality, including the
TTS owners. The configs, inventories, source SHA-256 values and header
hashes are included beside this document.

## Ownership and loading

`MiniCPMOQModel.loader` selects the packaged config, model, tokenizer and
processor through `LocalModelLoader`. There is no global Auto registry
registration and no global Transformers cache monkeypatch. Both API trust
values and all tested bypass settings leave the package-owned selection
unchanged. Tokenicer normalizes the owned tokenizer once through the shared
loader; the port adds no corrective tokenizer normalization.

```mermaid
flowchart TD
  A[Checkpoint minicpmo metadata] --> B{"Before: trust remote code"}
  B -->|enabled| C[Checkpoint AutoModel and AutoProcessor Python]
  B -->|disabled| D[No exact native Omni mapping]
  C --> E["Remote Qwen and custom vision/audio/TTS"]
  C --> F[Definition compatibility hooks]
  F --> G[Global Cache and DynamicCache replacement]
```

```mermaid
flowchart TD
  A[Checkpoint minicpmo metadata] --> B[Definition-owned LocalModelLoader]
  B --> C["Packaged config validates 4.5 or 2.6"]
  C --> D{Omni version}
  D -->|4.5| E[Native Qwen3 plus owned Omni modules]
  D -->|2.6| F[Native Qwen2 plus owned legacy Omni modules]
  B --> G["Owned tokenizer and image/audio processor"]
  G --> H[Tokenicer normalization]
  E --> I[Instance-owned Transformers 5 cache adapter]
  F --> I
  I --> J[Real media forward and generation]
  E --> K["Plain AWQ: decoder projections only"]
  F --> K
  K --> L[Safetensors save and reload]
  M[Checkpoint trust and auto_map hints] -. cannot override .-> B
```

| Behavior | Before | Local port |
| --- | --- | --- |
| Runtime ownership | Checkpoint Python | Packaged config/model/processor/tokenizer |
| Trust/bypass | Could select remote implementation | Cannot replace local ownership |
| Decoder | Legacy Qwen source assumptions | Exact native Qwen3/Qwen2 owners |
| Cache | Global legacy replacements | Instance-owned TF5 DynamicCache adapter |
| 2.6 media tags | Shared 4.5-style placeholders | Version-specific parenthesized tags |
| Serialization | Unqualified derived buffers | Deterministic resampler and FSQ reconstruction |
| AWQ | Generic inherited folding assumptions | Explicit plain quantization, no scaling groups |
| TTS provenance | OpenBMB source attributions | Original attributions and historical third-party license texts |

## Adaptations and evidence

- Preserve `llm.*`, `vpm.*`, `resampler.*`, `apm.*`,
  `audio_projection_layer.*` and `tts.*` checkpoint names.
- Construct nested TTS Llama using its own 32,000-entry vocabulary,
  not the outer Qwen vocabulary.
- Convert only instance cache representations; never mutate
  `transformers.cache_utils.Cache` or `DynamicCache`.
- Use local relative imports, including processor setup in chat/duplex helpers.
  Optional external codec/audio dependencies are imported only on demand.
  StepAudio initialization requires a caller-selected local `model_dir`,
  `trust_remote_code=True`, and `GPTQ_BYPASS_SAFETY_CHECKS=1`. Audio duplex
  setup accepts the same arguments or reuses an explicitly initialized codec.
- Rebuild deterministic nonpersistent resampler positions after materialization.
  FSQ layers retain constructor levels so LazyTurtle can rebuild their
  nonpersistent buffers after offload; shared loading code is unchanged.
- Preserve original OpenBMB/Google/Hugging Face headers and borrowed-code
  comments. No clean-room or relicensing claim is made.

### CPU qualification

Integrated validation uses Python 3.14.6t, `PYTHON_GIL=0`, Torch 2.15.0+cpu,
Transformers 5.23.0, released Tokenicer 0.0.20 and
`vector-quantize-pytorch==1.27.15`. The earlier standalone port used
Tokenicer 0.0.17.

The owned suite covers:

- Actual public package-owned loading with hostile config/tokenizer/image/audio
  processor maps; trust false/true and bypass unset/`0`/`1`.
- MHA/GQA/MQA and bias/no-bias decoder layouts for both versions.
- Native decoder numerical oracle, eager/SDPA, cache 2+4 continuation,
  greedy and beam generation.
- Independent float64 equations for NaViT vision, Whisper audio and
  resampler attention in FP32/FP16/BF16.
- Real image/audio/mixed-media conditioning and generation; image/audio
  chat using chunk attention, version-specific processor syntax.
- Media safetensors round trips in all three floating dtypes; exact state
  and logits; native TTS Llama 1+3 cache continuation and serialization.
- Invalid config, malformed processor metadata and wrong-shaped tensor
  failure followed by repaired public-load retry.
- Four public `CalibrationConfig` plain-AWQ image-calibration runs: both versions,
  offload false/true, `fallback=None`, 14 packed decoder linears,
  independent image-expanded first-decoder-input comparison, exact dense
  exclusions, inference and two exact safetensors save/reload cycles.

The combined checks exercise the actual registered module plan and fusion
exclusion. The trained MiniCPM-o 4.5 test requires the local checkpoint at
`/monster/data/model/MiniCPM-o-4_5`.

Pinned real checkpoint tokenizer/preprocessor assets were also checked
offline for both releases, outside the tiny fixture suite: IDs for Latin,
CJK and media/control tokens match direct raw `tokenizers.Tokenizer`
results, and real image/audio inputs survive processor save/reload exactly.
Those large tokenizer assets are not committed.

### Red/green checks

1. Original definition selected `AutoModel`, failing the owned-loader
   regression; the direct definition binding makes it pass.
2. A nonzero resampler query exposed uninitialized derived positions after
   safetensors reload: 6,144/11,776 differing logits, maximum absolute error
   0.28322792. Rebuilding deterministic positions makes all 18 media/dtype/
   version cases exact after reload.
3. 2.6 public AWQ with TTS and offload left all four FSQ codebook owners on
   meta. Retaining constructor levels fixes restoration; all four AWQ
   version/offload combinations now pass without shared loader edits.

## Reproduction

From the repository root:

```sh
./format/check.sh
python -m compileall -q \
  gptqmodel/models/ports/minicpm_o \
  gptqmodel/models/definitions/minicpm_o.py \
  tests/test_minicpm_o_local_port.py \
  tests/module_tree/test_minicpm_o_local_port.py
git diff --check
```

With the `minicpm-o` and `test` extras installed:

```sh
PYTHON_GIL=0 python -m pytest -q -o addopts= -o log_cli=false -ra \
  tests/test_minicpm_o_local_port.py tests/module_tree/test_minicpm_o_local_port.py
```

Run the shared baseline against the registered fixtures:

```sh
PYTHON_GIL=0 python -m pytest -q -o addopts= -o log_cli=false -ra \
  tests/module_tree/test_registered_module_plans.py \
  tests/module_tree/test_fusable_models.py \
  tests/test_local_model_loading.py tests/test_local_processor_loading.py \
  tests/test_local_tokenizer_loading.py tests/test_awq_scale_search_safety.py
```

Full Ruff 0.14.2, compileall and diff checks pass.
Port-wide default Pyright was also run: **826 errors and 7 warnings**,
including inherited dynamic module/output typing and missing optional
codec/audio imports. The port is **not type-check clean**.

## AWQ and dependencies

No source/consumer folding, scale search or Omni replay safety is claimed.
`awq_get_modules_for_scaling` deliberately returns `[]`. Real quantization,
packing and inference are tested through the public API with `fallback=None`.
Only the seven decoder projections per layer are quantized; Q/K norms,
vision, audio, resampler and TTS remain dense. Fusion is not admitted.

Install `gptqmodel[minicpm-o]` to load MiniCPM-o 2.6 with its TTS module.
The extra provides `vector-quantize-pytorch>=1.27.15` for the legacy FSQ
quantizer. File-based audio helpers can additionally use `librosa` and
`soundfile`.

## Unqualified boundaries

No trained weight payload, output quality, GPU, FlashAttention, fused kernel,
multi-GPU, compile, training, long-context or concurrent model execution was
qualified. GIL=0 CPU tests are not evidence of threaded PyTorch safety.

Text/image/audio forward and text generation are qualified, not complete
streaming duplex behavior or speech waveform generation. TTS state owners
and native Llama continuation are tested; neural codecs, StepAudio2,
BigVGAN, 2.6 Vocos assets and DVAE codec numerics are not qualified.
Legacy Vocos loading and container video decoding fail explicitly rather
than importing unowned checkpoint helpers. Decoded PIL frames/waveform
arrays may be supplied where supported; arbitrary container decoding,
foreign/static/offloaded/quantized caches and arbitrary media beam
expansion have no qualification claim.

The trained `tests/models/test_minicpm_o_4_5.py` now requests
`trust_remote_code=False` and retains its real image-chat test when the
checkpoint exists; absence of that payload is an explicit skip.
