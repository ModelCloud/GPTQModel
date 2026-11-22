# MiniCPM-o clean-room golden fixtures (ChatTTS-derived TTS pieces)

These were generated with the current (pre-clean-room) port at commit `41be89361`. Environment:
torch 2.15.0+cpu, transformers 5.23.0, vector-quantize-pytorch 1.34.1, float32, CPU,
`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`. The generator imports restricted code, so it is
not committed. Two generations in fresh processes gave byte-identical files. Total size is about
310 KB. The spec is `cleanroom/minicpm_o/SPEC.md`.

## Weights

Small modules ship their weights inside their `.safetensors` file under a `w.` prefix. Load them
with `module.load_state_dict({k[2:]: v for k, v in f.items() if k.startswith("w.")})`.

The full `DVAE` (~13M params, fixed geometry) and the tiny `ConditionalChatTTS` (whose Llama
`embed_tokens` alone is 32000x16) are too big to ship. Their weights are **regenerated** by
`weights_recipe.py` (torch only, no port imports) from `dvae.manifest.json` and
`tts.manifest.json`. Each manifest holds a seed, every state_dict key with its shape, and a
SHA-256 digest that `weights_recipe.load(name)` checks. The recipe: walk the keys in sorted order,
draw `randn` from `Generator().manual_seed(seed + i)` for the i-th key, then map the draw. Mel
`coef` becomes 1+0.1z, ConvNeXt `coef` becomes 0.5+0.1z, norm weights become 1+0.1z,
biases and 1-D tensors become 0.02z, and everything else becomes z/sqrt(fan_in). Load the result
with `load_state_dict(strict=True)` into a freshly constructed module. Do **not** use
`from_pretrained`; see SPEC section 10 for the defect with non-persistent quantizer buffers.

## Files

| File | Covers | Contents |
|---|---|---|
| `processors.safetensors` + `processors.json` | SPEC 6, 7 | Seven repetition-penalty cases: penalties 1.5, 0.5, 2.0 and 1.3; windows 16, 4 and 100; history longer than, shorter than and equal to empty versus the window; the row cutoff `max_input_ids=3`; scores with 0, -inf and +inf entries. Constructor ValueError cases with their message. Six `gen_logits` cases (defaults, rep-only, P+K+rep, K-only, P-only, all None), each giving the built list description and the output after applying processors and then warpers to `gl.logits` with history `gl.hist`. The 2.6 and 4.5 copies were checked to give identical outputs. |
| `convnext_ls.safetensors` | SPEC 2 | dim 8, inter 32, k7, dilation 2, layer scale present; input (2,8,13) and output |
| `convnext_k3d1.safetensors` | SPEC 2 | dim 6, inter 12, k3, dilation 1 |
| `convnext_noscale.safetensors` | SPEC 2 | dim 8, inter 16, k5, dilation 3, `layer_scale_init_value=0` (no `coef` key) |
| `dvae_decoder.safetensors` | SPEC 3 | idim 6, odim 5, n_layer 2, bn_dim 4, hidden 8, k7, d2; input (2,6,11) and output (2,5,11). Passing a conditioning argument gives the same output. |
| `gfsq.safetensors` | SPEC 4 | GFSQ(1024,(5,5,5,5),2,2). Covers codes to features (`embed.codes` (2,4,7), including all-0 and all-624 frames, giving `embed.output` (2,1024,7)) and features to codes (`forward.input` (2,1024,7) giving `forward.indices` (2,4,7) int32). |
| `dvae.manifest.json` + `dvae_io.safetensors` | SPEC 5 | Decode of (1,4,6) to (1,100,12) and of (2,4,5) to (2,100,10). Encode of unbatched mel (100,12) to (1,4,6) and of (100,9) to (1,4,4) int32. `roundtrip.mel` is the decode of the encode. |
| `tts_config.json` + `tts.manifest.json` + `tts_io.safetensors` | SPEC 8, 9 | Tiny ConditionalChatTTS: hidden 16, 1 layer, 2 heads, V=16 (eos 15), K=4, reserved text length 8, so C=11; eager attention. The inputs `inputs.text_ids` / `inputs.spk_hidden` built the starting caches with the kept `prefill_text`. Each scenario stores its exact starting cache (`sX.in.cache.0.{keys,values}`) and inputs, so the kept prefill code is not needed to replay them. |
| `fixtures_meta.json` | all | Shapes, constructor arguments and per-scenario metadata |
| `weights_recipe.py` | | Weight regeneration (see above) |

### Generation scenarios (`tts_io.safetensors`, prefix `sN.`; metadata in `fixtures_meta.json["tts"]`)

Each scenario calls `torch.manual_seed(seed)` immediately before `generate(...)`. Its stored
outputs are `out.new_ids`, `out.audio_input_ids`, `out.cache.0.*` (the cache after the call) and
`out.probs`. `out.probs` stacks the matrix passed to the single `torch.multinomial` call of every
step, with shape (steps, K, V). Unless noted, warpers and processors come from
`gen_logits(16, top_P=0.7, top_K=5, repetition_penalty=1.5)`, the temperature is
`[0.7, 1.0, 0.7, 1.0]` and `eos_token=tensor([15])`.

| id | Situation | seed | min/max new | in len, then out len / new_ids len / cache len | finished |
|---|---|---|---|---|---|
| s1 | first call, budget exhausted | 9 | 2 / 6 | 11 then 17 / 6 / 16 | False |
| s2 | continuation of s1 after 2 more text tokens prefilled (mask updated), `force_no_stop=True` | 77 | 0 / 5 | 17 then 22 / 11 / 21 | False |
| s3 | eos drawn mid-run; eos row dropped from new_ids | 0 | 2 / 8 | 11 then 17 / 5 / 16 | True |
| s4 | eos at step 0 on a first call; empty new_ids, cache 11 not 10 | 3 | 0 / 4 | 11 then 11 / 0 / 11 | True |
| s5 | eos at step 0 on a continuation (start = s1 state with seed 9); previous last row dropped | 1 | 0 / 3 | 17 then 17 / 5 / 17 | True |
| s6 | no warpers or processors, int eos 15, `force_no_stop`, temperature all 1 | 123 | 0 / 5 | 11 then 16 / 5 / 15 | False |
| s7 | one-element temperature `[0.5]` | 9 | 3 / 4 | 11 then 15 / 4 / 14 | False |

`mel.a.*` is `decode_to_mel_specs(s1.out.new_ids)`, giving (1,100,12). `mel.b.*` is
`decode_to_mel_specs([codes0 (3,4), codes1 (5,4)])`, giving (2,100,10) with zero-code right padding.
`decode_to_mel_specs([])` returns `numpy.ndarray`, float32, shape (0,). A zero-length element
raises (recorded in `fixtures_meta.json`).

## Tolerances achieved by the current implementation

- Every file is **bit-exact** when the current implementation is rerun (CPU, fp32, 1 thread).
- An independent re-derivation written only from SPEC.md, using plain `F.conv1d`, `F.linear`,
  `F.layer_norm` and `F.gelu`, was **bit-exact** (atol=0, rtol=0) on all repetition cases, all
  three ConvNeXt cases, both GFSQ directions and both full-DVAE decodes.
- Recommended assertions for the clean-room implementation:
  - Integer outputs (codes, `new_ids`, `audio_input_ids`, indices) and shapes, `finished`, cache
    lengths, and the returned types: exact.
  - Processor and warper outputs, ConvNeXt, the stack, GFSQ features, mels and caches:
    `atol=0, rtol=0` is expected. Use `atol=1e-5, rtol=1e-5` only if the implementation
    deliberately changes op order.
  - `out.probs`: `atol=1e-6`. Sampled ids must match exactly. If they do not, compare `out.probs`
    first, because a 1-ulp logits change can flip a multinomial draw.
