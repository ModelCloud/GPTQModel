# MiniCPM-o port: clean-room scope

Port directory: `gptqmodel/models/ports/minicpm_o/` (pinned upstream snapshots
`openbmb/MiniCPM-o-4_5@fb09628e` and `openbmb/MiniCPM-o-2_6@cf0b69a6`).

## Evidence

- `NOTICE` (port dir), section "Inherited ChatTTS boundary". It says the OpenBMB 2.6 source attributes
  GFSQ, DVAEDecoder, DVAE, repetition-penalty processing, audio-code generation and mel decoding to
  ChatTTS (`ChatTTS/model/dvae.py`, `processors.py`, `gpt.py`, `core.py`). It also says the 4.5
  repetition processor is AST-identical to the ChatTTS processor and the 4.5 `gen_logits` is
  substantially identical. The historical ChatTTS licenses were CC-BY-NC-ND-4.0, then CC-BY-NC-4.0
  (2024-06-01), then AGPL-3.0 (2024-07-15). No permissive grant was found, and the borrowing
  revision is unknown.
- License texts kept in the port dir: `LICENSE.ChatTTS-AGPL-3.0`, `LICENSE.ChatTTS-CC-BY-NC-4.0`,
  `LICENSE.Vocos-MIT`.
- Source markers: `tts_26.py` line 16 ("Inherited ChatTTS/Vocos code has an unresolved licensing
  boundary"), plus "Borrowed from .../ChatTTS/..." comments above GFSQ, DVAEDecoder, DVAE, the
  repetition processor and `gen_logits`. The `generate` docstring says code was borrowed from
  ChatTTS `gpt.py`, and the `decode_to_mel_specs` docstring says it was borrowed from ChatTTS
  `core.py`. `modeling_minicpmo.py` line 17 has a similar boundary note.
- All files carry OpenBMB Apache-2.0 headers. The ChatTTS-derived parts cannot be relicensed under
  those headers.

## RESTRICTED (replace with a clean-room implementation)

| File | Symbol | Lines (at `41be89361`) | Provenance |
|---|---|---|---|
| `tts_26.py` | `ConvNeXtBlock` | 111-161 | The source comment says "copied from Vocos" (MIT). The shipped text is the ChatTTS-modified copy: it takes a `dilation` argument and an ignored `cond` argument, names the layer scale `coef` (Vocos uses `gamma`), and uses ChatTTS's in-place transpose style. NOTICE lists its closest match as ChatTTS `dvae.py @ cff3b4e5` (AGPL). **Classification: restricted.** The MIT Vocos origin does not cover the ChatTTS changes, so replace it. Because the block is a standard ConvNeXt-1D design, it is easy to reimplement from the spec. |
| `tts_26.py` | `GFSQ` | 164-219 | ChatTTS `dvae.py` (AGPL) |
| `tts_26.py` | `DVAEDecoder` | 222-264 | ChatTTS `dvae.py` (AGPL) |
| `tts_26.py` | `DVAE` | 267-343 | ChatTTS `dvae.py @ e69ffacf` (AGPL) |
| `tts_26.py` | `CustomRepetitionPenaltyLogitsProcessorRepeat` | 450-475 | ChatTTS `processors.py @ 52d75d54` (CC-BY-NC-4.0) |
| `tts_26.py` | `ConditionalChatTTS.generate` | 817-1047 | ChatTTS `gpt.py` (docstring) |
| `tts_26.py` | `ConditionalChatTTS.decode_to_mel_specs` | 1049-1084 | ChatTTS `core.py` (docstring) |
| `tts_26.py` | `gen_logits` | 1087-1104 | ChatTTS `processors.py @ 52d75d54` |
| `modeling_minicpmo.py` | `CustomRepetitionPenaltyLogitsProcessorRepeat` (4.5 copy) | 4847-4869 | AST-identical to the ChatTTS processor (NOTICE) |
| `modeling_minicpmo.py` | `gen_logits` (4.5 copy) | 4872-4885 | Substantially identical to ChatTTS (NOTICE) |

`utils.py` defines no restricted symbol. Its `TTSStreamingGenerator` only *consumes* processors
and warpers built by the 4.5 `gen_logits` (see SPEC section 7.3), so `utils.py` stays as is.

After replacement, the two ChatTTS license files and the ChatTTS section of `NOTICE` can be
removed or rewritten if nothing ChatTTS-derived remains. `LICENSE.Vocos-MIT` can also go once
`ConvNeXtBlock` is reimplemented. That decision belongs to whoever lands phase 2.

## PERMISSIVE (OpenBMB Apache-2.0 or other permissive; keep)

- `tts_26.py`: `ConditionalChatTTSConfig`, `apply_spk_emb`, `make_streaming_chunk_mask_generation`,
  `ConditionalChatTTSGenerationOutput`, `MultiModalProjector`, and these parts of
  `ConditionalChatTTS`: the class itself, `__init__`, `merge_inputs_embeds`, `prefill_text` and
  `prefill_audio_ids`. `__init__` builds `DVAE()` and the weight-normed `head_code`; these are
  interface points for the replacement.
- `modeling_minicpmo.py`: everything except the two symbols above.
- `modeling_minicpmo_26.py`, `configuration_minicpmo.py`, `processing_minicpmo*.py`,
  `tokenization_minicpmo_fast.py`, `modeling_navit_siglip.py`, `utils.py`, `utils_26.py`,
  `cache.py`, `native.py`, `__init__.py`.
- `vector-quantize-pytorch` (`GroupedResidualFSQ`, MIT, Phil Wang) is an *installed dependency*,
  not vendored code. The clean-room GFSQ may keep using it.

## Flag for legal review (outside the assigned scope; not specified here)

The NOTICE does not attribute these OpenBMB 4.5 audio-code sampling loops, but they follow the same
step structure as ChatTTS `gpt.py` generation: per-codebook head loop, flattening to
(batch x codebook, vocab), temperature division, then processors, then warpers, then the eos
`-inf` mask, then multinomial sampling.

- `modeling_minicpmo.py`: the generation loops of `MiniCPMTTS` (around lines 4300-4345,
  4530-4545 and 4695-4712).
- `utils.py`: `TTSStreamingGenerator`'s sampling step (around lines 760-800).

The caller kept these as OpenBMB Apache-2.0. This note only records the resemblance.
