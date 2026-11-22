# LocateAnything clean-room golden fixtures

These fixtures were generated with the current (restricted) port on CPU: torch 2.15.0+cpu,
transformers 5.23.0, `CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=1`. All values are fp32.
The generator script imports restricted code, so it is not committed. Behavior is described in
`cleanroom/locateanything/SPEC.md`. Total size is about 1.2 MB.

Running the generator twice gave byte-identical files, so the reference is deterministic.

## Common tiny configuration (all four model fixtures)
Text: vocab 64, hidden 32, intermediate 64, 2 layers, 4 heads, head_dim 8, rope_theta 1e6,
rms_norm_eps 1e-6, max_position_embeddings 128, block_size 6, causal_attn False,
text_mask_token_id 60, null 61, switch 62, eos (im_end) 2, pad 0.
Vision: MoonViT hidden 16, 2 heads, **2 layers**, intermediate 24, patch 2, pos table 4×4.
Top level: image_token_index 59, box_start 32, box_end 33, ref_start 34, ref_end 35, none 36,
coordinates 16..31.

Weights are seeded random. Norm weights and the connector LayerNorm weight are drawn as
1 + 0.2·N(0,1), and every bias as 0.1·N(0,1), so norm and bias handling is actually exercised.
Each `config.json` and `model.safetensors` is a normal `save_pretrained` checkpoint. Load it
with `LocateAnythingQModel.loader.from_pretrained(dir, dtype=torch.float32, attn_implementation=...)`.

| dir | variant |
|---|---|
| `qwen2_gqa/` | Qwen2 decoder, 4q/2kv heads, untied head |
| `qwen2_mqa_tied/` | Qwen2, 4q/1kv, `tie_word_embeddings=True` (one shared tensor) |
| `qwen3_bias/` | Qwen3 decoder, q_norm/k_norm, `attention_bias=True`; native causal only |
| `qwen2_sliding/` | Qwen2, `use_sliding_window=True, sliding_window=2, max_window_layers=0` (all layers sliding) |

## `<variant>/expected.safetensors`
Keys starting `in.` are inputs. Keys starting `sdpa.` and `eager.` are outputs from the same
checkpoint loaded with that attention implementation. The model is in eval mode unless the key
says train.

| key (suffix after sdpa./eager.) | what it is |
|---|---|
| `text_logits`, `text_hidden_last` | `model(input_ids=in.text_ids)` logits, and decoder final-norm hidden state |
| `mtp_cached_logits` | `language_model(in.mtp_ids, use_cache=True)`: trailing-MASK cached window rule |
| `mtp_uncached_logits` | same input with `use_cache=False`: uncached window rule |
| `mtp_cached_causal{0,1}_logits` | cached rule with `text_config.causal_attn` False / True |
| `multiwin_uncached_causal{0,1}_logits` | `model(input_ids=in.multiwin_ids, use_cache=False)`, two windows |
| `mtp_gen_logits` | generation-style window `in.mtp_gen_ids` with `in.mtp_gen_pos` (last 6 positions lowered by 1), use_cache=True |
| `round0_logits`, `round1_logits` | two-round MTP cache test. Prefix [1,4,6] is cached first. Round r feeds `in.round{r}_full_ids[:, seen:]` with positions `in.round{r}_full_pos[:, seen:]` (seen = `in.round{r}_seen`), then crops the cache by 6 |
| `cont_prefix_logits`, `cont_suffix_logits` | `in.cont_ids[:, :3]` with use_cache, then `[:, 3:]` continued on that cache |
| `batch_logits` | `language_model(in.batch_ids, attention_mask=in.batch_mask, position_ids=in.batch_pos)` (left padded) |
| `patch_embed_out`, `vision_encoder_out` | vision patch projection plus pos table; encoder output after the final LN (before the merge) |
| `vision_merged_{0,1}` | `extract_feature(in.pixel_values, in.image_grid_hws)` per image (grids 2×4 = interpolated table, and 4×4 = table used as is) |
| `mlp1_out` | connector applied to the concatenated merged features |
| `mm_logits` | `model(input_ids=in.mm_ids, pixel_values, image_grid_hws)`, with 6 image tokens |
| `flag_logits` | `in.flag_ids` with `image_flags=in.image_flags` ([0,1]: the first image is skipped) |
| `flag00_logits` | ids [[1,5,7]] (no image tokens), flags [0,0]: equals text-only logits |
| `train_causal{0,1}_logits`, `train_causal{0,1}_loss` | `model.train()`, `model(input_ids=in.train_ids, position_ids=in.train_pos, use_cache=False, labels=in.train_ids)` |
| `sdpa.hf_greedy_ids` | `language_model.generate([[1,5,9,11,6]], max_new_tokens=6, do_sample=False)` |
| `sdpa.hf_greedy_batch_ids` | `language_model.generate(input_ids=in.batch_ids, attention_mask=in.batch_mask, max_new_tokens=4, do_sample=False, pad_token_id=0)` |

`qwen2_sliding` has no `hf_greedy_*` keys. HF generate currently raises with all-sliding layers,
and the error is recorded in `expected_meta.json["hf_generate_error"]`. This is a known
limitation, not a contract.

## `<variant>/expected_meta.json` → `localization_generate`
These are the outputs of `model.generate(input_ids, [pixel_values, image_grid_hws], tokenizer=T,
use_cache=True, max_new_tokens=M, generation_mode=mode)`. T is a stub with
`model_max_length=64`; its `batch_decode` returns comma-joined ids and its `decode` does the same.
Each entry holds:
- `response`: the returned string;
- `mtp_rounds`: one entry per MTP round, with the candidate 6 ids passed to `handle_pattern`, the
  resulting `type` and the appended `tokens`;
- `forward_calls_[input_len,cache_len_after_forward]`: one per language-model forward. The cache
  length is measured right after the forward, before the speculative crop.

| entry | prompt | M |
|---|---|---|
| `text_{slow,fast,hybrid}` | ids [[1,4,6]] | 12 |
| `image_{slow,fast,hybrid}` | `in.mm_ids` plus the two images | 12 |
| `rigged_{slow,fast,hybrid}` | ids [[1,4,6]] on a rigged copy of the checkpoint | 8 |

**Rigged model recipe.** Load the fixture checkpoint (sdpa). In `language_model`, zero every
parameter whose name does not contain "norm". Then set column 0 of
`model.embed_tokens.weight` to 1 and set `lm_head.weight[32, 0] = 1`. For the tied variant
these two writes hit the same tensor, so the rigged result differs (all token 0). That is still
deterministic and is recorded.

## `masks.safetensors`
These are 0/1 **visibility** tables (1 = visible) of the Qwen2 decoder mask for batch row 0,
with shape [L, K]. They come from the tiny config with bs = 6 and MASK = 60.

| key | case |
|---|---|
| `cached_window_9_causal{0,1}` | ids [1,4,5,6,60×5], past 0, use_cache |
| `uncached_window_9_causal{0,1}` | same ids, no cache |
| `cached_window_past5_len7_causal{0,1}` | 7 new ids [7,8,60×5] after 5 cached tokens (K = 12) |
| `uncached_multiwin_17_causal{0,1}` | ids [1,4,6,9,60×5,8,9,10,60×5] |
| `uncached_leading_mask_8_causal{0,1}` | ids [60,60,60,5,6,7,60,60]: the P = 0 fallback edge case |
| `train_18_causal{0,1}` | training mask, positions [0..5, 0..5, 3..8] |
| `train_12_causal{0,1}` | training mask, positions [0..4, 2..8] (split point 5, partial last block) |
| `train_nodrop_7_causal{0,1}` | training mask, positions 0..6 (no drop, so the split point is −1) |
| `cached_window_9_padded_first` | cached rule plus 2-D mask [0,1,1,…]: the window override re-opens nothing outside the window |

## `sampler.json` (SPEC §7)
- `token_ids`: the §7.1 table for the tiny config.
- `cases[]`: crafted [6, 64] logits (`logits`) for legal, empty, illegal, non-coordinate, ambiguous,
  ref, low-start, im_end and null-first frames, plus one random frame. probs are softmax(logits).
  Each case records:
  - `decode_bbox_avg_{hybrid,fast}_{keep4,default_keep5}`: §7.5 with keep_k 4 and with the
    default 5;
  - `frame_class`: the §7.4 classification (empty_box, legal_box or illegal_box);
  - `ref_decode`: the §7.6 result;
  - `select_{hybrid,fast}`: the §7.3 multi-position selection with default kwargs
    (`argmax_ids`, `confidence`, `box`).
- `handle_pattern[]`: 17 `candidate` id lists, with the hybrid and fast results of §7.7.
- `processors`: §7.2 repetition penalty (3-D with ρ = 1.3; 2-D with ρ = 0.7; previous ids include
  out-of-range 99 and −1), top-p 0.5, top-k 3 and 1000 (clamped), and `select_single_*`: §7.3
  on a single position under three greedy kwarg sets (ids, confidence, probs).

## `processor/`
- `tokenizer.json`, `tokenizer_config.json`: a WordLevel tokenizer (no pre-tokenizer), 64 ids.
  `<IMG_CONTEXT>` = 59 is a special token, `[PAD]` = 0, `[UNK]` = 1, `[EOS]` = 2. Plain text runs
  map to UNK, so ids mainly check special-token splitting and padding.
- `processor_expected.safetensors`:
  - `img.<name>`: the uint8 HWC (or HW) source arrays;
  - `iproc.<case>.pixel_values` / `.image_grid_hws`: image-processor outputs;
  - `chat_case.pixel_values`, `video_tensor.*`, `video_tensor01.pixel_values`,
    `batch_two_samples.pixel_values`.
- `processor_expected.json`: the kwargs for each image-processor case; chat template strings;
  expanded text; input_ids and attention_mask; grids; video tensor expansion with timestamps and
  fps; a two-sample padded batch; and `get_video_frame_indices` cases (SPEC §10.4).
  The processor uses `LocateAnythingImageProcessor(patch_size=2)` with this tokenizer, except in
  the `iproc.*` cases, which list their own kwargs. `p2_to_rgb_rgba__d` applies the RGBA→RGB
  white composite first, then the processor. Images are PIL images built with
  `Image.fromarray(img.<name>)`.

## Tolerances achieved by the current implementation
- Same implementation, same environment: **bit-exact** (two generation runs were byte-identical).
- sdpa versus eager on the same checkpoint (max |Δ|, excluding padded rows): Qwen2 ≤ 7.2e-7,
  Qwen3 ≤ 3.6e-7 on logits and features.
- Independent float64 equations (existing tests): within atol 6e-7–8e-7, rtol 2e-5.
- Recommended acceptance for the clean-room port:
  - integer and string outputs (generated ids, responses, masks, grids, sampler and processor
    JSON, input_ids): **exact**;
  - fp32 tensors: `assert_close(atol=1e-5, rtol=1e-5)`. A port that reuses the vendored native
    Qwen layers, keeps the same op order and runs on CPU should get bit-exact decoder outputs.
    Vision may differ by about 1e-6 if equivalent but different ops are used;
  - image-processor pixel_values: exact. They depend on PIL BICUBIC resize and torchvision
    to_tensor/normalize.
- `eager.batch_logits` **batch row 0 is all NaN** for the Qwen2 variants. Its two left-padded
  query rows are fully masked, and the NaN spreads to later layers through V. sdpa stays finite.
  Treat that row as unspecified (compare batch row 1 only, or compare with NaN equality).
  Positions 0–1 of batch row 0 in `sdpa.batch_logits` are padded queries and are also unspecified.
