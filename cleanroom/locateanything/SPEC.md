# LocateAnything behavioral specification (clean-room phase 1)

Target: Transformers 5.23, torch 2.15, using GPT-QModel's vendored Apache-2.0 blocks
(`gptqmodel/models/ports/native/qwen2`, `native/qwen3`). Golden data is in
`tests/fixtures/cleanroom/locateanything/` (see its README). Section numbers are referenced from
SCOPE.md and the fixture README.

Words used here: **D_v** is the vision width, **H_v** the vision head count, **d_v = D_v/H_v**, **D**
the text hidden size, **V** the vocabulary size, **bs** the block size (`text_config.block_size`,
default 6), **MASK** = `text_config.text_mask_token_id`.

LocateAnything is a VLM. It has a MoonViT-style vision encoder, a 2x2 patch-merge connector MLP,
and a Qwen2 (or Qwen3) decoder. The Qwen2 decoder is trained for **block-diffusion multi-token
prediction (MTP)**: after the real prefix, the model is given one copy of the last real token
followed by bs-1 MASK tokens. That window attends bidirectionally inside itself and predicts bs
tokens at once. A localization decoding loop turns these predictions into box, point and reference
token groups.

---------------------------------------------------------------------------------------------------

## 1. Public interface

### 1.1 Registry and type names
- `model_type` `"locateanything"` (top config). The vision sub-config `model_type` is `"moonvit"`.
  `"locateanything"` must **not** be registered in Transformers' `CONFIG_MAPPING`. MODEL_MAP key
  `"locateanything"` maps to `LocateAnythingQModel`, and that is the only key mapping to it.
- Class names and dotted paths that the definition and tests use:
  - `gptqmodel.models.ports.locateanything.configuration_locateanything.LocateAnythingConfig`
  - `...configuration_locateanything.MoonViTConfig`
  - `...modeling_locateanything.LocateAnythingForConditionalGeneration`
  - `...processing_locateanything.LocateAnythingProcessor`
  - `...image_processing_locateanything.LocateAnythingImageProcessor`
  - `...generate_utils.decode_bbox_avg`, `...generate_utils.handle_pattern` (signatures in §7)
- `type(config.text_config)` must be **exactly** the vendored
  `gptqmodel.models.ports.native.qwen2.configuration_qwen2.Qwen2Config`, or `native.qwen3...Qwen3Config`.
  No class in the language model's MRO may come from a module under `transformers.models.`.
- Checkpoint `architectures`: `["LocateAnythingForConditionalGeneration"]`.

### 1.2 Model object attributes the definition and tests rely on
- `model.vision_model`: the vision tower. It has `.encoder` and `.patch_embed` children, and `.device`
  and `.dtype` properties (a PreTrainedModel).
- `model.mlp1`: the connector, a 4-element sequential container (indices 0, 1, 2, 3; index 2 has
  no parameters).
- `model.language_model`: the causal LM. It has `.model` (with `.embed_tokens`, `.layers`, `.norm`,
  `.rotary_emb`) and `.lm_head`, and it supports HF `.generate`.
- `model.image_token_index` (int), `model.token_ids` (dict, §7.1), `model.extract_feature(pixel_values, image_grid_hws)`.
- `get/set_input_embeddings` and `get/set_output_embeddings` delegate to the language model.
  `get_decoder` and `set_decoder` delegate too.
- Class attributes: `base_model_prefix = "model"`, `main_input_name = "input_ids"`,
  `supports_gradient_checkpointing = True`, `_supports_sdpa = True`, `_supports_flash_attn = False`,
  `_skip_keys_device_placement = "past_key_values"`. The instance-level `_no_split_modules` is
  `["Qwen3DecoderLayer"]` when the decoder architecture string contains "Qwen3", and
  `["Qwen2DecoderLayer"]` otherwise.
- Tied weights: `_tied_weights_keys = {"language_model.lm_head.weight": "language_model.model.embed_tokens.weight"}`.
  When `text_config.tie_word_embeddings` is true, the two parameters must be **the same object**
  after construction and after loading. When it is false they are separate. `lm_head.weight` is
  always present in `state_dict()`.
- The decoder's embedding has `padding_idx = text_config.pad_token_id`, as in native Qwen2.

### 1.3 State-dict keys (exact)
Vision (`vision_model.`), with L_v = `vision_config.num_hidden_layers` and p = patch size:
- `patch_embed.proj.weight` [D_v, 3, p, p], `patch_embed.proj.bias` [D_v]
- `patch_embed.pos_emb.weight` [init_pos_emb_height, init_pos_emb_width, D_v]
- for each i < L_v, `encoder.blocks.{i}.`:
  `norm0.weight`/`.bias` [D_v], `norm1.weight`/`.bias` [D_v],
  `wqkv.weight` [3·D_v, D_v], `wqkv.bias` [3·D_v], `wo.weight` [D_v, D_v], `wo.bias` [D_v],
  `mlp.fc0.weight` [I_v, D_v], `mlp.fc0.bias` [I_v], `mlp.fc1.weight` [D_v, I_v], `mlp.fc1.bias` [D_v]
- `encoder.final_layernorm.weight`/`.bias` [D_v]

Connector:
- `mlp1.0.weight`/`.bias` [4·D_v] (LayerNorm)
- `mlp1.1.weight` [D, 4·D_v], `mlp1.1.bias` [D]
- `mlp1.3.weight` [D, D], `mlp1.3.bias` [D]

The connector input width is hard-wired to 4·D_v, whatever `merge_kernel_size` says.

Decoder (`language_model.`): exactly the vendored native Qwen2ForCausalLM or Qwen3ForCausalLM key
set under `language_model.` (`model.embed_tokens.weight`, `model.layers.{i}.input_layernorm.weight`,
`self_attn.{q,k,v}_proj.{weight,bias}`, `self_attn.o_proj.weight`,
`post_attention_layernorm.weight`, `mlp.{gate,up,down}_proj.weight`, `model.norm.weight`,
`lm_head.weight`. Qwen3 adds `self_attn.q_norm.weight` and `self_attn.k_norm.weight` [head_dim].
For Qwen3, the q/k/v/o biases follow `attention_bias`).

Released checkpoint (config in `tests/fixtures/ports/locateanything/checkpoint_config.json`):
770 tensors. Building it on the meta device must give a sorted "name:comma-joined-shape" inventory,
newline-joined, whose sha256 is
`ad24aadd8935a6853b5cb17c1b0c1daf8a4a895764ab8c44ae34781aba7aeca1`. The embedding is
[152681, 2048], and lm_head is tied.

`wqkv` layout: its output axis is (3, H_v, d_v) row-major. Rows [0, D_v) are q, [D_v, 2D_v) are
k, [2D_v, 3D_v) are v. Inside each block, head h owns rows [h·d_v, (h+1)·d_v).

---------------------------------------------------------------------------------------------------

## 2. Configuration

### 2.1 `MoonViTConfig` (a PretrainedConfig; extra kwargs go to the base)
| field | default |
|---|---|
| patch_size | 14 |
| init_pos_emb_height | 64 |
| init_pos_emb_width | 64 |
| num_attention_heads | 16 |
| num_hidden_layers | 27 |
| hidden_size | 1152 |
| intermediate_size | 4304 |
| merge_kernel_size | (2, 2), stored as a **list** |

### 2.2 `LocateAnythingConfig`
`sub_configs = {"vision_config": MoonViTConfig, "text_config": Qwen2Config}`.

| field | default | notes |
|---|---|---|
| vision_config | None → empty dict | accepts a config object or a dict |
| text_config | None → empty dict | accepts a config object or a dict |
| use_backbone_lora | 0 | when non-zero, wraps the vision tower with PEFT LoRA (r = value, alpha = 2r, dropout 0.05) at construction |
| use_llm_lora | 0 | same for the LM (q/k/v/o, gate/up/down; task CAUSAL_LM) |
| downsample_ratio | 0.5 | stored only |
| template | None | stored, and copied to `model.template` |
| loss_version | "v1" | stored only |
| mlp_checkpoint | False | stored, and copied to `model.mlp_checkpoint` |
| image_token_index | 151667 | the released checkpoint uses **151665** |
| box_start_token_id | 151668 | |
| box_end_token_id | 151669 | |
| coord_start_token_id | 151677 | inclusive |
| coord_end_token_id | 152677 | inclusive |
| ref_start_token_id | 151672 | |
| ref_end_token_id | 151673 | |
| none_token_id | 4064 | |

Construction rules, in order:
1. Read the attention implementation from kwarg `attn_implementation`, falling back to
   `_attn_implementation`. Both are removed from kwargs. `"magi"` becomes `"sdpa"`. Any value
   other than None, "eager" or "sdpa" raises **ValueError** whose message contains
   "supports eager or sdpa". (This is how `flash_attention_2` is rejected.)
2. The base config is initialised with the remaining kwargs (for example `auto_map`, or a stray
   top-level `rope_parameters`). A top-level `rope_parameters` must **not** affect
   `text_config.rope_parameters`.
3. `config._attn_implementation` = the chosen implementation, or "sdpa" when none was given.
4. Vision: convert to a dict. A `model_type` other than "moonvit" (when present) raises
   ValueError "LocateAnything requires MoonViT vision_config". Build a `MoonViTConfig`.
5. Text: convert to a dict. The architecture is `architectures[0]`, defaulting to
   "Qwen2ForCausalLM". It must be Qwen2ForCausalLM (vendored Qwen2Config) or Qwen3ForCausalLM
   (vendored Qwen3Config). Anything else raises ValueError
   `"Unsupported LocateAnything decoder: <name>"`. Write `architectures = [name]` back, drop any
   `_attn_implementation` key, and build the text config. Extra keys (`block_size`,
   `causal_attn`, `text_mask_token_id`, `null_token_id`, `switch_token_id`) survive as attributes.
6. Set both sub-configs' `_attn_implementation` to the top-level value.
7. Store the remaining fields. Set `tie_word_embeddings = text_config.tie_word_embeddings`.

The config must round-trip through `save_pretrained` / `from_pretrained`. Loading through
`gptqmodel.utils.hf.load_model_config(path, trust_remote_code=..., attn_implementation="eager")`
must give a `LocateAnythingConfig` (never checkpoint Python, even when `auto_map` points at
hostile modules and `GPTQ_BYPASS_SAFETY_CHECKS` is set) with `_attn_implementation == "eager"`.

The model class must also accept `attn_implementation` through the HF loading path with the same
mapping (`magi`→`sdpa`, others rejected with the same message). Model `_init_weights`: Linear,
Conv2d and Embedding weights are drawn from normal(0, std), where std is `config.initializer_range`
if present and otherwise `text_config.initializer_range`. Biases are zeroed, and the Embedding
padding row is zeroed. (Only random-init tests depend on this. The fixtures ship weights.)

Text-config attributes the runtime reads, with their fallbacks: `block_size` (6), `causal_attn`
(False), `text_mask_token_id` (151676), `null_token_id` (152678), `switch_token_id` (152679),
`eos_token_id` (151645, used as **im_end**), `layer_types`, `sliding_window`.

---------------------------------------------------------------------------------------------------

## 3. Vision tower (MoonViT variant), model-specific

Inputs: `pixel_values` of shape [N_patches, 3, p, p], with all images' patches concatenated in
image order and each image's patches in row-major (grid-row, grid-col) order. `grid_hws` is an
integer tensor [N_img, 2] holding (h, w) in patches.

1. **Patch projection**: a conv with kernel = stride = p, applied to each p×p patch. This is a
   linear map 3·p·p → D_v with bias. Output [N_patches, D_v].
2. **Learned absolute position table** `pos_emb.weight` [H0, W0, D_v]. For each image: if
   (h, w) == (H0, W0), use the table as is. Otherwise resize it to (h, w) with torch **bicubic**
   interpolation (channels-first, `align_corners=False`, no antialias). Flatten row-major over
   (h, w) and add to that image's tokens.
3. **Encoder blocks** (pre-norm, for each block):
   - x ← x + Wo · Attn(RoPE2D(q, k), v), where q, k, v come from `wqkv(LN0(x))` (Linear with
     bias). LN is torch LayerNorm with eps 1e-5 and affine parameters.
   - x ← x + fc1(GELU_tanh(fc0(LN1(x)))). GELU here is the tanh approximation.
   - Attention is **non-causal**, restricted to tokens of the same image (block-diagonal over
     images), with scale 1/√d_v.
     - "sdpa": torch SDPA with a boolean visibility mask.
     - "eager": scores = (q·kᵀ)/√d_v in the input dtype, invisible entries filled with −inf,
       softmax computed in float32 and cast back to the input dtype, then multiplied by v.
   - Heads are concatenated head-major before `wo`.
4. After the last block, apply `final_layernorm` (LayerNorm, eps 1e-5).
5. **2D rotary embedding**, shared by all blocks and heads. It requires d_v divisible by 4, a
   frequency base of 10000, and a maximum grid of 512×512 (an assertion fails if any h or w is
   outside [1, 512]).
   - For j = 0 … d_v/4 − 1, the inverse frequency is ω_j = 10000^(−4j/d_v).
   - For a patch at grid row y and column x, build a complex vector of length d_v/2. Element 2j is
     cis(x·ω_j) (the **column** angle comes first). Element 2j+1 is cis(y·ω_j).
   - Each q or k head vector of length d_v is read as d_v/2 consecutive (real, imag) pairs. Each
     pair is multiplied as a complex number by the matching element, then written back
     interleaved. This is computed in float32 and cast back to the input dtype.
   - Rotation tensors are rebuilt on every call on the device of `grid_hws`. There is no cached
     buffer and nothing appears in the state dict.
6. **Patch merger** (kernel kh × kw = `merge_kernel_size`, normally 2×2). For each image, view
   its tokens as [h/kh, kh, w/kw, kw, D_v]. Output tokens are in row-major order over the
   (h/kh, w/kw) merged grid. Each output token is the concatenation of its kh·kw source patches,
   ordered (dy, dx) row-major, giving width kh·kw·D_v. h and w must be divisible by the kernel.
   The tower returns a **Python list** with one tensor [h·w/(kh·kw), kh·kw·D_v] per image, and
   `extract_feature` returns that list unchanged.
7. The vision attention implementation follows `vision_config._attn_implementation` ("sdpa" when
   unset). A `flash_attention_2` request without flash-attn installed falls back to sdpa. In
   practice the config already forbids it.

## 4. Connector (`mlp1`)
y = Linear₃(GELU_erf(Linear₁(LayerNorm(x)))). LayerNorm has width 4·D_v and eps 1e-5. GELU is the
exact (erf) form, unlike the vision MLP. x is the row-wise concatenation of all images' merged
tokens.

## 5. Top-level forward `LocateAnythingForConditionalGeneration.forward`
Arguments: `pixel_values`, `input_ids`, `attention_mask`, `position_ids`, `image_grid_hws`,
`image_flags`, `past_key_values`, `labels`, `use_cache`, `**kwargs`. The forward supports
`return_dict=False` (returns a tuple) through the HF can-return-tuple convention.

1. If `pixel_values` is given, compute the per-image feature list (§3).
2. **image_flags** filter, applied only when flags are given **and their sum is > 0**. Walk the
   flags in order with a cursor into the feature list. For a flag with integer value c > 0, take
   features [cursor, cursor+c) and advance the cursor by c. For a flag equal to 0, take nothing
   and advance by 1. When the flags sum to 0, no filtering happens and every feature is kept.
3. Visual embeddings are computed only when the (filtered) list is non-empty, `input_ids` is
   given, and `input_ids` contains at least one `image_token_index`. They are mlp1 applied to the
   concatenated features. Otherwise no visual features are passed, so images with no placeholder
   are silently ignored.
4. Call the language model with `input_ids`, `visual_features`, `image_token_index`,
   `attention_mask`, `position_ids`, `past_key_values`, `use_cache` and the extra kwargs.
5. Loss when `labels` is given: standard next-token cross-entropy on the returned logits. Use
   logits [:, :-1] against labels [:, 1:], take the mean, keep the default ignore index −100, and
   flatten over `text_config.vocab_size`.
6. Return a CausalLMOutputWithPast with (loss, logits, past_key_values, hidden_states, attentions)
   taken from the LM output.

## 6. Decoder

### 6.1 Image injection (both decoder variants)
Look up the token embeddings first. If `inputs_embeds` is supplied, use it and skip the lookup.
If neither `input_ids` nor `inputs_embeds` is given, raise ValueError. When `visual_features` is
not None:
- `input_ids` must be present, otherwise raise ValueError.
- Reshape the features to [-1, D].
- The number of positions equal to `image_token_index` must equal the number of feature rows.
  Otherwise raise ValueError containing **"Image placeholder count does not match visual features"**.
  This check runs **before** any layer runs, so a caller's DynamicCache stays at length 0 and a
  retry with correct features gives logits identical to a fresh call.
- Replace those positions, in row-major (batch, then sequence) order, with the feature rows cast to
  the embedding dtype. Clone first, so the input tensor is not modified.

### 6.2 Qwen2 variant (`text_config.architectures[0] == "Qwen2ForCausalLM"`), model-specific
The causal LM is the vendored native Qwen2ForCausalLM with an inner model of the same parameter
layout but replaced forward logic:

- `past_key_values` must be None or a Transformers `DynamicCache`. Anything else (for example an
  empty tuple) raises ValueError containing "DynamicCache only".
- If `use_cache` is truthy and no cache is given, create `DynamicCache()` **without a config**. The
  cache must never apply sliding-window eviction, because MTP needs the full prefix.
- `use_cache=None` resolves from config defaults. Qwen2Config defaults `use_cache` to True, so
  this behaves as **True** for mask selection below.
- Giving both `input_ids` and `inputs_embeds` raises ValueError.
- past = cache length before this call. L = number of new tokens. K = past + L.
- `position_ids` defaults to past … past+L−1, broadcast over the batch.
- Rotary embedding: native Qwen2 rotary (default rope, theta from `rope_parameters["rope_theta"]`,
  rotate-half form). Native decoder layers run unchanged: RMSNorm with eps `rms_norm_eps` and a
  float32 variance, q/k/v Linear with bias, o_proj without bias, GQA repeat, scale head_dim^−½,
  SwiGLU MLP. All layers get the **same** additive mask, built as in §6.3.
- The final RMSNorm feeds `lm_head`. `logits_to_keep` and labels follow native Qwen2ForCausalLM
  (native loss_function).
- **Logits are cast to float32** after the native forward, whatever the model dtype.
- The output cache is returned only when use_cache is truthy.

### 6.3 Visibility mask (Qwen2 variant)
An additive float mask in the model dtype (0 means visible, −inf means hidden), shape [B, 1, L, K].
Query row i has absolute index past+i. Key column j has absolute index j.

1. If `attention_mask` is 4-D, use it as given (cast to the dtype). Skip every rule below.
2. If the module is in **training mode**, use the training block mask (§6.4) and ignore
   `attention_mask`.
3. Otherwise, start from causal: j ≤ past+i.
   - Sliding window applies when `layer_types` contains "sliding_attention". Every layer must then
     be sliding, otherwise raise NotImplementedError whose message contains "mixed". With
     sliding, also require j > past+i − sliding_window. With all layers "full_attention" (for
     example `use_sliding_window=False`), there is no window even if `sliding_window` is set.
   - A 2-D `attention_mask` must have shape exactly [B, K], otherwise raise ValueError "attention_mask must include
     the cached prefix". Columns where it is 0 become hidden for every query.
4. **MTP window rule.** It triggers when L > 1, `input_ids` is given, and the **last id of batch row 0**
   equals MASK. That single row decides for the whole batch. Then, for every batch row b, apply
   one of two rules:
   - **Cached rule** (use_cache truthy). It acts on the last bs query rows and refers to keys by
     their index from the end:
     - if not `causal_attn`, the last bs × last bs block is made **visible**, overriding causal,
       padding and sliding hiding;
     - then, always, key column K−bs−1 (the real token just before the window) is **hidden**
       from those last bs rows.
     All other rows keep the causal mask.
   - **Uncached rule** (use_cache falsy). It assumes past = 0 and that the mask is L×L. Define
     window positions as those that hold MASK or are directly followed by MASK. Every other
     position is non-window. For a window column c:
     - P(c) = largest non-window index ≤ c, or **0** when no such index exists;
     - N(c) = smallest non-window index ≥ c, or +∞ when none exists.
     - Hide column c from every row r ≥ N(c), but only when c > P(c). With the P = 0 fallback, a
       window column at index 0 is therefore never hidden.
     - If not `causal_attn`, also make column c visible to rows r with P(c) < r < c (bidirectional
       inside a window segment).
     The effect is that later real tokens cannot see any earlier speculative window, including
     that window's repeated predecessor token, and each window sees all of its own members.
   Fixture `masks.safetensors` holds both rules, the edge cases above (leading window, padding
   column) and the training mask, as 0/1 visibility tables.
5. Rows that end up fully hidden (for example left-padded query rows) are not specified. With
   "eager" they give NaN, which spreads to the whole sequence of that batch row in later layers.
   With "sdpa" they stay finite (see the fixture README).

### 6.4 Training block-diffusion mask (Qwen2, `model.training` True)
Per batch row:
- Let the split point s be the first index i ≥ 1 where `position_ids[i] < position_ids[i−1]`,
  or −1 when there is no drop. The clean region is the indices below s. Every other index is in
  the block region. For an index t in the block region, its block is floor((t − s)/bs). Its block
  start is s + block·bs, clamped to [0, L−1]. Its prefix length is `position_ids[block start]`.
- Query q may see key k when one of these holds:
  - both are in the clean region and k ≤ q;
  - both are in the block region, in the same block, and (k ≤ q, required only when `causal_attn`);
  - q is in the block region, k is in the clean region, and k < prefix length of q's block.
- Build the mask in bfloat16 (0 / −inf) and then cast it to the model dtype. Past length must be 0.

### 6.5 Qwen2 `prepare_inputs_for_generation` (shared with Qwen3)
Inputs: (input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kw).
- If a cache is given and the input is longer than the cache length S, keep only `input_ids[:, S:]`.
- Positions: use `kw["position_ids"]` if given. Otherwise, when `attention_mask` is given, use the
  cumulative sum of the mask minus 1, with padded slots set to 1. Then keep only the last
  L positions, where L is the (possibly sliced) input length.
- Return exactly the keys `input_ids`, `past_key_values`, `attention_mask` (unchanged),
  `position_ids` and `use_cache` (from kw, default True). Do not return `inputs_embeds` or
  `cache_position`.
- HF `language_model.generate(...)` must work with this in greedy mode for unpadded single inputs
  and for left-padded batches with a 2-D attention mask (fixtures `sdpa.hf_greedy_*`).
  Known limitation: with all-sliding layer types, HF generate currently fails, because HF builds a
  config-aware sliding cache that holds fewer keys than the mask expects. No fixture covers this.

### 6.6 Qwen3 variant (`architectures[0] == "Qwen3ForCausalLM"`)
- The vendored native Qwen3ForCausalLM is used **unchanged**: native causal mask creation,
  native cache creation, q_norm/k_norm, and `attention_bias` on q/k/v/o. There is **no**
  block-diffusion window rule, so MTP inputs get plain causal attention (fixture:
  `qwen3_bias` `mtp_cached_logits == mtp_uncached_logits`). There is no DynamicCache type check
  and no training mask.
- Forward takes `visual_features` and `image_token_index`. When features are given, inject them
  as in §6.1 and call the native forward with `inputs_embeds` and `input_ids=None`. Otherwise
  pass `input_ids` through.
- **Logits are not upcast**. They keep the model dtype.
- Uses the same `prepare_inputs_for_generation` as §6.5.

---------------------------------------------------------------------------------------------------

## 7. Localization sampler (pure functions, model-specific)

### 7.1 Token-id table (`model.token_ids`)
Keys: `box_start_token_id`, `box_end_token_id`, `coord_start_token_id`, `coord_end_token_id`,
`ref_start_token_id`, `ref_end_token_id` and `none_token_id`, read from the top config with the
§2.2 defaults. Also `null_token_id`, `im_end_token_id` (= `text_config.eos_token_id`),
`switch_token_id` and `default_mask_token_id` (= `text_mask_token_id`), read from `text_config`
with the §2.2 fallbacks.

"Coordinate token" means coord_start ≤ id ≤ coord_end.

### 7.2 Logit processors (V = last axis)
- **Repetition penalty** ρ (skipped when ρ = 1). Collect the token set from all previously
  generated ids of the batch row, keeping only ids in [0, V). For those vocabulary entries,
  divide logits > 0 by ρ and multiply logits ≤ 0 by ρ. Input may be 2-D [B, V] or 3-D [B, S, V],
  and the shape is preserved.
- **Temperature** τ: if τ > 0, divide the logits by τ.
- **Top-p** (applied when top_p is given and < 1): sort descending and take the cumulative
  softmax. Drop entries whose cumulative probability is > top_p, but shift the drop set one place
  to the right so the first entry crossing the threshold is kept. The top entry is always kept.
  Dropped logits become the dtype's most negative finite value.
- **Top-k** (applied when top_k is given): k' = min(k, V). Logits strictly below the k'-th
  largest become the most negative finite value. Ties at the threshold survive.
- Order of application: penalty, temperature, top-p, top-k, then softmax.
- Choice: if τ > 0, sample from the categorical distribution with torch's global RNG, and fall back
  to argmax if sampling raises. Otherwise take argmax. Confidence is the chosen probability.

### 7.3 Multi-position selection (input: logits [B,S,V], previously generated ids, token table, generate kwargs)
Returns (probs, confidence, chosen ids, box). The kwargs read are `repetition_penalty` (1.0),
`temperature` (0), `top_p`, `top_k`, `keep_k_avg` (**4**) and `generation_mode` ("hybrid").
Others are ignored. In particular, the generate loop passes a `keep_k=5` argument that has **no
effect**.
- When S == 1, box is None.
- Otherwise, for each batch row, box is the first of these that applies: the box decode of §7.5
  (processed logits and probs, keep_k = keep_k_avg, start threshold 0.7, end threshold 0.2), then
  the reference decode of §7.6 (keep 5, start threshold 0.6), then the single-element tensor [0]. Rows are
  stacked into a long tensor (batch 1 in practice).

### 7.4 Box-frame validity (6 positions, probs P[6, V])
- "empty_box" when P[0, box_start] ≥ start_thresh (0.7 as called), P[1, none] > 0.2,
  P[2, box_end] > 0.2, P[3, null] > 0.1 and P[4, null] > 0.1.
- Otherwise "legal_box" when P[5, box_end] + P[5, null] + P[5, im_end] ≥ end_thresh (0.2).
  This test does **not** require a box start at position 0.
- Otherwise "illegal_box".

### 7.5 `decode_bbox_avg(logits, probs, token_ids, keep_k=5, start_thresh=0.7, end_thresh=0.2, generation_mode='hybrid')`
- empty_box → long tensor [box_start, none, box_end, null, null, null].
- illegal_box → None.
- legal_box: take the top-keep_k entries of P at positions 1–4. If any of those four positions has
  no coordinate token among its top-k, return None. Otherwise, at each position, take the
  highest-ranked coordinate token c (probability p_c).
  - Hybrid: position is "abnormal" when p_c < 0.9, more than one coordinate token is in its top-k,
    and (largest coordinate id − smallest coordinate id) among those top-k coordinates > 60.
    Abnormal positions become token id **0**.
  - Fast: always use c.
  - Return [box_start, c1, c2, c3, c4, box_end]. The first and last are fixed ids, regardless of
    what was predicted at positions 0 and 5.

### 7.6 Reference decode (keep_k=5, start threshold 0.6)
- If P[0, ref_start] < start_thresh, return None.
- For every later position (1 … S−1), the top-keep_k must contain at least one non-coordinate
  token. Otherwise return None.
- Return [ref_start] followed by the highest-ranked non-coordinate token at each later position.

### 7.7 `handle_pattern(candidate, token_ids, generation_mode='hybrid') -> dict`
The candidate c (first positional argument) is a sequence of 6 ids. The result has keys `type`, `tokens` (a Python list), `need_switch_to_ar`
and `is_terminal`. Rules are checked in this order:
1. c[0] == null or c[0] == im_end → type "im_end", tokens [im_end], terminal.
2. c[0:2] == [box_start, none] → "empty_box", tokens [box_start, none, box_end].
3. c[0] == box_start. Let n = the number of consecutive coordinate tokens starting at c[1]
   (checking at most c[1..4]).
   - n == 4 and c[5] == box_end → "coord_box", tokens = all 6.
   - n == 2 and c[3] == box_end → "point_box", tokens = c[0:4].
   - Otherwise, fast mode → "coord_box" with all 6 tokens, and hybrid (or any non-fast) mode →
     "error_box" with tokens = c[0 : n+1] and `need_switch_to_ar` True.
4. Anything else → "ref_object". Cut c just before the first null, if there is one (a null at
   index 0 was already handled). Then, if at least 2 tokens remain and the last two are both
   ref_end, drop the final one. Tokens = the result. The result may contain MASK, image or any
   other ids.

## 8. Localization `generate` (top model), model-specific
Signature: `generate(pixel_values=None, input_ids=None, attention_mask=None, visual_features=None,
image_grid_hws=None, tokenizer=None, n_future_tokens=6, **kw)`. It runs under no-grad, and
`attention_mask` is ignored.

Validation, in this order:
1. `input_ids` may also arrive as `kw["inputs"]`. Missing → ValueError "input_ids is required".
2. `n_future_tokens` must equal `text_config.block_size` (default 6), otherwise ValueError.
3. Cast `pixel_values` to the vision tower's device and dtype.
4. `tokenizer` is required (ValueError). It must provide `model_max_length` and `batch_decode`,
   plus `decode` when verbose.
5. A numpy `image_grid_hws` is converted to an int32 tensor.
6. Batch size must be 1 and `kw["use_cache"]` must be truthy (assertions).
7. `generation_mode` ∈ {"fast", "slow", "hybrid"} (default "hybrid"), otherwise an assertion fails.

Setup:
- T_max = min(tokenizer.model_max_length, prompt_len + kw.get("max_new_tokens", 2048)).
- Visual embeddings: use `visual_features` when given. Otherwise use the feature list from
  pixel_values, or None. If `image_grid_hws` is given, concatenate and apply mlp1. They are passed
  only on the **first** forward call, together with `image_token_index`.
- MTP starts enabled in fast and hybrid modes and disabled in slow mode.

Loop while the length of `generated` (prompt plus accepted tokens) is below T_max:
1. Inputs.
   - MTP round: sequence = generated + [last token of generated] + (bs−1) × MASK.
     Position ids over the full sequence are 0, 1, …. The last bs entries are each lowered by 1,
     so the duplicated token shares the real last token's position and the MASKs get positions
     len … len+bs−2. Then apply §6.5 with the current cache: the unseen suffix plus matching
     positions, `use_cache=True` and `attention_mask` None.
   - AR round: the sequence is just `generated`, with positions 0 … len−1, through §6.5.
2. Run the language model on those inputs. The Qwen2 MTP window rule triggers because the last id
   is MASK and use_cache is True.
3. **Cache crop**: let the excess be cache_len − len(generated), where generated has not yet been
   extended. If the excess is > 0, crop the cache by that many tokens (Transformers 5.23
   `DynamicCache.crop` with a **negative** count). Only the real prefix stays cached. The new
   tokens' K/V are computed in the next round as part of its input.
4. Choose tokens.
   - MTP: take the logits of the last bs positions → §7.3 with the generate
     kwargs. If the box row is all zeros, the candidate is the chosen ids (argmax or sampled). Otherwise
     the candidate is the box/ref decode. Then apply `handle_pattern` (§7.7) to the candidate.
   - AR: take the last position → §7.3 → one token t. In hybrid mode, the type is
     "box_end_ar" if t == box_end, "coord_ar" if t is a coordinate or none token, and "im_end"
     otherwise (so **any other token ends generation**). In slow mode, the type is "im_end" only
     when t == im_end, else "continue_ar".
5. Append the tokens. They are always appended, including the im_end token and an
   oversized final block. There is **no truncation** to T_max.
6. Stop if the type is "im_end". In hybrid mode, "error_box" switches to AR and counts one switch,
   and "box_end_ar" switches back to MTP. Fast mode never switches, and slow mode stays in AR.

Return `tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=False)[0]`, a string.
With `kw["verbose"]` true, print a statistics line and return (response, history, info_string).
Each history entry is ('ar' if the type contains "ar" else 'mtp', decoded tokens).

Fixtures `expected_meta.json → localization_generate` record, for slow, fast and hybrid modes:
the response, every MTP candidate with its type and tokens, and each forward call's
(input length, cache length after the forward and before the crop). They cover random weights
(text and image prompts) and a "rigged" deterministic model (see the README). For example, the
rigged hybrid model makes 2 forward calls and returns "32,32": one MTP error_box, then one AR
token classed as "im_end".

---------------------------------------------------------------------------------------------------

## 9. Image processor `LocateAnythingImageProcessor` (a BaseImageProcessor)
`model_type = "locateanything"`. Init kwargs and defaults: patch_size 14, image_mean (0.5,0.5,0.5),
image_std (0.5,0.5,0.5), in_token_limit 4096, merge_kernel_size (2,2), plus base kwargs. It saves
to `preprocessor_config.json` with keys `image_mean`, `image_std`, `in_token_limit`,
`merge_kernel_size`, `patch_size` and `image_processor_type: "LocateAnythingImageProcessor"`.

`__call__` and `preprocess(images, return_tensors=None)`: wrap the input as a list of images and
validate it ("Invalid image type..." ValueError). Each image must be a PIL image. For each one:
1. Size limit: with (w, h) the PIL size, if (w//p)·(h//p) > in_token_limit, set
   s = √(in_token_limit / ((w//p)·(h//p))) and resize with PIL BICUBIC to (int(w·s), int(h·s)).
2. Round up: the target size is each side rounded **up** to a multiple of merge·p (width uses
   merge[1], height uses merge[0]). If it differs from the current size, **resize** (stretch,
   not pad) with PIL BICUBIC.
3. If w//p ≥ 512 or h//p ≥ 512, raise ValueError "Exceed pos emb".
4. Convert to RGB, then to a float tensor in [0, 1] (torchvision to_tensor), then normalize per
   channel as (x − mean)/std.
5. Patchify the [C, H, W] tensor into [(H/p)·(W/p), C, p, p], with patches in row-major grid
   order and each patch keeping its C, p, p layout. The grid is (H/p, W/p).

Output: a BatchFeature with `pixel_values` (all images' patches concatenated as a torch tensor)
and `image_grid_hws` (a numpy integer array [N, 2]), converted according to `return_tensors`.

## 10. Processor `LocateAnythingProcessor` (ProcessorMixin)
`attributes = ["image_processor", "tokenizer"]`. `valid_kwargs` = chat_template, num_image_tokens,
image_token, video_token, images_kwargs, videos_kwargs, text_kwargs.

Init kwargs: image_processor, tokenizer, chat_template=None, image_token='<IMG_CONTEXT>',
video_token='<IMG_CONTEXT>', merge_kernel_size=(2,2), image_placeholder='image',
video_placeholder='video', image_start_token='<img>', image_end_token='</img>'.
- If the tokenizer has an `image_token` (or `video_token`) attribute, that overrides the kwarg.
  Token ids come from the tokenizer's `image_token_id` / `video_token_id` when truthy, and from
  converting the token string otherwise.
- `save_pretrained(dir)`: raise ValueError if dir is an existing file, create the dir, save as
  usual (a `processor_config.json` that embeds the image-processor dict plus image_end_token,
  image_placeholder, image_start_token, image_token, merge_kernel_size, processor_class,
  video_placeholder and video_token), and **also** write the image processor's own
  `preprocessor_config.json`.
- `from_pretrained(path, **kw)`: drop `trust_remote_code` and `code_revision`. The config comes
  from `kw["config"]`, or `LocateAnythingConfig.from_pretrained` otherwise. The tokenizer comes
  from `kw["tokenizer"]`, or `gptqmodel.utils.hf.load_hf_tokenizer(path, model_config=config,
  trust_remote_code=False, ...)` otherwise. The image processor comes from
  `LocateAnythingImageProcessor.from_pretrained(path)`. Read the processor dict and pass only
  chat_template, image_token, video_token, merge_kernel_size, image_placeholder, video_placeholder,
  image_start_token and image_end_token. **Never** go through the Auto* classes or the checkpoint
  `auto_map` (tests plant hostile modules and check they never run). `from_args_and_dict` also
  has to work: remove `processor_class`, validate the kwargs, construct, then apply any leftover
  kwargs that match processor attributes.
- `model_input_names`: tokenizer names followed by image-processor names, with duplicates removed
  in order. `batch_decode` and `decode` delegate to the tokenizer.

### 10.1 `__call__(images=None, text=None, audio=None, videos=None, **kw)`
- Merge kwargs with defaults (text padding False). `text` must be a str or a non-empty list of
  str, otherwise ValueError "Invalid input text. Please provide a string, or a list of strings".
- `videos_kwargs` may carry `timestamps` and `fps`. These are per-video lists, removed before the
  image processor is called.
- For each text sample in order, expand its placeholders (§10.2) against the images and videos
  **not yet consumed** by earlier samples, plus the matching slices of timestamps and fps. Then
  advance the consumption counters by the number of placeholders expanded.
- Tokenize all expanded texts in one tokenizer call with text_kwargs.
- When any media appeared, add `pixel_values` (all media of all samples concatenated in order) and
  `image_grid_hws` (concatenated). Return a BatchFeature, using text_kwargs' `return_tensors` as
  the tensor type.
- A placeholder that points past the end of the list raises (IndexError).

### 10.2 Placeholder expansion
This is a public method, because tests call it directly:
`replace_media_placeholder(text, image_list, video_list, timestamps_list, fps_list, **kw)`, where
kw must contain `images_kwargs` and `videos_kwargs` dicts. It returns a 5-tuple: (expanded text,
pixel_values as a torch tensor that is an empty tensor when no media appeared, grids as a numpy
array that is empty when no media appeared, number of images expanded, number of videos expanded).
`timestamps_list` and `fps_list` are per-video lists or None.
- Pattern: `<{image_placeholder}-N>` or `<{video_placeholder}-N>`, where N is a decimal 1-based
  index into the remaining list of that kind.
- Image: run the image processor on that one image (with images_kwargs). The token count is
  h·w // (kh·kw), using the **image processor's** merge kernel. Replace the placeholder with
  `"<image N>" + image_start + image_token × count + image_end`. Note the literal text
  "<image N>" with a space.
- Video: the input is either a list of PIL frames or a [T, C, H, W] tensor. Tensor frames become
  PIL frames like this: floating frames whose max > 1 are cast straight to uint8 (truncation);
  other floating frames are multiplied by 255 and then cast. Run the image processor on all
  frames (with videos_kwargs) and count tokens per frame.
  - Each frame segment is `"Frame-{i}-{t:.2f}s: " + start + tokens + end` when timestamps are
    available, and `"Frame-{i}: " + ...` otherwise (i is 1-based).
  - Timestamps are "available" when a timestamps list was supplied for this sample and the
    integer −1 is not an **element of the outer per-video list**. Inner −1 values therefore do
    not suppress them.
  - If the timestamp count differs from the frame count, both lists are cut to the shorter one
    and a warning is logged.
  - Prefix: `"The {ordinal} video sampled with {fps:.2f} fps: "` when an fps is supplied,
    otherwise `"The {ordinal} video: "`. The ordinal is first … tenth for N = 1 … 10, and other
    N fail. All frame segments are joined with no separator.
- pixel_values and grids are collected in placeholder order.

### 10.3 `py_apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`
tokenize=True is unsupported (assertion). First, concatenate the text of all **user** messages:
string content, plus `text` items and bare string items. Then render each message:
- If the first message is not a system message, emit
  `"<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"` before it.
- Emit `"<|im_start|>{role}\n"`. String content is emitted as is. For list content, take the items
  in order:
  - image items (`type=="image"`, or a key `image` / `image_url`) increment a global image counter
    and emit `<image-k>`, unless that exact string appears in the concatenated user text;
  - video items do the same with `<video-k>`;
  - `text` items emit their text, and bare strings are emitted as is.
- End the message with `"<|im_end|>\n"`.
- If add_generation_prompt is true, finish with `"<|im_start|>assistant\n"`.

### 10.4 Vision info and media fetching
- `extract_vision_info(conversations)`: accepts a single conversation or a list of them. Collects
  every list-content element that has an `image`, `image_url` or `video` key, or whose `type` is
  one of those.
- `process_vision_info(conversations, return_video_kwargs=False, video_reader_backend="torchvision")`
  returns (images or None, videos or None), plus `{'fps': [...], 'timestamps': [...]}` when the
  flag is set. Images go through fetch_image and videos through fetch_video. Any other element
  raises ValueError.
- `fetch_image(ele)`: the source is `ele["image"]`, else `ele["image_url"]`.
  - An http(s) string is loaded through the **module-level** name `load_image_from_url(url,
    max_bytes=REMOTE_MEDIA_MAX_BYTES)` (64 MiB), so tests can monkeypatch it.
  - Anything else goes through `gptqmodel.utils.image.fetch_image({"image": src})`.
  - Then convert to RGB: RGBA is composited over white using alpha, and other modes are converted
    to RGB. Fetch errors (private or loopback addresses, HTTP errors) propagate as `ImageFetchError`.
- `fetch_video(ele, return_video_sample_fps=False, video_reader_backend="torchvision")`:
  - For an http(s) URL, read the bytes through the module-level `load_bytes_from_url(url,
    max_bytes=...)`, write them to a temp file whose suffix is the URL path's extension (query
    stripped), and recurse.
  - For a path string, strip `file://`. Read with `VIDEO_READER_BACKENDS[backend]`, looked up at
    call time ("torchvision" means torchvision.io read_video in TCHW with start/end pts in
    seconds; "decord" is also available; backend None selects decord when installed). Any reader
    exception logs a warning and retries with the torchvision entry.
  - Readers return (TCHW uint8 video, sample_fps, timestamps). Frame indices come from
    `get_video_frame_indices`.
  - If T·H·W > `ele.get("video_total_pixels", 25913856)`, scale both sides by √(cap/current),
    truncate to int, and resize with torchvision bicubic and antialias. In every case the output
    is converted to float.
  - For a list or tuple of frames, fetch each one as an image (extra ele keys passed along). The
    timestamps are [−1]·n and the fps is `ele.get("fps", 2.0)`. Return the PIL list.
- `get_video_frame_indices(ele, total, fps)`: n = max(1, round(total/fps · ele.get("fps", 2.0))),
  capped at `ele.get("max_frames", 64)` and at total. Indices are 0 … total−1 when n == total,
  otherwise linspace(0, total−1, n) rounded. Sample fps = n / max(total, 1e-6) · fps.
  Reader timestamps = video_start + index/fps.

---------------------------------------------------------------------------------------------------

## 11. Definition integration (`definitions/locateanything.py`, kept). The port must support:
- `LocalModelLoader.from_pretrained(path, dtype=..., attn_implementation=...)` and `from_config`,
  including on the meta device. Safetensors-only save. Save → load → save → load gives identical
  tensors, dtypes and logits.
- `language_model.model.layers.{i}` with native submodule names, so the quantization module tree
  resolves: `input_layernorm`, `self_attn.{q,k,v,o}_proj`, `post_attention_layernorm` and
  `mlp.{gate,up,down}_proj`, plus `self_attn.{q,k}_norm` for qwen3, detected through
  `text_config.model_type == "qwen3"`. No other owners exist in a layer.
- `language_model(input_ids=..., visual_features=..., image_token_index=..., attention_mask=...,
  labels=..., use_cache=...)` is the calibration capture entry point.
- `model.extract_feature` returning a list, `model.mlp1`, `model.image_token_index`, and
  `model.vision_model` (its parameters and dtype are used to cast pixels; grids are cast to int32).
- Text-only wrapper calls go straight to `language_model(...)` and `language_model.generate(...)`.
- The processor methods `py_apply_chat_template`, `process_vision_info` and `__call__`.

## 12. Behaviors the existing tests require (summary)
From `tests/test_locateanything_local_port.py`:
- Decoder logits match independent equations (§6.2–6.3) for sdpa and eager, kv heads ∈ {1, 2, 4},
  with and without a trailing MASK window. The MTP output must differ measurably from plain causal.
- Sliding window follows `layer_types` (full means no window, all sliding means banded, mixed
  raises NotImplementedError "mixed").
- Vision features and the multimodal logits match independent equations (§3–§6.1). Images change
  the logits.
- Cache continuation equals a full forward. The prefix cache is a DynamicCache whose length grows
  to 5. HF greedy generate produces new tokens. The model saves to safetensors only. A config
  whose architecture string is foreign raises "Unsupported LocateAnything decoder". Reload works
  for fp32, fp16 and bf16 and for tied and untied heads, keeping tie identity and exact tensors.
  Qwen2 and Qwen3 are both covered.
- `GPTQModel.load` with hostile `auto_map` (any trust and bypass setting) gives the owned classes,
  never runs checkpoint code, and sets `trust_remote_code` False. The processor reloads as owned
  classes.
- Processor: chat template, image plus 2-frame list video → pixel_values (12, 3, 2, 2) and grids
  [[2, 2]]×3. The first pixel equals x/127.5 − 1. Placeholder expansion gives 3 image tokens and
  "Frame-1:"/"Frame-2:". Save and reload gives identical outputs. A missing image raises.
- Remote media: bounded fetch with `max_bytes = REMOTE_MEDIA_MAX_BYTES`. Private and loopback
  hosts are rejected and HTTP 404 propagates, both through `ImageFetchError` (tests patch the
  module-level `load_image_from_url`, `load_bytes_from_url` and
  `VIDEO_READER_BACKENDS["torchvision"]`). The temp file keeps the ".mp4" suffix.
- Sampler: coord_box, point_box, error_box, null→[eos] and empty-box truncation. Hybrid
  ambiguous-coordinate rejection versus fast top-1 (§7.5).
- Localization generate runs in all modes for Qwen2 and Qwen3. With the rigged model the result
  is "32"×{slow 8, fast 12, hybrid 2} over {8, 2, 2} forward calls.
- A visual feature count mismatch leaves the cache untouched. Dummy and flagged images follow
  §5 step 2.
- `magi`→sdpa, `flash_attention_2` is rejected, and a non-DynamicCache raises.
- The MTP cache keeps only the real prefix across two speculative windows (§8 step 3).
- Training-mode logits and loss match the §6.4 oracle for causal_attn False and True. Uncached
  multiple windows match the §6.3 uncached rule.
- AWQ quantize, pack, save and reload: 14 quantized linears in the tiny model (2 layers × 7).
  Vision, mlp1, embed, norm and head stay exact. Folding is neutral under causal and MTP
  visibility.
From `tests/test_locateanything_support.py`: definition contract fields, model_type selection
without checkpoint Python, rope and attention policy (§2.2), base modules (`vision_model`, `mlp1`,
embed, norm, lm_head), dataset preparation calls, the text-forward bypass, the pixel dtype cast and
the input-capture path. From `tests/module_tree/...`: the plan covers every decoder weight, and
the released 770-tensor inventory hash matches.

## 13. Standard versus model-specific
- Standard, so reuse the vendored blocks: everything inside the decoder layers, rotary, RMSNorm,
  embeddings, lm_head and native loss (Qwen2/Qwen3); the Qwen3 path entirely; LayerNorm, GELU,
  Linear and Conv.
- Model-specific, so implement from this spec: the MoonViT pieces (2D complex RoPE with column
  angle first, the interpolated learned table, packed per-image attention, the 2×2 merge), the
  connector, image injection, the MTP and training visibility rules, the cache policy, the float32
  logits upcast (Qwen2 only), the localization sampler and loop, the image processor, and the
  processor templating.
