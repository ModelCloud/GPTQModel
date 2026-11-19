# MobileLLM: behavioral specification (clean-room phase 1)

Target: Transformers 5.23, CPU and accelerator, built from GPT-QModel's vendored Apache-2.0
Llama port (`gptqmodel/models/ports/native/llama`) plus standard Transformers utilities
(`ACT2FN`, `ALL_ATTENTION_FUNCTIONS`, `create_causal_mask`, `DynamicCache`,
`GradientCheckpointingLayer`, `capture_outputs` / `merge_with_config_defaults`,
`slice_logits_to_keep`, `PreTrainedModel.loss_function`, `transformers.initialization`).

In short, MobileLLM is a pre-norm Llama decoder (RMSNorm, GQA, rotary, SwiGLU MLP) with
three model-specific features:

1. **Embedding sharing** (`share_embedding`, default on): no `lm_head` module exists. Logits
   are computed by multiplying with the input embedding matrix.
2. **Immediate block-wise layer sharing** (`layer_sharing`): each physical decoder layer runs
   twice in a row, and each run gets its own KV-cache slot.
3. **Source-faithful `pretraining_tp` slicing and stateful dynamic-NTK RoPE**: these behave
   differently from current upstream Llama (see sections 4.3 and 4.6).

Golden fixtures that pin all of this numerically are in `tests/fixtures/cleanroom/mobilellm/`
(see that README).

---

## 1. Public interface (must stay compatible)

### 1.1 Package and class names
- Package `gptqmodel.models.ports.mobilellm` exports exactly `MobileLLMConfig`,
  `MobileLLMForCausalLM`, `MobileLLMModel`.
- `gptqmodel/models/definitions/mobilellm.py` (Apache, kept) loads the port through
  `LocalModelLoader(config_class="gptqmodel.models.ports.mobilellm.MobileLLMConfig",
  model_class="gptqmodel.models.ports.mobilellm.MobileLLMForCausalLM")`. These dotted paths
  must keep resolving.
- `MODEL_MAP["mobilellm"]` is `MobileLLMQModel` (in `gptqmodel/models/auto.py`; not part of the port).
- The decoder-layer class name `MobileLLMDecoderLayer` is used as the `_no_split_modules`
  entry. Keep that name.
- `MobileLLMForCausalLM.model` is a `MobileLLMModel`. `MobileLLMModel` has the attributes
  `embed_tokens`, `layers` (ModuleList), and `norm`.
- Each layer has `self_attn` (with `q_proj`, `k_proj`, `v_proj`, `o_proj`, all
  `torch.nn.Linear`), `mlp` (with `gate_proj`, `up_proj`, `down_proj`, all `nn.Linear`),
  `input_layernorm`, and `post_attention_layernorm`.
- Each attention module owns a rotary submodule reachable as `self_attn.rotary_emb`. The tests
  read and compare two of its attributes:
  - `rotary_emb.inv_freq`: a non-persistent fp32 buffer.
  - `rotary_emb._base`: a Python float holding the current RoPE base.

  Both attribute names are test-visible interface (see 4.6 and 7).
- The norm class has a `weight` parameter of shape `[hidden_size]` and no bias.

### 1.2 Config: `MobileLLMConfig` (subclass of `transformers.PreTrainedConfig`)
`model_type = "mobilellm"`, `keys_to_ignore_at_inference = ["past_key_values"]`.

Constructor keywords and defaults:

| field | default | notes |
|---|---|---|
| vocab_size | 32000 | |
| hidden_size | 576 | |
| intermediate_size | 1536 | |
| num_hidden_layers | 30 | number of *physical* layers |
| num_attention_heads | 9 | |
| num_key_value_heads | 3 | `None` -> equals num_attention_heads |
| head_dim | None | `None` -> hidden_size // num_attention_heads |
| hidden_act | "silu" | looked up in `ACT2FN` |
| max_position_embeddings | 2048 | |
| initializer_range | 0.02 | |
| rms_norm_eps | 1e-5 | |
| use_cache | True | |
| pad_token_id | None | passed to super |
| bos_token_id | 1 | passed to super |
| eos_token_id | 2 | passed to super |
| pretraining_tp | 1 | |
| tie_word_embeddings | False | passed to super |
| rope_theta | 10000.0 | |
| rope_scaling | None | legacy alias for rope_parameters |
| rope_parameters | None | |
| attention_bias | False | bias on q/k/v/o |
| attention_dropout | 0.0 | |
| mlp_bias | False | bias on gate/up/down |
| share_embedding | True | MobileLLM-specific |
| layer_sharing | False | MobileLLM-specific |
| **kwargs | | forwarded to `PreTrainedConfig.__init__` |

How RoPE parameters are normalized:
- Source dict: `rope_parameters` if it is not None, otherwise `rope_scaling`, otherwise `{}`.
  The dict is deep-copied.
- `rope_theta` comes from the dict's `"rope_theta"` if present, otherwise from the keyword.
- The type is the dict's `"rope_type"`, falling back to its legacy `"type"`, then `"default"`.
- `factor` comes from the dict, default 1.0.
- The stored value is `config.rope_parameters = {"rope_type": t, "rope_theta": theta}`. A
  `"factor"` key is added only when the type is not `"default"`. `config.rope_theta` is also set.
- A serialized `config.json` contains both `rope_parameters` and top-level `rope_theta`.
  Reloading must give an identical `rope_parameters`. A checkpoint that has
  `"rope_scaling": null` and no `rope_parameters` loads as default RoPE.

Validation: every violation raises `ValueError` during construction.
- vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, the
  resolved num_key_value_heads, and max_position_embeddings must each be an `int` (not `bool`)
  and >= 1.
- `0 <= attention_dropout <= 1`.
- `rope_theta > 0`.
- The RoPE type must be one of `{"default", "linear", "dynamic"}`.
- `factor` must be an int or float and >= 1. This is checked even for the default type.
- `head_dim` must be an int >= 1 and even, with `head_dim * num_attention_heads == hidden_size`
  (so a free `head_dim` is not supported), and num_attention_heads must be divisible by
  num_key_value_heads.
- The dynamic type also requires `head_dim > 2`.
- `pretraining_tp` must be an int >= 1 that divides each of hidden_size, intermediate_size,
  num_key_value_heads*head_dim, and vocab_size.

Cases the tests require to fail:
- `num_key_value_heads=3` with 4 heads
- `head_dim=3`
- `head_dim=0`
- unknown RoPE type
- linear factor 0.5
- `pretraining_tp=7`
- kv 0
- heads 0
- hidden 0
- layers -1
- `rope_theta=0`
- `attention_dropout=1.1`

### 1.3 Checkpoint state_dict (exact keys, safetensors)
For each physical layer `i` in `0..num_hidden_layers-1`:
```
model.layers.{i}.input_layernorm.weight            [H]
model.layers.{i}.post_attention_layernorm.weight   [H]
model.layers.{i}.self_attn.q_proj.weight           [H, H]
model.layers.{i}.self_attn.k_proj.weight           [KV*D, H]
model.layers.{i}.self_attn.v_proj.weight           [KV*D, H]
model.layers.{i}.self_attn.o_proj.weight           [H, H]
model.layers.{i}.mlp.gate_proj.weight              [I, H]
model.layers.{i}.mlp.up_proj.weight                [I, H]
model.layers.{i}.mlp.down_proj.weight              [H, I]
(+ .bias of shape [out] on the four attention projections iff attention_bias,
   and on the three MLP projections iff mlp_bias)
model.embed_tokens.weight                          [V, H]
model.norm.weight                                  [H]
lm_head.weight                                     [V, H]   only when share_embedding is False
```
H = hidden_size, D = head_dim, KV = num_key_value_heads, I = intermediate_size, V = vocab_size.

Notes:
- `layer_sharing` does not add any tensors.
- The rotary `inv_freq` is non-persistent and never saved.
- Without biases there are exactly `9 * num_hidden_layers + 2` keys (share_embedding=True).

Embedding tie behavior:
- `share_embedding=True`: there is no `lm_head` attribute, and `_tied_weights_keys` must be empty
  (`{}`) on the instance. `get_output_embeddings()` returns `model.embed_tokens`.
  `set_output_embeddings(x)` replaces the input embedding, so afterwards input and output
  embeddings are the same object.
- `share_embedding=False`: `lm_head = Linear(H, V, bias=False)`. The class declares
  `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`, so standard HF tying
  applies when `tie_word_embeddings=True`:
  - `lm_head.weight` is the embedding tensor;
  - safetensors omits `lm_head.weight`;
  - after reload, `get_output_embeddings().weight is model.embed_tokens.weight`;
  - `set(state_dict())` still contains `lm_head.weight`.

  When `tie_word_embeddings=False` (the default), the head is independent.
  `set_output_embeddings(x)` sets `lm_head`.
- `embed_tokens` is `nn.Embedding(V, H, padding_idx=pad_token_id)`.

### 1.4 PreTrainedModel class attributes
- `config_class = MobileLLMConfig`
- `base_model_prefix = "model"`
- `supports_gradient_checkpointing = True`
- `_no_split_modules = ["MobileLLMDecoderLayer"]`
- `_skip_keys_device_placement = ["past_key_values"]`
- `_supports_sdpa = True`
- `_supports_attention_backend = True`
- output capture: hidden states are recorded from every decoder-layer *call*, and attentions
  from every attention *call*
- `inv_freq` is kept strictly in fp32 (`_keep_in_fp32_modules_strict = ["inv_freq"]`)
- `MobileLLMForCausalLM._supports_default_dynamic_cache()` returns **False**, so generation
  does not pre-build a per-physical-layer cache (see 5).

### 1.5 Tokenizer
The port has no tokenizer or processor. Tests use a generic `PreTrainedTokenizerFast` saved next
to the checkpoint. Token ids pass straight through to `input_ids`.

---

## 2. Initialization (only affects missing checkpoint keys)
- Use the standard `PreTrainedModel._init_weights`:
  - Linear: normal(0, initializer_range), bias zero
  - Embedding: normal(0, initializer_range), padding row zero
  - RMSNorm weight: ones
- Additionally, reset each rotary module's `inv_freq` to the default frequencies (formula in 4.6,
  using `config.rope_theta`).
- All of these writes must go through `transformers.initialization` helpers so that tensors
  already loaded from the checkpoint are not overwritten. The test drops only
  `layers.0.self_attn.q_proj.bias` and checks two things:
  - the loaded q weight is untouched and the missing bias becomes zero;
  - `model.norm.weight` stays as loaded.

---

## 3. Forward computation (`MobileLLMModel`)

### 3.1 Input checks (all `ValueError`, raised before any cache mutation)
- Exactly one of `input_ids` / `inputs_embeds` must be given.
- If `past_key_values` is given, it must be a `DynamicCache`. A `StaticCache` is rejected, and
  the error message must contain "DynamicCache".
- Let `uses = num_hidden_layers * (2 if layer_sharing else 1)`. A cache is rejected if all three
  of these hold:
  - it already has layers;
  - it is not a lazily-growing cache (its `layer_class_to_replicate` attribute is None or
    missing);
  - its layer count != `uses`.

  The error message must contain "one slot for each decoder use". Example:
  `DynamicCache(config=model.config)` with sharing on pre-builds 2 layers for 4 uses and must be
  rejected without touching it.
- `attention_mask`, if given, must be 2-D with shape `[batch, past_len + cur_len]`. The error
  message must contain "attention_mask". `past_len = cache.get_seq_length()`, or 0.

### 3.2 Setup
- `use_cache` defaults from config (True), handled by the `merge_with_config_defaults`
  decorator.
- If `use_cache` is true and no cache was passed, create an empty lazily-growing
  `DynamicCache()` (no config argument).
- `position_ids` default to `arange(past_len, past_len + cur_len)[None]`.
- Get the mask from `create_causal_mask(config, inputs_embeds, attention_mask, past_key_values,
  position_ids)`. This is standard and identical to the vendored Llama.
- Return `BaseModelOutputWithPast(last_hidden_state=norm(h),
  past_key_values=cache if use_cache else None)`.

### 3.3 Layer schedule (layer sharing)
Physical layers run in order. With sharing on, each layer is applied **twice consecutively**
(L0, L0, L1, L1, …), not as a repeated whole stack. Use number `u` (0-based, counting every
call) uses KV-cache slot `u`. This means:
- physical layer `i`, repeat `r`, uses slot `i*2 + r` with sharing;
- physical layer `i` uses slot `i` without sharing.

The slot index has to be passed per call, because the attention's own `layer_idx` (the physical
index) is wrong when sharing. Both uses share the same parameters and the same rotary module.

### 3.4 Decoder layer (pre-norm, standard Llama structure)
```
h = h + Attn(RMSNorm_in(h))
h = h + MLP(RMSNorm_post(h))
```

### 3.5 RMSNorm
- `y = weight * ( (x32 * rsqrt(mean(x32^2, last dim) + eps)).to(x.dtype) )`, where
  `x32 = x.float()`.
- eps is inside the rsqrt.
- The weight is multiplied *after* casting back to the input dtype. This is the same as Llama
  RMSNorm.

### 3.6 MLP
`down(act(gate(x)) * up(x))`, where `act = ACT2FN[hidden_act]` (default SiLU). This is the same
as Llama MLP, but see 4.3 for `pretraining_tp > 1`.

### 3.7 Attention (GQA, standard math)
Steps:
1. `q = q_proj(x)` -> `[B, nh, T, D]`; `k = k_proj(x)` and `v = v_proj(x)` -> `[B, nkv, T, D]`
   (the usual view and transpose).
2. RoPE (4.6) is applied to q and k. Its cos/sin are computed from `position_ids` and cast to
   the dtype of the activations.
3. If a cache exists: `k, v = cache.update(k, v, slot)`. The cache stores **post-RoPE** keys
   with layout `[B, nkv, S, D]`.
4. Dispatch through `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation,
   eager)`. Arguments:
   - `scaling = head_dim ** -0.5`
   - `dropout = attention_dropout` when training, 0 otherwise
   - the attention module must expose `num_key_value_groups = nh // nkv` and
     `is_causal = True`

   The eager path is identical to the vendored Llama eager path:
   - expand KV heads by repeat-interleave;
   - compute `q k^T * scaling` and add the 4-D mask;
   - softmax in fp32, then cast back to the query dtype;
   - apply dropout;
   - matmul with v;
   - transpose to `[B, T, nh, D]`.

   It returns the probabilities as the attention weights.
5. Reshape to `[B, T, H]` and apply `o_proj`.
6. The attention module returns `(output, weights)`. Output capture records `weights` per call.
   SDPA returns None for the weights. In Transformers 5.23 only eager returns probabilities.

### 3.8 Output capture
- `output_hidden_states=True`: the tuple has `uses + 1` entries. Entries `0..uses-1` are the
  *inputs* to each decoder-layer call (entry 0 is the raw embedding). The last entry is the final
  **normed** state, equal to `last_hidden_state`.
- `output_attentions=True` (eager): `uses` tensors, each `[B, nh, T, S]`.

---

## 4. Model-specific numerics

### 4.1 Logits (`MobileLLMForCausalLM.forward`)
- `states = slice_logits_to_keep(last_hidden_state, logits_to_keep)` (default 0 keeps all).
- With `share_embedding=True`: `logits = linear(states, model.embed_tokens.weight)`, no bias.
- Otherwise: `logits = lm_head(states)`.
- Logits are always upcast with `.float()`, so bf16 and fp16 models return fp32 logits.
- With `labels`: `loss = self.loss_function(logits=logits, labels=labels,
  vocab_size=config.vocab_size, **kwargs)`. This is the standard shifted causal-LM cross entropy
  with ignore_index -100. Out-of-range labels raise an `IndexError` ("out of bounds").
- Returns `CausalLMOutputWithPast(loss, logits, past_key_values, hidden_states, attentions)`.
  Supports `return_dict=False` through `can_return_tuple`, and supports `inputs_embeds`.

### 4.2 dtype rules
- Norm statistics are computed in fp32.
- Attention softmax is computed in fp32.
- RoPE angles are computed in fp32 with autocast disabled, then cast to the activation dtype.
- `inv_freq` stays fp32 after `.to(fp16/bf16)` and after reload in low precision. The tests
  require it to equal `1/10000**(arange(0,8,2)/8)` exactly for head_dim 8.
- Logits are always fp32.

### 4.3 `pretraining_tp > 1` (source-faithful sliced projections)
With `tp = pretraining_tp > 1`, some projections are computed as weight slices **without
adding any bias, even when the module has one**:
- *Column-sliced* projection: split the weight into `tp` equal chunks along the output dim, run
  `linear(x, chunk)` for each (no bias), and concatenate along the last dim. This is
  mathematically the full matmul without bias.
- *Row-sliced* projection: split the input along its last dim into `tp` chunks, split the weight
  along the input dim into `tp` chunks, and sum the `linear(x_j, W_j)` results (no bias).
  Numerically this is a partial-sum order that differs from the full matmul by about 1e-6.

Where slicing applies:
- **MLP**: always when tp > 1. gate and up are column-sliced; down is row-sliced, applied to
  `act(gate) * up`. **All MLP biases are ignored.**
- **Attention**: only when tp > 1 **and** `config._attn_implementation != "sdpa"` (that is,
  eager, flash, flex, and so on). q, k, v are column-sliced and o is row-sliced, and all
  attention biases are ignored. Under `"sdpa"` the attention projections run as normal Linear
  layers *with* bias.

  As a result, eager and sdpa outputs differ by O(1) when `attention_bias=True` and tp > 1
  (fixture `tp2_*`: max |Δlogit| ≈ 3–4). This is intentional and must be preserved.
- **LM head**: column-sliced over `get_output_embeddings().weight` (the embedding or lm_head
  weight). It has no bias either way, then `.float()`.

Separately, the GPT-QModel definition (kept) rejects quantization when tp > 1, with the message
"dense weight slices".

### 4.4 Masks
Standard Transformers 5.23 causal masking, combined with the 2-D padding mask. Query rows that
are entirely padding are convention-dependent and are not compared.

### 4.5 Default positions with caching
Default `position_ids` continue from the cache length (`past_len`). Explicit `position_ids` are
used as given. For example, a left-padding fixture uses `cumsum(mask)-1`, with pad positions
filled with 1.

### 4.6 Rotary embedding (per attention module, stateful)
- Default frequencies: `inv_freq = 1 / theta ** (arange(0, D, 2, fp32) / D)`, with
  `theta = config.rope_theta`.
- `angles = position_ids.float()[..., None] * inv_freq`. Then `emb = cat(angles, angles)`,
  `cos = emb.cos()`, `sin = emb.sin()`, both cast to the activation dtype and broadcast over
  heads.
- Application uses the standard half-split rotation:
  `x*cos + cat(-x2, x1)*sin`, where x1 and x2 are the first and second halves of the last dim.
- There is **no** attention-scaling multiplier (unlike some HF rope types).

**linear**: positions are divided by `factor` before the angle product (in float).

**dynamic** (dynamic NTK), checked on every forward of every attention module:
- Let `L = position_ids.max() + 1`. This is the max over the *current call's* position ids
  across the whole batch, not the cache length.
- If `L > max_position_embeddings`:
  `base = rope_theta * (factor * L / max_pos - (factor - 1)) ** (D / (D - 2))`. The arithmetic
  is fp32 tensor math, always starting from `config.rope_theta` and not from the previous base.
  Then:
  - recompute `inv_freq` from this base;
  - **store it on the module** (`inv_freq` replaced, `_base = float(base)`);
  - use it for this call.
- If `L <= max_pos`: use the **currently stored** `inv_freq` unchanged. This is the stale-state
  behavior: after any long call, later short calls keep the enlarged base and do *not* go back
  to `rope_theta`. A freshly loaded model starts at `rope_theta`.
- Keys already in the cache keep the rotation they were computed with. They are never
  re-rotated.

Consequence: for dynamic RoPE, chunked cached decoding is **not** equal to a single
full-sequence forward once the context passes max_pos (fixture `rope_dynamic_sharing`:
`cached_chunk_logits` differs from `logits` by about 0.24). The fixture pins the chunked
behavior.

**Device/dtype moves** (`Module._apply`, e.g. `.to()`, `.half()`, `.cuda()`): after the move,
`inv_freq` is recomputed in fp32 on the new device from the stored `_base`. A dynamic base that
was in effect therefore survives a move.

Every layer has its own rotary module. In one forward they all update identically.

### 4.7 Failure atomicity (cache and rotary state rollback)
`MobileLLMModel.forward` and `MobileLLMForCausalLM.forward` (which also covers the head and the
loss) each wrap their work so that **any exception** restores the state from before the call.

What is snapshotted before the call:
- The DynamicCache: the list of its layer objects, plus a shallow copy of each layer object's
  attribute dict. Take references, not tensor copies (`DynamicLayer.update` concatenates into
  new tensors, so the old references stay valid).
- Every decoder layer's rotary module: `_base` and the `inv_freq` reference.

What happens on an exception:
- Truncate or restore `cache.layers` in place, keeping the same list object, so that layers
  appended during the failed call disappear.
- Restore each surviving layer's attribute dict, so each layer object is the same object with
  identical attribute *objects*.
- Restore each rotary module's `_base` and `inv_freq`.
- Re-raise the exception.

The tests check all of this for failures raised after a cache write in the first decoder layer,
in the head, and in the loss (bad label). They cover cold caches (empty `DynamicCache()`) and
warm caches, with and without sharing, and dynamic NTK. After a failed call, a retry must give
the correct logits and grow the same cache to the expected lengths.

Input-validation errors (3.1) happen before any mutation.

---

## 5. Cache and generation semantics
- Only `DynamicCache` is supported. The cache has one slot per decoder *use* (`uses` slots),
  each storing post-RoPE K/V `[B, nkv, S, D]`, and it grows lazily. After a call, every slot's
  `get_seq_length()` equals the total tokens processed.
- Chunked prefill/decoding with default positions matches the uncached full forward:
  - fp32: within 2e-6 to 1e-5;
  - bf16: within 2e-3 (test tolerance).

  The exception is dynamic RoPE beyond max_pos (4.6).
- Causality: changing the last token leaves all earlier logits bit-identical (atol=0).
- Generation goes through standard `GenerationMixin`. Because `_supports_default_dynamic_cache`
  is False, generate passes no pre-built cache and the model creates its own lazily on the first
  step. Requirements:
  - Greedy `generate` equals manual uncached argmax extension (non-dynamic RoPE).
  - Beam search (`num_beams=2`) with and without cache gives identical sequences. Beam
    reordering of the cache works through the standard DynamicCache reorder.
- `use_cache` is effectively True whenever a cache is passed and `use_cache` is not explicitly
  False.

---

## 6. Behaviors the existing tests rely on (summary checklist)
From `tests/test_mobilellm_local_port.py`:
- Definition and loader:
  - the definition uses `LocalModelLoader` and `require_trust_remote_code = False`;
  - a checkpoint carrying a hostile `auto_map` and a `.py` file loads `MobileLLMConfig` and
    `MobileLLMForCausalLM` without ever importing checkpoint Python, under both
    `trust_remote_code` values and with `GPTQ_BYPASS_SAFETY_CHECKS` set to 0, 1, or unset.
- Equation match against an independent oracle (rtol 2e-5, atol 2e-6) over:
  - kv in {1,2,4};
  - bias on or off;
  - sharing on or off;
  - share_embedding on or off;
  - left padding;
  - eager attention, including the hidden-state and attention counts and causal-triangular
    attention weights.
- RoPE default, linear, and dynamic with max_pos=4 and sharing on or off; save_pretrained writes
  only safetensors; reload is equal and `rope_parameters` round-trips.
- Cache and generation matrix: sharing × share_embedding × {eager,sdpa} × {fp32,bf16}. Covers:
  - chunked caching;
  - causality;
  - greedy and beam generation;
  - save and reload twice with bit-exact state and logits.
- Failure rollback (4.7), including dynamic-NTK rollback.
- Rejected inputs: a bad mask shape, a `StaticCache`, and a pre-built physical-size cache (with
  sharing). Each is rejected without mutation, and a retry then succeeds.
- A corrupt safetensors file raises `SafetensorError`; restoring the file allows a reload.
- `load_state_dict` (with `assign` True or False) into a model that has already run gives
  bit-exact logits. Rotary state must not depend on parameters.
- Rotated-cache oracle with explicit positions and masks, crossing max_pos=3 for
  default/linear/dynamic × eager/sdpa × sharing. The cached K/V tensors equal the oracle's
  post-RoPE K/V. The stale dynamic base is retained on a later short call, and
  `rotary_emb._base == base.item()`.
- `pretraining_tp=2` slice and bias semantics (4.3) for eager and sdpa, cache continuation, and
  reload through `MobileLLMQModel.loader.from_pretrained`.
- Loss, `inputs_embeds`, `logits_to_keep`, `return_dict=False`, and output-embedding
  replacement semantics (1.3).
- `tie_word_embeddings` with `share_embedding=False` (1.3).
- Partial-checkpoint init guard (2).
- fp32 `inv_freq` under fp16/bf16 (4.2).
- AWQ quantize / pack / save / load cycles (definition-level):
  - 14 packed linears for 2 layers;
  - layer norms change from AWQ scale folding;
  - all other dense tensors are bit-exact;
  - the quantized cache continuation stays close.

From `tests/module_tree/test_mobilellm_local_port.py`:
- Module tree and state-dict ownership on cpu and on meta.
- The sharing case keeps only 2 physical layers.
- The layer-module block plan:
  `[input_layernorm:!] [q,k,v] [o] [post_attention_layernorm:!] [gate,up] [down]`.
- AWQ fold neutrality per layer. `v->o` is exact only when kv == heads.
- Official geometry on meta for (576/1536/30/9/3), (960/2560/32/15/5), (1152/3072/40/18/6),
  (1280/3584/54/20/5), and (1600/4352/54/25/5): vocab 32000, head_dim 64, and
  `9*layers+2` state keys.

## 7. Standard vs model-specific
Standard (reuse the vendored Llama port as-is):
- RMSNorm math
- SwiGLU MLP for tp = 1
- eager/SDPA attention interface and GQA expansion
- half-split rotate
- causal mask creation
- loss
- logits_to_keep
- decoder pre-norm residual structure

Model-specific (must be written new):
- the config class and its validation;
- `share_embedding` (no lm_head; embedding-matmul head);
- consecutive per-layer sharing with per-use cache slots and the cache-size check;
- the lazily created DynamicCache and `_supports_default_dynamic_cache = False`;
- tp-sliced bias-dropping projections, including the sdpa exemption for attention;
- per-module stateful dynamic NTK with no attention-scaling factor, plus base retention across
  `_apply`;
- fp32 `inv_freq` init through guarded init;
- the strict input checks;
- the exception rollback of cache and rotary state.

Note: the vendored Llama rotary uses Transformers' rope-init functions, which have their own
dynamic implementation. MobileLLM's dynamic and linear behavior must follow 4.6 exactly.
