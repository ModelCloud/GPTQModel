# Instella: behavioral specification (clean-room phase 1)

Target: Transformers 5.23, CPU and GPU, using GPT-QModel's vendored Apache-2.0 blocks under
`gptqmodel/models/ports/native/` (mainly `llama` and `qwen2`). The model is a dense, decoder-only causal LM. It is close
to OLMo2. It differs in three ways: it uses **pre-norm** residual blocks (OLMo2 uses post-norm), it supports two legacy
RoPE scaling modes, and its eager attention scales scores by division.

Golden fixtures: `tests/fixtures/cleanroom/instella/` (see the README there). In this document "fixture X" refers to an
entry in that directory.

---

## 1. Public interface

### 1.1 Package and class names

- The package is `gptqmodel.models.ports.instella`. It must export `InstellaConfig`, `InstellaModel` and
  `InstellaForCausalLM` (`__all__` holds exactly these three).
- `gptqmodel/models/definitions/instella.py` (`InstellaQModel`, Apache-2.0, kept) loads through
  `LocalModelLoader(config_class="gptqmodel.models.ports.instella.InstellaConfig",
  model_class="gptqmodel.models.ports.instella.InstellaForCausalLM")`. `MODEL_MAP["instella"]` points to
  `InstellaQModel`. No other MODEL_MAP key may map to it.
- `InstellaForCausalLM` subclasses the package's `PreTrainedModel` base class and also `GenerationMixin`.
  `InstellaModel` is the bare decoder. Both share a pretrained-model base class with `config_class = InstellaConfig`
  and `base_model_prefix = "model"`.
- The decoder layer class should be named `InstellaDecoderLayer`. It is listed in `_no_split_modules` and is the
  gradient-checkpointing unit. It must derive from Transformers' `GradientCheckpointingLayer`, so that
  `activation_checkpointing_enable()` / `gradient_checkpointing_enable()` work and give bit-identical gradients.
- `tests/test_awq_scaling_groups.py` imports `tests.test_instella_local_port.tiny_config`. That is test code and is
  not part of the port.

### 1.2 Configuration (`InstellaConfig`, `model_type = "instella"`)

`InstellaConfig` subclasses `transformers.PreTrainedConfig`. Set `keys_to_ignore_at_inference = ["past_key_values"]`.

| field | default | notes |
|---|---|---|
| vocab_size | 50304 | |
| hidden_size | 4096 | |
| intermediate_size | 11008 | |
| num_hidden_layers | 32 | |
| num_attention_heads | 32 | |
| num_key_value_heads | None → num_attention_heads | resolved at construction time |
| hidden_act | "silu" | used by the MLP via `ACT2FN` |
| max_position_embeddings | 2048 | threshold for dynamic NTK |
| initializer_range | 0.02 | init only |
| use_cache | True | |
| pad_token_id | 1 | also the embedding `padding_idx` |
| bos_token_id | None | |
| eos_token_id | 50279 | |
| tie_word_embeddings | False | |
| rope_theta | 10000.0 | stored as a plain attribute **and** inside `rope_parameters` |
| attention_bias | False | bias on q/k/v/o projections only |
| attention_dropout | 0.0 | |
| rms_norm_eps | 1e-5 | used by all six norms |

The token ids and `tie_word_embeddings` go to the base-class constructor. All other unknown kwargs are forwarded too.

**RoPE input normalization.** Do this before calling the base constructor, and pop both keys so the base class never
sees them:

1. Accept `rope_parameters` (the TF5 form) and `rope_scaling` (the legacy checkpoint form).
2. If `rope_parameters` is given, it takes precedence and `rope_scaling` is ignored. If it has a `rope_theta` key, that
   value replaces the `rope_theta` kwarg. Otherwise the scaling spec is the `rope_scaling` value.
3. Read the scaling kind from key `rope_type`, or else from key `type`. If the kind is `"default"`, there is no
   scaling.
4. Without scaling, store `rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}`.
5. With scaling, the kind must be `"linear"` or `"dynamic"` and `factor` must be an int or float `> 1`. Store
   `rope_parameters = {"rope_type": kind, "rope_theta": rope_theta, "factor": factor}`. Keep the factor's type: an int
   stays an int in JSON.
6. A scaling value that is not None but fails rule 5 raises `ValueError` with exactly the message
   `"Instella rope_scaling requires type linear/dynamic and factor > 1"`. This covers an unknown type, factor ≤ 1, a
   string factor, and an empty dict `{}`. Tests match the substring `rope_scaling`.
7. The saved `config.json` contains `rope_parameters` and `rope_theta`, and **never** a `rope_scaling` key. No
   `FutureWarning` may be emitted while constructing, saving or reloading. Note that `config.rope_scaling` reads back
   as `rope_parameters`: this alias is provided by the TF5 base class.

**Validation**, after setting the fields (all raise `ValueError`):
- `hidden_size % num_attention_heads != 0`, or an odd head dim (`hidden_size // num_attention_heads`), raises
  `"Instella requires an even head dimension and hidden_size divisible by num_attention_heads"`.
- `num_key_value_heads <= 0`, or `num_attention_heads % num_key_value_heads != 0`, raises
  `"num_key_value_heads must divide num_attention_heads"`. Tests match `divide`.
- A dynamic kind with head_dim ≤ 2 raises `"Dynamic NTK scaling requires head_dim > 2"`.

Every valid and invalid case above appears with its expected result in `config_cases.json`.

The default `_attn_implementation` resolved by TF5 is `sdpa`.

### 1.3 Checkpoint state_dict (exact keys)

head_dim `D = hidden_size / num_attention_heads`, `H = hidden_size`, `KV = num_key_value_heads * D`,
`I = intermediate_size`, `V = vocab_size`.

```
model.embed_tokens.weight                          [V, H]
model.layers.{i}.pre_attention_layernorm.weight    [H]
model.layers.{i}.self_attn.q_proj.weight           [H, H]     (+ .bias [H]  if attention_bias)
model.layers.{i}.self_attn.k_proj.weight           [KV, H]    (+ .bias [KV] if attention_bias)
model.layers.{i}.self_attn.v_proj.weight           [KV, H]    (+ .bias [KV] if attention_bias)
model.layers.{i}.self_attn.o_proj.weight           [H, H]     (+ .bias [H]  if attention_bias)
model.layers.{i}.self_attn.q_norm.weight           [H]        <- full projected width, NOT head_dim
model.layers.{i}.self_attn.k_norm.weight           [KV]       <- full projected width, NOT head_dim
model.layers.{i}.pre_feedforward_layernorm.weight  [H]
model.layers.{i}.mlp.gate_proj.weight              [I, H]     (never biased)
model.layers.{i}.mlp.up_proj.weight                [I, H]     (never biased)
model.layers.{i}.mlp.down_proj.weight              [H, I]     (never biased)
model.norm.weight                                  [H]
lm_head.weight                                     [V, H]     (omitted from saved file when tied)
```

- `state_dict()` contains no other keys. In particular there are no persistent rotary buffers.
- The decoder layer has **no** `input_layernorm` / `post_attention_layernorm` attributes. Tests assert that both are
  absent.
- With `tie_word_embeddings=True`, `lm_head.weight` is the same Parameter object as `model.embed_tokens.weight`
  (declare `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`). `save_pretrained` then omits
  `lm_head.weight`, and `from_pretrained` re-ties.
- Production 3B: V=50304, H=2560, I=6912, 36 layers, 32 heads, 32 KV heads, max_pos 4096, bos=eos=0. That gives 399
  tensors, and the sha256 of the sorted key names joined by `\n` is
  `f21aaded620da6bf22f2a85f3328c144644250e0975533cef7d1a56ea5185a1a`. `q_norm.weight` has shape `(2560,)`.
- Saving goes to a single `model.safetensors` with no `.bin` file. A missing weights file must raise `OSError` and
  leave nothing behind that breaks a later valid retry on the same directory.

### 1.4 Module attributes that tests and tooling touch

- `model.model.layers[i].self_attn` exposes `q_proj, k_proj, v_proj, o_proj, q_norm, k_norm` (q_norm and k_norm have
  `.weight`), plus `num_key_value_groups`, `head_dim`, `is_causal=True`, `layer_idx`, `config`. The attention
  interfaces in `ALL_ATTENTION_FUNCTIONS` (e.g. sdpa) read these.
- **Each attention module owns its own rotary module** at `self_attn.rotary_emb`. There is no shared model-level rotary
  module. It exposes:
  - `inv_freq`: a non-persistent fp32 buffer of shape `[D/2]`.
  - `scaling_factor`: the configured factor, or `1.0` without scaling.
  - a call `rotary_emb(x, position_ids) -> (cos, sin)`, each of shape `[B, T, D]` and dtype `x.dtype`.
- The attention module can be called on its own as `self_attn(hidden_states, attention_mask=None, position_ids=...,
  past_key_values=cache)`. It returns `(output, attn_weights)`.
- `mlp` exposes `gate_proj, up_proj, down_proj`. The vendored `native.qwen2.Qwen2MLP` fits this exactly: no bias,
  `down(act(gate(x)) * up(x))`.
- The layer exposes `pre_attention_layernorm`, `self_attn`, `pre_feedforward_layernorm`, `mlp`.

### 1.5 Model-class flags

- `supports_gradient_checkpointing = True`, `_no_split_modules = ["InstellaDecoderLayer"]`,
  `_skip_keys_device_placement = ["past_key_values"]`, `_supports_sdpa = True`, `_supports_attention_backend = True`.
- **Do not** declare FlashAttention or flex support. `attn_implementation="flash_attention_2"` must raise `ValueError`
  whose message contains `"Flash Attention 2"`, at model construction (this is Transformers' standard rejection).
  `flex_attention` is rejected the same way.
- Outputs are recorded through TF5's `capture_outputs` mechanism: hidden states come from decoder-layer outputs, and
  attention weights from the attention module's second return value. See §3.4.

---

## 2. Forward math

Notation: `x` has shape `[B, T, H]`. Everything below runs in the parameter dtype unless an upcast is stated.

### 2.1 RMSNorm (all six norm sites)

This is identical to `native.llama.LlamaRMSNorm`: upcast to fp32, `y = x * rsqrt(mean(x², last dim) + eps)`, cast back
to the input dtype, **then** multiply by `weight`. `eps = rms_norm_eps`. Weights initialize to ones.

### 2.2 Decoder layer (pre-norm, which is model-specific compared with OLMo2)

```
h = h + Attn(RMSNorm_pre_attention(h))
h = h + MLP(RMSNorm_pre_feedforward(h))
```

After the last layer: `last_hidden_state = RMSNorm_final(h)`, then `logits = lm_head(last_hidden_state)` (no bias).

### 2.3 Attention (model-specific Q/K normalization; otherwise Llama-style GQA)

1. `q = q_proj(x)` with width H, `k = k_proj(x)` with width KV, `v = v_proj(x)` with width KV. Biases are present if
   `attention_bias`.
2. **QK-norm over the full projected width, before splitting into heads.** Apply `q = RMSNorm_q(q)` over all H
   channels and `k = RMSNorm_k(k)` over all KV channels. v is not normalized. Tests use non-unit norm gains to catch
   per-head normalization. Per-head normalization is wrong.
3. Reshape q to `[B, nH, T, D]`, and k and v to `[B, nKV, T, D]`.
4. Compute the rotary `cos, sin` from this layer's rotary module, with `x = v` (so the output dtype is v's dtype) and
   the layer's `position_ids`. Apply RoPE to q and k with the vendored `native.llama.apply_rotary_pos_emb`
   (rotate-half convention: `x*cos + rotate_half(x)*sin`, with cos and sin broadcast over heads).
5. If a cache is given, call `cache.update(k, v, layer_idx)` with the **already rotated** k. Attention then uses the
   full k and v that update returns.
6. Dispatch through `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, <eager>)`. Pass
   `dropout = attention_dropout if training else 0.0` and `scaling = D ** -0.5`, and forward the remaining kwargs.
7. Reshape the output `[B, T, nH, D]` to `[B, T, H]`, then apply `o_proj`.

**Eager kernel.** This is model-specific in one numerically visible way:
- Expand k and v to nH heads with `native.llama.repeat_kv` (head j uses KV head `j // (nH/nKV)`).
- `scores = (q @ kᵀ) / sqrt(D)`: a **division** by the Python float `math.sqrt(D)`. The vendored Llama eager kernel
  multiplies by `scaling` instead, and the two give different bits (verified). Use division to reproduce the fixtures
  exactly.
- If an attention mask is given, add the 4-D additive mask, sliced on its last dimension to the key length.
- Softmax over the last dimension **in fp32**, then cast to q's dtype. Apply dropout (training only), then compute
  `probs @ v`. Transpose to `[B, T, nH, D]`, make it contiguous, and return it with `probs`.

**SDPA and other backends:** use the standard Transformers integration with `scaling = D**-0.5` and `is_causal=True`.
Eager and SDPA agree to about 3e-6 in fp32 on valid positions.

### 2.4 Rotary embedding (model-specific scaling and state)

Let `θ = rope_parameters["rope_theta"]` and `D = head_dim`. Set `kind` and `factor` from `rope_parameters`, with
factor = 1.0 when absent. Initial frequencies: `inv_freq = 1 / (θ ** (arange(0, D, 2).float() / D))` in fp32, computed
as a Python-scalar base raised to an fp32 tensor exponent. This is the same as Llama's default RoPE.

Each call `(x, position_ids)`, with position_ids of shape `[Bp, T]` (Bp may be 1 and broadcast over the batch):

- `kind == "linear"`: `pos = position_ids.float() / factor`. inv_freq is unchanged.
- `kind == "dynamic"`: let `L = max(position_ids) + 1`, a 0-d integer tensor over the whole call, all batch rows. Note
  that this is **not** the cache length.
  - If `L > max_position_embeddings`, compute a new base in fp32 tensor arithmetic, in this order:
    `base = θ * ((factor * L) / max_position_embeddings - (factor - 1)) ** (D / (D - 2))`.
    The exponent is a Python float. Then `new_inv = 1 / (base ** (arange(0, D, 2, device=x.device).float() / D))`.
  - **Assign `new_inv` to the module's `inv_freq` buffer exactly once** in the call. The assignment replaces the
    attribute; it is not an in-place copy. The call then **uses its own local `new_inv`**, never a re-read of the
    attribute. A test swaps the attribute during assignment and checks that cos and sin still come from the local
    value.
  - If `L ≤ max_position_embeddings`: **do not reset**. Use whatever `inv_freq` currently holds. After one long call,
    later short calls keep using the enlarged-base ("warm") frequencies. This stateful behavior is intentional and
    covered by a fixture (`stateful.*` in `rope_dynamic`: `short_after ≠ short_before`).
  - The positions themselves are not divided.
- `"default"`: neither positions nor frequencies change.

Then, with autocast disabled and in fp32: `freqs[b, t, i] = pos[b, t] * inv_freq[i]`, computed as the batched matmul
`inv_freq[None, :, None].expand(Bp, -1, 1) @ pos[:, None, :]`, transposed. Then `emb = concat(freqs, freqs)` on the
last dimension, and `cos = emb.cos()`, `sin = emb.sin()`, cast to `x.dtype`. There is no attention-scaling multiplier.

Consequences that tests check:
- Each layer's rotary module updates independently. In one forward pass all layers see the same L, so they end with
  identical buffers.
- Cached keys keep the rotation they had when written. When a later call crosses the dynamic threshold, old keys are
  not rotated again. Only new q and k use the new frequencies. As a result, cached dynamic decoding beyond max_pos
  differs from a single full forward (fixture tolerance about 0.39 abs). That difference is expected behavior.
- **Reload rebuilds the default frequencies.** In the model's `_init_weights` hook, for every rotary module, call the
  base-class init and then refill `inv_freq` with the initial (unscaled-θ) frequencies on the buffer's device. TF5
  materializes non-persistent buffers from meta with uninitialized storage, so without this hook RoPE would be garbage
  after `from_pretrained`. A warm dynamic buffer is never saved: a reloaded model starts cold.

### 2.5 MLP

SwiGLU, no biases: `down_proj(act(gate_proj(x)) * up_proj(x))`, with `act = ACT2FN[hidden_act]` (silu). Use the
vendored `native.qwen2.Qwen2MLP` exactly as it is.

### 2.6 Embedding

`nn.Embedding(V, H, padding_idx=pad_token_id)`. The padding index only affects init (zeroed row) and gradients. Output
values come from the checkpoint.

---

## 3. Model-level behavior

### 3.1 `InstellaModel.forward(input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs)`

Decorate it with TF5's `merge_with_config_defaults` and `capture_outputs`, which handle `use_cache`,
`output_hidden_states`, `output_attentions` and `return_dict` defaults.

- Exactly one of `input_ids` and `inputs_embeds` must be given. Otherwise raise `ValueError`.
- A non-None `past_key_values` that is not a `transformers.cache_utils.Cache` raises `TypeError` with a message
  containing `"Transformers Cache"`. A legacy tuple must not be accepted.
- `inputs_embeds = embed_tokens(input_ids)` when not given. The two paths are bit-identical, as checked by fixtures.
- If `use_cache` is set and no cache is given, create `DynamicCache(config=self.config)`.
- If `position_ids` is None: `arange(T) + cache.get_seq_length()` (0 without a cache), shape `[1, T]`. This
  **ignores padding**, and generation relies on it (§3.3).
- Build the mask with `transformers.masking_utils.create_causal_mask(config=..., inputs_embeds=...,
  attention_mask=..., past_key_values=..., position_ids=...)`. It is standard causal with 2-D padding. There is no
  sliding window.
- Run every layer with `(hidden, attention_mask=mask, position_ids=..., past_key_values=..., use_cache=..., **kwargs)`.
- Return `BaseModelOutputWithPast(last_hidden_state=final_norm(h), past_key_values=cache if use_cache else None)`.
- **Partial-failure semantics:** if layer k raises, layers before k have already written into the cache. Do no
  rollback. The caller has to `cache.reset()` before a retry, and after reset the retry must be bit-identical to a
  fresh cache.

### 3.2 `InstellaForCausalLM.forward(..., labels=None, use_cache=None, logits_to_keep=0, **kwargs)`

Decorate it with `can_return_tuple`, so that `return_dict=False` returns a tuple.
- `logits_to_keep`: an int n keeps the last n positions, and 0 keeps all of them. A tensor or slice is used directly
  as an index on the sequence dimension.
- `loss = self.loss_function(logits=..., labels=..., vocab_size=V, **kwargs)`. This is the standard shifted causal-LM
  cross-entropy.
- Return `CausalLMOutputWithPast(loss, logits, past_key_values, hidden_states, attentions)`.

### 3.3 Caching and generation

- Uses the standard `GenerationMixin` with no overrides. DynamicCache and StaticCache both work. Continuing a prefix
  through the cache must match the full forward on valid (non-pad) positions within about 2e-6 in fp32 (non-dynamic
  rope).
- `generate` in TF5.23 passes `position_ids=None` to forward, as observed. Positions therefore come from §3.1 (arange
  plus cache length), and left padding is **not** subtracted. Greedy and beam fixtures depend on this.
- Cached and uncached greedy and beam generation give identical token ids. Beam search reorders the cache through the
  standard Cache API.
- The `generation_config` comes from the model config (bos, eos, pad, use_cache).

### 3.4 Output recording

- `output_hidden_states=True` returns `num_hidden_layers + 1` tensors: `[embeddings, out(layer 0), …, out(layer
  L-2), final_norm(out(layer L-1))]`. The last entry equals `last_hidden_state`. This is TF5 `capture_outputs`
  behavior with `_can_record_outputs = {"hidden_states": <decoder layer class>, "attentions": <attention class>}`.
- `output_attentions=True` (eager) returns one probability tensor `[B, nH, T, S]` per layer. It is strictly causal,
  so everything above the diagonal is 0.

### 3.5 Padding note

On query positions that are themselves padding (left-pad rows), the outputs are not meaningful and differ between
eager and SDPA, by up to about 5 abs. Fixtures and tolerances cover valid positions only.

---

## 4. Behavior the GPT-QModel tests rely on (summary)

From `tests/test_instella_local_port.py`, `tests/module_tree/test_instella_local_port.py`,
`tests/test_instella_support.py` and `tests/test_awq_scaling_groups.py`:

1. Loading through `InstellaQModel.loader` / `GPTQModel.load` uses the packaged classes even when `config.json` has a
   malicious `auto_map` and `trust_remote_code=True`. `resolve_trust_remote_code` returns False. This is mostly
   handled by the definition and loader, but the config class must load from `config.json` through
   `load_model_config`.
2. Eager logits match an independent reimplementation of §2 (rtol 2e-5, atol 2e-6) for KV heads in {1, 2, 4}, with
   and without bias, and for the none, linear and dynamic rope modes, with right padding.
3. Cache continuation (Dynamic and Static), causality (changing the last token leaves earlier logits bit-identical),
   and cached == uncached greedy, across fp32, fp16 and bf16 and both eager and sdpa.
4. safetensors round trip: logits bit-exact; `rope_theta` and `rotary_emb.scaling_factor` restored; inv_freq equal to
   the cold default for θ; the state_dict key set unchanged.
5. Partial cache failure, then reset, then a bit-exact retry (§3.1).
6. `load_state_dict(strict=True, assign=True/False)` and tied/untied reload are bit-exact, and lm_head is tied iff
   configured.
7. Rejection of flash_attention_2 (§1.5); beam search cached == uncached; `output_attentions` and
   `output_hidden_states` lengths; loss is finite; tuple return; `logits_to_keep=1` equals the last position.
8. Dynamic RoPE: a cross-threshold call through the attention module leaves the first cached keys unchanged; inv_freq
   equals the enlarged-base formula with L = 10, max = 8, factor 2; a later short call keeps the warm buffer; reload
   gives cold frequencies; the call snapshot is used after buffer replacement (§2.4).
9. Config validation messages and the `rope_parameters` serialization (§1.2).
10. AWQ, via the definition's module tree (unchanged): the scale folds are
    `pre_attention_layernorm → q,k,v`, `v_proj → o_proj` (only when nKV == nH), `pre_feedforward_layernorm → gate,up`,
    and `up_proj → down_proj`. The full-width q_norm and k_norm weights must **not** be modified by AWQ, and folds
    must preserve layer outputs. This holds automatically if the module names and shapes in §1.3 and §1.4 are kept.
    The quantizable linears per layer are exactly the 7 projections.
11. The gradients with checkpointing on are bit-identical to those without it, for eager and sdpa.

## 5. Standard vs model-specific

- Standard (reuse vendored Llama/Qwen2): RMSNorm, rotate-half RoPE application, repeat_kv, SwiGLU MLP, the sdpa path,
  causal masking, the cache, generation, the loss.
- Model-specific: the pre-norm layer names; the full-width Q/K RMSNorm placed before the head split; a rotary module
  per attention layer with the linear and dynamic-NTK formulas and the stateful, publish-then-use-snapshot buffer; the
  eager kernel's division by sqrt(D); the config's legacy `rope_scaling` → `rope_parameters` conversion and validation
  rules; rebuilding inv_freq on reload.
