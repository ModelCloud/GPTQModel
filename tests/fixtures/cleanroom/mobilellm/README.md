# MobileLLM clean-room golden fixtures

These fixtures were generated with the current (FAIR-licensed) `gptqmodel.models.ports.mobilellm`
port, using torch 2.15 CPU, transformers 5.23.0, `CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=1`.
The generator script is not committed because it imports the restricted port. Running it twice
produced byte-identical files.

## Shared geometry and weights

All fixtures share one tiny geometry:

| Setting | Value |
|---|---|
| `vocab_size` | 48 |
| `hidden_size` | 32 |
| `intermediate_size` | 64 |
| `num_hidden_layers` | 2 |
| `num_attention_heads` | 4 |
| `head_dim` | 8 |
| `num_key_value_heads` | 2, unless the fixture changes it |
| `max_position_embeddings` | 32, unless the fixture changes it |
| `pad_token_id` | 0 |
| dtype | fp32 |

Weights are random, from a fixed seed:

- Linear and embedding weights: N(0, 0.3²).
- Biases: N(0, 0.2²).
- Norm gains: U(0.5, 1.5).

## Files in each fixture directory

- `config.json` and `generation_config.json`, as written by `save_pretrained`.
- `model.safetensors`: the checkpoint and its interface key names (see SPEC 1.3).
- `io.safetensors`: the inputs and the expected outputs.

## Fixtures

| dir | config changes from the shared geometry | covers |
|---|---|---|
| `base_gqa` | none (share_embedding=True, kv=2) | GQA, embedding-matmul head, default RoPE |
| `sharing_lmhead_bias_mqa` | layer_sharing, share_embedding=False, attention_bias and mlp_bias, kv=1 | consecutive layer reuse with 4 cache slots, separate lm_head, biases, MQA |
| `sharing_mha_shared_emb` | layer_sharing, kv=4 | MHA with sharing |
| `rope_linear` | rope `{"type":"linear","factor":2}`, max_pos=4 | linear RoPE, legacy `rope_scaling` alias |
| `rope_dynamic_sharing` | rope `{"type":"dynamic","factor":2}`, max_pos=3, sharing | stateful dynamic NTK (SPEC 4.6): base update, stale reuse, cache chunks |
| `tp2_bias_sharing` | pretraining_tp=2, sharing, both biases, separate lm_head | sliced projections drop bias; eager and sdpa differ in attention |
| `tp2_shared_emb` | pretraining_tp=2, attention_bias, share_embedding | tp head slicing over the embedding |
| `lmhead_tied` | share_embedding=False, tie_word_embeddings=True | HF tie; `lm_head.weight` is absent from the safetensors file |

Each fixture directory is about 150–190 KB, and the whole set is about 1.3 MB.

## Contents of `io.safetensors`

Each output key appears twice: once with the prefix `eager.` and once with `sdpa.`. The model was
reloaded with `from_pretrained(dir, attn_implementation=impl, dtype=float32)`, `eval()`, under
`inference_mode`.

### Inputs

- `input_ids`: [2, 7].
- `pad_input_ids` and `pad_attention_mask`: [2, 5]. Row 0 is left-padded and row 1 is
  right-padded.
- `pad_position_ids`: `cumsum(mask) - 1`, with padded positions set to 1.

### Outputs

The stateful outputs below have to be reproduced in the same order on one model instance, because
dynamic NTK keeps state between calls.

| key | how it is produced | shape |
|---|---|---|
| `logits` | `model(input_ids, use_cache=False)` | [2, 7, 48] |
| `hidden_states.{i}` | the same call with `output_hidden_states=True` | `uses + 1` entries; the last entry is after the final norm |
| `attentions.{i}` | the same call with `output_attentions=True` | eager only, one per use, [2, 4, 7, 7] |
| `pad_logits` | `model(pad_input_ids, attention_mask=pad_attention_mask, use_cache=False)` | [2, 5, 48] |
| `cached_chunk_logits` | default-position chunks of 1, 4 and 2 tokens, concatenated | [2, 7, 48] |
| `cache_num_slots` | number of cache slots after the chunked run | scalar |
| `cache.{slot}.keys` and `cache.{slot}.values` | final post-RoPE K/V after the chunked run | [2, kv, 7, 8] |
| `after_long_short_logits` | `model(input_ids[:, 1:3], use_cache=False)`, run next on the same instance | [2, 2, 48] |
| `fresh_short_logits` | the same input on a freshly loaded model | [2, 2, 48] |
| `pad_cached_logits` | on a fresh model: chunks of 2, 2 and 1 tokens of the padded batch, with the explicit `attention_mask[:, :end]` and `pad_position_ids` | [2, 5, 48] |
| `loss` | `labels=input_ids` | scalar |
| `logits_to_keep2` | `logits_to_keep=2` | [2, 2, 48] |
| `embeds_logits` | `inputs_embeds=embed(input_ids)` | [2, 7, 48] |
| `greedy` | on a fresh model: `generate(input_ids[:, :3], max_new_tokens=5, do_sample=False, eos_token_id=None, pad_token_id=0)` | [2, 8] |
| `beam2` | on a fresh model: `generate(input_ids[:1, :3], max_new_tokens=3, num_beams=2, do_sample=False, eos_token_id=None, pad_token_id=0)` | [1, 6] |
| `eager.bf16_logits` | reload with `dtype=bfloat16`, eager, `model(input_ids, use_cache=False)`; logits are returned as fp32 | [2, 7, 48] |

Notes:

- `uses` is `num_hidden_layers × (2 if layer_sharing else 1)`.
- `cache_num_slots` is 2, or 4 with sharing.
- In `rope_dynamic_sharing`, `after_long_short_logits` differs from `fresh_short_logits` by about
  4e-2. This is the stale-base behavior. In every other fixture the two keys are equal.

## Tolerances

The current implementation reproduces every key bit-exactly on CPU, since its output is
deterministic.

An independent oracle written only from SPEC.md (fp32, eager-style math) gave these maximum
absolute differences:

| output | eager | sdpa |
|---|---|---|
| `logits` | ≤ 9e-6 | ≤ 7e-6 |
| `hidden_states` | ≤ 3e-5 | ≤ 3e-5 |
| `pad_logits`, valid query rows only | ≤ 5e-6 | ≤ 5e-6 |
| dynamic `cached_chunk_logits`, cache K/V and `pad_cached_logits` | ≤ 7e-6 | ≤ 7e-6 |

The oracle's `greedy` ids matched exactly in all fixtures.

### Recommended assertions for the reimplementation

- fp32 logits, loss and cache tensors: `atol=5e-5, rtol=1e-5`.
- Hidden states: `atol=1e-4, rtol=1e-5`.
- `pad_logits` and `pad_cached_logits`: compare only where `pad_attention_mask == 1`. Query rows
  that are entirely padding depend on convention.
- `greedy` and `beam2` ids, `cache_num_slots`, and the state-dict key set: exact.
- `bf16_logits`: `atol=5e-2, rtol=2e-2`. The spread between bf16 and fp32 is 0.1–0.4, so this
  still catches dtype-rule mistakes. Expect a bit-exact match if the same op order is used.
- `tp2_*` fixtures: `eager.*` and `sdpa.*` intentionally differ by O(1), up to 4.4 in logits,
  because sdpa keeps the attention biases (SPEC 4.3). Compare each implementation only against its
  own prefix.
- Reference values for the cache-versus-whole-forward difference in the current implementation:
  - non-dynamic fixtures: ≤ 8e-6;
  - `rope_dynamic_sharing`: 0.24. This gap is intended (SPEC 4.6).
