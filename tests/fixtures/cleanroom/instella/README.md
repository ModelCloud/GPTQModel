# Instella clean-room golden fixtures

These were generated with the current port (`gptqmodel/models/ports/instella`) on Transformers 5.23.0 and torch 2.15
CPU, with `CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=1`, `torch.use_deterministic_algorithms(True)`, all in fp32.
The generator script is not committed because it imports the restricted port. Total size is about 0.88 MB.

## Model variants (each directory has `config.json`, `generation_config.json`, `model.safetensors`, `expected.safetensors`)

Common dims: vocab 47, hidden 32, intermediate 48, 2 layers, 4 heads (head_dim 8), max_position_embeddings **8**,
pad=1, bos=2, eos=3, rms_norm_eps 1e-5. The weights are random: N(0, 0.2²) for matrices, N(0, 0.1²) for biases, and
norm gains set to `linspace(0.6, 1.4)`, which is non-unit so that per-head normalization mistakes are caught.

| dir | covers |
|---|---|
| `gqa_default` | 2 KV heads (GQA), no bias, default RoPE θ=1e4, untied |
| `mha_bias` | 4 KV heads (MHA), `attention_bias=True` on q/k/v/o |
| `mqa` | 1 KV head (MQA); k_norm width 8 |
| `rope_linear` | legacy `rope_scaling={"type":"linear","factor":2.0}`, θ=2e4 |
| `rope_dynamic` | legacy `rope_scaling={"type":"dynamic","factor":2.0}` (dynamic NTK, stateful buffer) |
| `tied` | `tie_word_embeddings=True` (no `lm_head.weight` in the file) |

Load each model with `from_pretrained(dir, attn_implementation=<backend>)` **fresh for every entry**, because dynamic
RoPE is stateful. The one exception is `stateful.*`, which uses a single instance.

## Inputs (`inputs.safetensors`)

`input_ids` [2,12]: row 0 is unpadded and row 1 is left-padded with 2 pad tokens. `attention_mask` is `ids != 1`.
`position_ids = clamp(cumsum(mask) - 1, 0)`. Since max position is 11, L = 12 > 8, so every 12-token call goes through
the dynamic-NTK branch.

## Entries in `expected.safetensors` (prefix `eager.` or `sdpa.`)

| key | how produced |
|---|---|
| `short.logits`, `short.hidden_states.{0,1,2}`, `short.attentions.{0,1}` (eager only) | `ids[:1,:6]`, default positions, `use_cache=False`, output_hidden_states (and attentions for eager) |
| `padded.logits` | full `ids`, `attention_mask`, `position_ids`, no cache |
| `loss` | `ids[:1]` with `labels=ids[:1]`, no cache |
| `cache_{dynamic,static}.step{0,1,2}` | prefill `[:, :4]`, then `[:, 4:6]`, then `[:, 6:]`. Each step passes the mask up to the current end and the matching position slice. Static uses `max_cache_len=16` |
| `cache_dynamic.{keys,values}.{layer}` | DynamicCache contents after the 3 steps ([2,nKV,12,8], post-RoPE keys) |
| `generate_greedy_cached` / `_uncached` | `generate(ids[:, :6], attention_mask=mask[:, :6], max_new_tokens=8, do_sample=False, eos_token_id=None, pad_token_id=1)`. No position_ids are passed |
| `generate_beam3` | `generate(ids[:, :3], num_beams=3, max_new_tokens=4, do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=1)` |
| `inputs_embeds.logits` | `inputs_embeds=embed_tokens(ids[:1,:6])` |
| `logits_to_keep1` | `ids[:1,:6]`, `logits_to_keep=1` |
| `stateful.short_before`, `stateful.long`, `stateful.short_after`, `stateful.inv_freq_after.layer{0,1}` | **one** model instance: a 6-token call, then the 12-token `ids[:1]`, then the 6-token call again (no cache), then the buffers |
| `rotary.inv_freq_initial`, `rotary.pos0_11.{cos,sin}`, `rotary.inv_freq_after_pos0_11` | layer-0 `self_attn.rotary_emb` of a fresh eager model, called as `(zeros(1,2,8), [[0..11]])` |

## Tolerances achieved by the current implementation (fp32 CPU; full numbers in `tolerances.json`)

- **Bit-exact (atol=rtol=0):** a rerun of every entry on a fresh load reproduces the saved tensors. This was verified
  for eager `padded.logits`, and the computation is deterministic for all entries. A clean-room implementation that
  follows SPEC.md (including the eager division by sqrt(D) and the fp32 RoPE order) should reproduce the eager entries
  bit-exactly. Accept `atol=2e-6, rtol=2e-5` as a fallback for floating-point reassociation.
- Generated token ids must match exactly, both cached and uncached, with cached == uncached for every variant and
  backend.
- `inputs_embeds.logits == short.logits`: bit-exact.
- `logits_to_keep1` vs `short.logits[:, -1:]`: ≤ 7.2e-7.
- eager vs sdpa on **valid (non-pad) positions**: ≤ 3.1e-6. Padded query positions (row 1, cols 0-1) are undefined
  and differ by up to ~5. Exclude them.
- Cache continuation, concatenating the steps, vs `padded.logits` on valid positions: ≤ 1.9e-6, except `rope_dynamic`,
  where it is **0.393**. That is expected, because the prefill keys were rotated with cold frequencies (L ≤ 8) and are
  not rotated again (SPEC §2.4).
- `rope_dynamic` `stateful.short_after - short_before`: max abs 0.0654, which is non-zero by design because the warm
  buffer persists. It is 0 for the other variants. `inv_freq_after` equals `1/(b**(arange(0,8,2)/8))` with
  `b = 1e4*(2*12/8 - 1)**(8/6)`.

## `config_cases.json`

`config_cases.json` lists `InstellaConfig` construction cases, each run on top of the common dims above.
- `valid`: each case gives the input kwargs, the resulting `rope_parameters`, `rope_theta` and `num_key_value_heads`,
  and the saved `config.json` `rope_parameters`. `saved_has_rope_scaling` must be false.
- `invalid`: each case gives the kwargs and the expected exception type and exact message.
