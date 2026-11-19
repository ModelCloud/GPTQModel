# MiniCPM-o: behavioral specification of the ChatTTS-derived pieces

Audience: the phase-2 engineer who reimplements the restricted symbols listed in `SCOPE.md`
without seeing them. All other port code stays. This document covers the replaced symbols and the
kept code they plug into. Golden data is in `tests/fixtures/cleanroom/minicpm_o/` (see its README).
Target: Transformers 5.23, torch CPU float32, `vector-quantize-pytorch` >= 1.27.15 (fixtures were
made with 1.34.1).

Notation: `B` is batch, `T` is time frames, `K` is the number of audio codebooks (`num_vq`, 4 in
real checkpoints), `V` is the audio vocabulary (`num_audio_tokens`, 626 in real checkpoints; the
last id, `V-1` = 625, is the audio eos). "Exact" means bit-identical on CPU fp32.

---------------------------------------------------------------------------------------------------

## 1. Interface that must stay compatible

### 1.1 Module locations and names

The kept code imports these names, so they must keep existing at the same module paths:

| Name | Module | Used by |
|---|---|---|
| `ConvNeXtBlock`, `GFSQ`, `DVAEDecoder`, `DVAE` | `tts_26.py` | `ConditionalChatTTS.__init__` (kept) calls `DVAE()` with no arguments and stores it as `self.dvae`. The other three are internal to the DVAE but keep the names. |
| `CustomRepetitionPenaltyLogitsProcessorRepeat` | `tts_26.py` and `modeling_minicpmo.py` | Built by the two `gen_logits` functions. It also appears in a type hint of `generate`. |
| `gen_logits` | `tts_26.py` | `modeling_minicpmo_26.py` (`from .tts_26 import gen_logits`) |
| `gen_logits` | `modeling_minicpmo.py` | Same-module callers in `MiniCPMO`, `MiniCPMODuplex` and `MiniCPMTTS` |
| `ConditionalChatTTS.generate`, `ConditionalChatTTS.decode_to_mel_specs` | `tts_26.py` | `modeling_minicpmo_26.py` |

Call signatures that callers use. Keep these parameter names and defaults:

- `gen_logits(num_code, top_P=0.7, top_K=20, repetition_penalty=1.0)`. This is the 2.6 version.
  Note the upper-case `P` and `K`.
- `gen_logits(num_code, top_p=0.7, top_k=20, repetition_penalty=1.0)`. This is the 4.5 version.
- `CustomRepetitionPenaltyLogitsProcessorRepeat(penalty, max_input_ids, past_window)`. Instances are
  callables `(input_ids, scores) -> scores`.
- `ConditionalChatTTS.generate(input_ids, past_key_values, temperature, eos_token,
  streaming_tts_text_mask=None, force_no_stop=False, min_new_token=10, max_new_token=50,
  logits_warpers=[], logits_processors=[], show_tqdm=False)`. Callers always pass keywords.
- `ConditionalChatTTS.decode_to_mel_specs(result_list)`.
- `DVAE.forward(inp, mode="decode")`. `mode` is `"decode"` or `"encode"`.
- `generate` returns the kept `ConditionalChatTTSGenerationOutput(new_ids, audio_input_ids,
  past_key_values, finished)`.

### 1.2 Config fields used (`ConditionalChatTTSConfig`, `model_type="conditional_chattts"`, kept)

`generate` reads these attributes, which the kept `ConditionalChatTTS.__init__` copies from the
config: `num_spk_embs` (default 1), `use_speaker_embedding` (True), `streaming_text_reserved_len`
(300), `streaming_text_chunk_size` (10), `audio_bos_token_id` (21132), `num_vq` (4) and
`num_audio_tokens` (626). It also uses the submodules `emb_text`, `emb_code[k]`, `head_code[k]` and
`model`, which is the vendored Llama decoder (`.native.LlamaModel`).

The DVAE geometry is **not** config-driven. It is fixed: 100 mel bins, 512/1024/256 widths, 12
blocks per stack, and 4 codes per frame. `num_mel_bins` in the config is not consulted.

### 1.3 Checkpoint state_dict keys under `tts.dvae.` (exact strings and shapes; real 2.6 checkpoint)

    dvae.coef                                        (1, 100, 1)
    dvae.downsample_conv.0.weight / .bias            (512, 100, 3) / (512,)
    dvae.downsample_conv.2.weight / .bias            (512, 512, 4) / (512,)
    dvae.{encoder,decoder}.conv_in.0.weight / .bias  (128, 512, 3) / (128,)
    dvae.{encoder,decoder}.conv_in.2.weight / .bias  (256, 128, 3) / (256,)
    dvae.{encoder,decoder}.decoder_block.{0..11}.coef            (256,)
    dvae.{encoder,decoder}.decoder_block.{i}.dwconv.weight/.bias (256, 1, 7) / (256,)
    dvae.{encoder,decoder}.decoder_block.{i}.norm.weight/.bias   (256,) / (256,)
    dvae.{encoder,decoder}.decoder_block.{i}.pwconv1.weight/.bias (1024, 256) / (1024,)
    dvae.{encoder,decoder}.decoder_block.{i}.pwconv2.weight/.bias (256, 1024) / (256,)
    dvae.encoder.conv_out.weight                     (1024, 256, 1)   (no bias)
    dvae.decoder.conv_out.weight                     (512, 256, 1)    (no bias)
    dvae.out_conv.weight                             (100, 512, 3)    (no bias)
    dvae.vq_layer.quantizer.rvqs.{0,1}.project_in.weight / .bias   (4, 512) / (4,)
    dvae.vq_layer.quantizer.rvqs.{0,1}.project_out.weight / .bias  (512, 4) / (512,)

These are the only persistent tensors. The quantizer's derived constants (level counts, bases,
per-stage scales, clamp values and materialized codebooks) must **not** appear in the state_dict.
In `vector-quantize-pytorch` they are non-persistent buffers. The module-tree test hashes the full
2.6 key/shape list (1599 keys), so adding or removing any key fails it. The key naming under
`vq_layer.quantizer.rvqs.{g}.project_{in,out}` is exactly what
`vector_quantize_pytorch.GroupedResidualFSQ(dim=1024, levels=[5,5,5,5], num_quantizers=2, groups=2)`
produces when stored as attribute `quantizer` of a module stored as `vq_layer`. Reusing that
dependency is the simplest way to keep the keys.

The `head_code.{k}` linears are built by kept code with torch weight-norm parametrization (keys
`head_code.{k}.parametrizations.weight.original0` (V,1) and `...original1` (V,H)). The effective
weight is row-wise `g * v / ||v||`.

---------------------------------------------------------------------------------------------------

## 2. ConvNeXt-1D residual block (`ConvNeXtBlock`)

Constructor arguments: `dim`, `intermediate_dim`, `kernel`, `dilation` and
`layer_scale_init_value=1e-6`. A fifth forward argument for conditioning is accepted and ignored.
Parameter names: `dwconv`, `norm`, `pwconv1`, `pwconv2` and `coef`.

Input and output shape: `(B, dim, T)`. The block preserves `T` because the depthwise padding is
`dilation * (kernel // 2)`.

1. `h = DepthwiseConv1d(x)`. This has `dim` groups, kernel `kernel`, dilation `dilation`, padding
   `dilation*(kernel//2)` and a bias.
2. Move channels last, then apply `LayerNorm(dim, eps=1e-6)` with affine weight and bias.
3. Apply `Linear(dim -> intermediate_dim)`, then exact (erf) GELU, then
   `Linear(intermediate_dim -> dim)`. Both linears have a bias.
4. If a layer scale exists, multiply per channel by `coef` (shape `(dim,)`). It exists iff
   `layer_scale_init_value > 0` and is initialized to that value times ones. With a value of 0
   there is no `coef` key and no scaling.
5. Move channels back to `(B, dim, T)` and add the input `x` (residual).

Inside the DVAE every block uses `kernel=7`, `dilation=2` (padding 6) and
`intermediate_dim = 4*dim`.

## 3. Convolutional stack (`DVAEDecoder`)

Constructor: `idim, odim, n_layer=12, bn_dim=64, hidden=256, kernel=7, dilation=2, up=False`.
`up` is stored and unused. Submodule names: `conv_in` (a 3-element sequential where index 1 is a
parameterless GELU), `decoder_block` (a list of blocks) and `conv_out`. `forward(x,
conditioning=None)` ignores the conditioning argument, and passing one changes nothing.

`(B, idim, T) -> (B, odim, T)`:
1. `Conv1d(idim -> bn_dim, k=3, stride 1, pad 1, bias)`, then exact GELU, then
   `Conv1d(bn_dim -> hidden, k=3, pad 1, bias)`.
2. `n_layer` ConvNeXt blocks (section 2) with `dim=hidden`, `intermediate_dim=4*hidden` and the
   given `kernel` and `dilation`.
3. `Conv1d(hidden -> odim, k=1, no bias)`.

## 4. Grouped residual finite-scalar quantizer wrapper (`GFSQ`)

Constructor: `dim, levels, G, R, eps=1e-5, transpose=True`. The DVAE always uses `dim=1024`,
`levels=(5,5,5,5)`, `G=2` and `R=2`. Also store `n_ind = prod(levels)` (625), `eps`, `transpose`,
`G` and `R` as attributes. `eps` is otherwise unused.

**Dependency handling (test-relied).** Import `GroupedResidualFSQ` from `vector_quantize_pytorch`
lazily, at construction. If that import fails with a `ModuleNotFoundError` whose `name` is
`"vector_quantize_pytorch"`, raise `ImportError` with a message that contains the literal text
`gptqmodel[minicpm-o]`. The current text is: "MiniCPM-o 2.6 TTS requires vector-quantize-pytorch.
Install it with pip install 'gptqmodel[minicpm-o]'." Re-raise any other ModuleNotFoundError
unchanged. Construction must also succeed inside a `with torch.device("meta"):` context, which the
checkpoint-layout test uses. The current code builds the quantizer under an explicit CPU device
context and then moves it to whatever device is current.

Construct the dependency as `GroupedResidualFSQ(dim=dim, levels=list(levels), num_quantizers=R,
groups=G)` and keep it as attribute `quantizer`.

### 4.1 Code layout

A frame carries `G*R` = 4 integer codes in [0, 625). Code channel `c` belongs to group
`g = c // R` and residual stage `r = c % R`, so the order is (g0r0, g0r1, g1r0, g1r1).

### 4.2 Codes to features (used by DVAE decode)

Input `(B, G*R, T)` integer codes when `transpose` is true; otherwise `(B, T, G*R)`.
Output `(B, dim, T)` float, or `(B, T, dim)` when `transpose` is false. The math, which matches
`vector-quantize-pytorch` exactly and was verified bit-exact against the fixtures:

- Split an index into 4 base-5 digits, least significant first: `d_j = (idx // 5**j) % 5` for j=0..3.
- Map each digit to a code value `q_j = d_j * (2/(5-1)) - 1 = d_j/2 - 1`, which lies in
  {-1, -0.5, 0, 0.5, 1}.
- For group g, sum over stages `r` of `q(idx_{g,r}) * 5**(-r)` in float32. That is the stage-0
  code plus 0.2 times the stage-1 code. The result is a 4-vector.
- Apply that group's `project_out` (Linear 4 -> 512, with bias).
- Concatenate the groups along features: group 0 gives features 0..511 and group 1 gives 512..1023.

Calling `quantizer.get_output_from_indices(indices)` with indices arranged as `(G, B, T, R)`
produces exactly this.

### 4.3 Features to codes (used only by DVAE encode; no caller in the port)

`forward(x)`: input `(B, dim, T)` (with `transpose` true). Output `(B, G*R, T)` codes in the
layout of section 4.1, in the dtype the dependency returns (int32 with 1.34.1). The result is a
transposed, non-contiguous view. Quirk: the current implementation transposes its argument
*in place*, so the caller's tensor is left as `(B, T, dim)`. No caller depends on this.
Quantization is delegated to `GroupedResidualFSQ.forward` in eval mode. In 1.34.1 that does the
following per group: `project_in` (512 -> 4), then the soft clamp `z = tanh(z/c)*c` with
`c = 1 + 1/(5-1) = 1.25`, then for each stage r: `y = residual / 5**(-r)`; digit
`d = floor(4*(clamp(y,-1,1)+1)/2 + 0.5)`; quantized `q = d/2 - 1`; `residual -= q * 5**(-r)`.
The index is `sum_j d_j * 5**j`. Encode results depend on the dependency version. Decode results
do not, because for odd level counts the code values are the same under every FSQ variant.

## 5. DVAE (`DVAE`)

No constructor arguments. Submodule names: `coef`, `downsample_conv`, `encoder`, `decoder`,
`out_conv` and `vq_layer` (see 1.3). `coef` is a learned per-mel-bin scale of shape `(1, 100, 1)`.
Its initializer draws `torch.rand(100)`, which consumes the global RNG at construction but does not
matter once weights are loaded.

- `downsample_conv`: `Conv1d(100->512, k3, s1, p1)`, then GELU, then `Conv1d(512->512, k4, s2, p1)`,
  then GELU. The keys at indices 0 and 2 are what this section's structure implies.
- `encoder`: stack (section 3) with idim 512, odim 1024, hidden 256, 12 layers, bn_dim 128.
- `decoder`: stack with idim 512, odim 512, hidden 256, 12 layers, bn_dim 128.
- `out_conv`: `Conv1d(512 -> 100, k3, p1, no bias)`.
- `vq_layer`: GFSQ(1024, (5,5,5,5), G=2, R=2).

The whole forward runs under `torch.inference_mode()`, so its outputs are inference tensors.

**Decode** (`mode="decode"`, the default). Input: codes `(B, 4, T)` (int64 from callers).
1. Use the GFSQ to turn the codes into features `F` of shape `(B, 1024, T)` (4.2).
2. Interleave in time. Split `F` into two 512-channel halves, `F0` = channels 0..511 and
   `F1` = channels 512..1023. Build `X` of shape `(B, 512, 2T)` with `X[:, :, 2t] = F0[:, :, t]` and
   `X[:, :, 2t+1] = F1[:, :, t]`. So group 0's reconstruction supplies even frames and group 1's
   supplies odd frames.
3. `mel = out_conv(decoder(X)) * coef`, broadcast over batch and time.
4. Return `(B, 100, 2T)` in the module dtype.

(The current code also accepts `vq_layer is None`, in which case the input is treated directly as
features. This never happens in practice.)

**Encode** (`mode="encode"`). Input: an *unbatched* mel `(100, T)`. A copy is used and the
caller's tensor is not modified.
1. Divide each mel bin by `coef`.
2. Apply `downsample_conv`, giving `(512, floor((T-2)/2)+1)` (`T=12` gives 6 and `T=9` gives 4).
   Then add a batch dimension of 1.
3. Apply `encoder`, giving `(1, 1024, T')`.
4. Apply the GFSQ forward (4.3), giving `(1, 4, T')` codes.

Decoding encoder output is a valid round trip, though not an identity. The fixture
`roundtrip.mel` records it.

## 6. Repetition penalty processor (`CustomRepetitionPenaltyLogitsProcessorRepeat`)

The 2.6 copy and the 4.5 copy behave identically.

Constructor `(penalty, max_input_ids, past_window)`. Validation: if `penalty` is not an instance
of Python `float`, or is not `> 0`, raise `ValueError` with the message
``"`penalty` has to be a strictly positive float, but is {penalty}"``. So the int `2`, `0.0`,
`-1.0` and `"1.5"` all raise, while numpy float64 passes because it subclasses float.
`max_input_ids` and `past_window` are not validated.

Call `(input_ids, scores)`. `input_ids` is int64 `(N, L)`, the code history of each row. `scores`
is float `(N, V)`, the logits of each row. Result: a new tensor with the same shape and dtype as
`scores`, contiguous. The input is never modified.

1. If `L > past_window`, keep only the last `past_window` columns of `input_ids`.
2. Count occurrences: `count[n, v]` is the number of times `v` appears in the kept history of
   row `n`. Every id must be in `[0, V)`, and an out-of-range id is an error. Empty histories give
   zero counts.
3. Rows with index `n >= max_input_ids` get all their counts set to 0. The cutoff applies to
   the **row** index, not to token ids. In practice N = B*K is far below `max_input_ids`
   (`num_code`), so this never fires, but it is the specified behavior, and fixture case
   `p1_3_rows3_long` covers it.
4. `alpha = penalty ** count`, elementwise in float (default float dtype).
5. Where `scores < 0` the output is `scores * alpha`. Elsewhere, including 0, +inf and NaN,
   it is `scores / alpha`. `-inf` stays `-inf`.

## 7. Sampler factory (`gen_logits`)

### 7.1 Behavior (both copies)

Returns a pair `(warpers, processors)` of Python lists:
- `warpers`: `TopPLogitsWarper(top_p, min_tokens_to_keep=3)` if `top_p is not None`, then
  `TopKLogitsWarper(top_k, min_tokens_to_keep=3)` if `top_k is not None`. The order is P before K.
  Both are the stock Transformers 5.23 classes, so their validation applies: top_p must be a float
  in [0,1] and top_k a positive int.
- `processors`: `[CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16)]`
  if `repetition_penalty is not None and repetition_penalty != 1`, otherwise `[]`. The window is
  fixed at 16. The ValueError from section 6 surfaces for, say, an int `2`.

### 7.2 Signatures

The 2.6 copy uses keywords `top_P`, `top_K` and the 4.5 copy uses `top_p`, `top_k`. Defaults
everywhere: 0.7, 20, 1.0.

### 7.3 How callers consume it (kept code; for orientation)

The callers below are kept code. They do not need changes, but they constrain the factory's
output:

- 2.6 (`modeling_minicpmo_26.py`) and the 4.5 `MiniCPMTTS` loops apply every processor and then
  every warper, each as `f(history, logits)`. They skip both on the step that consumes the
  audio-bos token.
- `utils.TTSStreamingGenerator` does the same with its own history tensor.
- `MiniCPMODuplex` stores the *whole pair* as `tts_logits_processors` and passes it on to
  `MiniCPMTTS.generate_chunk`. That method rebuilds its own processors with `gen_logits` and
  ignores the argument, so the pair is never iterated. Keep the return type a 2-tuple of lists so
  that unpacking `w, p = gen_logits(...)` keeps working.

## 8. `ConditionalChatTTS.generate`: streaming audio-code sampling

All of this runs under `torch.inference_mode()`. Only batch size 1 is supported: assert
`input_ids.shape[0] == 1` and `past_key_values is not None`.

### 8.1 Sequence layout and inputs

The TTS sequence has a fixed *condition* prefix of length
`C = 1 + num_spk_embs*use_speaker_embedding + streaming_text_reserved_len + 1`. That is 1 text bos
slot, the speaker-embedding slots, the reserved text slots, then 1 audio-bos slot. Audio codes
follow from index `C` on.

- `input_ids`: int64 `(1, P, K)`. Rows `[0, C)` are placeholders, normally zeros. Rows `[C, P)` are
  audio codes generated so far, including by earlier calls. The first call has `P == C`.
- `past_key_values`: a Transformers `DynamicCache` holding exactly `P - 1` positions. Kept code
  builds it: zeros of length `C-1`, then `prefill_text` writes text keys and values in place, and
  each generate step appends one position. The caller is responsible for that state.
- `temperature`: float tensor `(K,)` (callers pass `[0.1, 0.3, 0.1, 0.3]`). Codebook `k` uses
  `temperature[k]`. A 1-element tensor also works (broadcast), and fixture s7 covers it.
- `eos_token`: Python int or 1-element int64 tensor (callers pass `tensor([625])`).
- `streaming_tts_text_mask`: length-`C` int tensor. Pass it unchanged to the kept mask helper.
- `min_new_token`, `max_new_token`, `force_no_stop`, `logits_warpers`, `logits_processors` and
  `show_tqdm`: described below.

### 8.2 Per-step algorithm (at most `max_new_token` steps, index `i = 0, 1, ...`)

Keep a working buffer that is the input codes followed by room for `max_new_token` new rows.
`cur` is the current length and starts at `P`. Keep a boolean "finished" flag, initially false.

Each step:
1. Check the invariant `cur == cache.get_seq_length() + 1`, and raise AssertionError on
   violation.
2. Build the step input embedding `(1, 1, H)`:
   - When `cur == C` (the step that has no audio code yet, the "audio-bos step"), use
     `emb_text(audio_bos_token_id)`.
   - Otherwise, sum over `k` of `emb_code[k](code[cur-1, k])`, embedding the last row.
3. Set position id `cache.get_seq_length()`, as a `(1, 1)` long tensor.
4. Build the attention mask with the kept helper `make_streaming_chunk_mask_generation`, passing
   the step embedding, `past_seen_tokens = cache.get_seq_length()`, the text mask,
   `streaming_reserved_length = streaming_text_reserved_len` and
   `streaming_text_chunk_size = streaming_text_chunk_size`. Leave its other arguments
   (`streaming_audio_chunk_size`, `num_spk_emb`, `use_spk_emb`) at their defaults (50, 1, True),
   not the config values. This is the current behavior. The helper returns an additive float mask
   `(1, 1, 1, cache_len+1)`.
5. Run the Llama decoder: `self.model(attention_mask=mask, position_ids=..., past_key_values=cache,
   inputs_embeds=..., use_cache=True)`. The cache is appended in place, and the cache object in the
   output is the same object.
6. Compute per-codebook logits from the last hidden state. For each `k` apply `head_code[k]`
   (weight-normed, no bias). Place the results into a **float32** tensor regardless of model dtype,
   then flatten to `(K, V)` in codebook order. Row `k` is codebook `k`, or in general row
   `b*K + k`.
7. Divide row `k` by `temperature[k]`.
8. On every step except the audio-bos step, apply each entry of `logits_processors` in list order
   and then each entry of `logits_warpers`, as `logits = f(history, logits)`. `history` is int64
   `(K, cur - C)`: row `k` holds codebook `k`'s codes at positions `C..cur-1`. That is every audio
   code so far, including earlier calls, but not this step's code.
9. If `i < min_new_token`, set column `eos_token` of every row to `-inf`. If `force_no_stop`, do the
   same. The two conditions are independent.
10. `probs = softmax(logits, -1)`. Sample **one** `torch.multinomial(probs, 1)` call on the
    `(K, V)` matrix, which gives one id per codebook. This is the only consumer of the global torch
    RNG in the loop, and golden token ids depend on it. Nothing else in `generate` may draw random
    numbers.
11. Reshape the ids to `(1, K)`. Set finished to true if **any** codebook drew `eos_token`. The
    flag is sticky.
12. Write the row into the buffer at position `cur`.
13. Quirk to preserve: if `i == 0` and finished, stop **without** advancing `cur`. The sampled row
    is discarded from the returned sequence, but the cache has already grown by one.
14. Otherwise `cur += 1`. Stop if finished.

### 8.3 Result

- `audio_input_ids` is the buffer truncated to `cur` rows, `(1, cur, K)`.
- `new_ids`:
  - If finished, it is `audio_input_ids[:, C:cur-1]`. The final row, normally the eos-bearing one,
    is dropped.
  - Otherwise it is `audio_input_ids[:, C:cur]`.

  Note that `new_ids` covers **all** audio codes from `C`, not only this call's. Consequences of
  the step-0 quirk, both pinned by fixtures:
  - first call (`P == C`): `new_ids` is empty `(1, 0, K)` (s4).
  - continuation call: the last code row of the *previous* call is dropped from `new_ids` (s5).

  In both cases the cache ends up holding `cur` positions, not `cur - 1`. A further `generate` call
  on that state would fail the invariant, but callers stop once finished.
- `past_key_values` is the same cache object, mutated.
- `finished` is a 0-dim bool tensor (`all` over the batch).
- If `show_tqdm` is set, show a progress bar of `max_new_token` steps (cosmetic). Also, only when
  `show_tqdm` is set, log at INFO that the result is incomplete when the token budget runs out.

Normal-case invariant: after a call that does not finish at step 0,
`cache_len == audio_input_ids.shape[1] - 1`.

## 9. `ConditionalChatTTS.decode_to_mel_specs`

Runs under inference mode. Input: a non-empty sequence of int code tensors, each `(L_i, K)`.
Callers pass `outputs.new_ids` directly, a `(1, L, K)` tensor, which iterates as one `(L, K)`
element.
- If the sequence is empty (`len == 0`), return `numpy.array([], dtype=float32)` (a numpy array,
  not a tensor).
- Otherwise, build a batch `(N, K, max L_i)` with the dtype and device of the first element.
  Element `i` is transposed to `(K, L_i)`, left-aligned, and right-padded with code **0**. Pass the
  batch to `self.dvae` in decode mode and return the resulting `(N, 100, 2*max L_i)` tensor.
  Padded frames decode as code 0, not silence, and are not trimmed.
- An element with `L_i == 0` (all max length 0) raises inside the quantizer dependency.

Decode requires `K == 4`. With the tiny test configs' `num_vq=2` the model can be constructed and
can run `generate`, but cannot decode.

## 10. Numerics and dtype notes

- Every conv, linear and LayerNorm above is the standard torch op. Plain `F.conv1d`, `F.linear`,
  `F.layer_norm` (eps 1e-6) and exact `F.gelu` in float32 reproduce the fixtures bit-exactly.
  This was verified independently from this spec, for every non-sampling fixture.
- The GFSQ code arithmetic is exact in float32 (multiples of 0.5 times 1 or 0.2, summed over two
  terms).
- Generation logits are float32. The sampling result depends on the exact `probs`. Any change of
  op order that perturbs logits by 1 ulp can, rarely, flip a sampled id. To be bit-compatible,
  compute head projections as the weight-normed linear (`g * v / ||v||`, then matmul) in float32,
  as torch's parametrization does.
- **Pre-existing defect to fix (not preserve).** Transformers 5.23 `from_pretrained` leaves the
  quantizer's *non-persistent* buffers uninitialized, because they are allocated empty and are not
  in the checkpoint. With the current port, `MiniCPMO.from_pretrained(...)` of a 2.6 model with TTS
  gives garbage `implicit_codebook`, `_levels`, `_basis`, `scales` and `soft_clamp_input_value`, so
  `decode_to_mel_specs` after loading produces wrong mels. A save/load round trip confirmed this:
  decoded mels differ. The Llama rotary buffers are re-initialized by Transformers and are fine.
  The clean-room version should compute these constants arithmetically (section 4.2), or rebuild
  them after loading in the model's `_init_weights` hook, so that a loaded model decodes exactly
  like a freshly built one. The fixtures record the freshly built behavior.

## 11. Behaviors the existing GPT-QModel tests rely on

From `tests/test_minicpm_o_local_port.py` and `tests/module_tree/test_minicpm_o_local_port.py`:

1. **Missing optional dependency.** With `sys.modules["vector_quantize_pytorch"] = None`, building a
   2.6 model with `init_tts=True` raises `ImportError` matching `gptqmodel\[minicpm-o\]`. Building
   one without TTS has no `tts` attribute. Tests gate 2.6-TTS cases on
   `find_spec("vector_quantize_pytorch")`.
2. **Checkpoint layout.** Building the real 2.6 config on the meta device yields exactly 1599
   state_dict keys with a pinned SHA-256 over sorted `key:shape` lines. All keys start with `llm.`,
   `vpm.`, `resampler.`, `apm.`, `audio_projection_layer.` or `tts.`. The DVAE keys of section 1.3
   are part of that hash, and no extra persistent buffers may be added.
3. **Save/load.** `save_pretrained` then `from_pretrained` round-trips every state_dict tensor
   exactly, for 2.6 with TTS too.
4. **No cache subscripting.** An AST scan of `tts_26.py` (and `utils.py`) fails if any subscript
   expression is applied to a name or attribute containing `past_key_values`. Use
   `cache.layers[i].keys` or `.values`, `get_seq_length()`, or the helpers in `cache.py`.
   Never write `past_key_values[...]`.
5. **Meta construction.** `MiniCPMO(config)` under `torch.device("meta")` must work with 2.6 TTS
   enabled. That includes building the DVAE and the GFSQ.
6. The TTS Llama (`tts.model`) is checked against stock Transformers `LlamaModel`. That is kept
   code and is unaffected.

## 12. Standard vs model-specific

- Standard: the depthwise/pointwise ConvNeXt-1D block, GELU, LayerNorm, Conv1d stacks, the
  Transformers TopP/TopK warpers, softmax plus multinomial sampling, and the vendored Llama decoder
  (`gptqmodel/models/ports/native/llama`) used through `.native.LlamaModel`.
- Model-specific: the fixed DVAE geometry, the even/odd time interleave of the two quantizer
  groups, the per-mel-bin `coef` scale, the row-indexed repetition-penalty cutoff, the
  multi-codebook flattening and per-codebook temperature, the audio-bos first step, sticky
  any-codebook eos, the step-0 eos quirk, and `new_ids` covering all audio codes from `C`.
