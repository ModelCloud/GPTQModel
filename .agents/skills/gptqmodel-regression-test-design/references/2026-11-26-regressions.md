<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Regression fixtures from the Nov 24–26 audit

29 PRs merged into main in the 48 hours ending Nov 26 at 01:01 UTC.
The [Nov 25 audit](2026-11-25-regressions.md) covers the overlapping kernel
discovery and packed-field fixes. The cases below were fixed during review
of four feature PRs. [Audited revision](https://github.com/ModelCloud/GPTQModel/commit/51f697c0333f77af84b96893d2f45e1d7afb973e)

## GGUF export must describe a graph the target can execute

Use the one-block fixture in the [packaging suite](../../../../tests/test_gguf_package.py).
Derive the expected result from the target architecture. Exercise
`package(GGUFPackageConfig(...))` for each case.

| Change to the checkpoint | Expected result |
| --- | --- |
| Hidden size 256, eight query heads of width 64 | Qwen2 rejects query width 512. Qwen3 and Llama accept it. Load each accepted file with `llama_cpp.Llama`, evaluate a short input and inspect its logits. |
| Qwen3 output projection bias | Independently set `attention_bias=True` and supply a stray `o_proj.bias` with the flag false. Both exports must refuse. A bias-free checkpoint loads. |
| Llama with `mlp_bias=True`, intermediate size 512, hidden size 256 | Remove the down bias, then supply 512 entries. Require an error in each case. The complete checkpoint with a 256-entry down bias loads. |
| Omit context length and RMS epsilon from the source config | Assert the family defaults in the emitted metadata and load the file. Also try an explicit unsupported `hidden_act="gelu"`. Export must refuse because the target computes SiLU. |
| Add a derived buffer, then add a similarly named weight | Accept `model.rotary_emb.inv_freq`. Reject the unknown `model.rotary_emb.inv_freq_scale.weight` and name it in the error. |

On refusal, check that a new destination is absent or an existing sentinel is
intact when overwrite was requested. File read-back checks the plan's names,
shapes and types. A native forward checks loadability and execution. For
changes to the computation, compare logits on identical token IDs with the
source model as in the [cross-engine suite](../../../../tests/test_gguf_package_e2e.py).
[Defaults, activation and skipped tensors](https://github.com/ModelCloud/GPTQModel/commit/50f9c5f49a49b24ae42eb0fe5f876ef734bed42a) ·
[width and biases](https://github.com/ModelCloud/GPTQModel/commit/a10507127ba4c60f2c8732488619a0e619eaf616) ·
[proposed target-layout contract](https://github.com/ModelCloud/GPTQModel/issues/6067)

## A failed prefill must preserve the prefix that existed on entry

In the [LeapQuant state suite](../../../../tests/test_leapquant_state.py), seed one
live token and use a two-token prompt. Give key, value, alpha or beta one token
and then three, keeping query length two. The longer case detects silent
truncation. The shorter case detects consumption before a length error.

Equal lengths still permit a late failure. With window 2 and an FP16 buffer,
start from state 2 with one occupied slot. The first valid token flushes the
window. Give the second token a finite value of `2**20` that overflows the
buffer. After rejection compare every field of `packed_state()` to the pre-call
snapshot. Retry both tokens with unit values and beta 0.5. The expected outputs
are 1.5 and 1.25, and the final materialized state is 1.25. This detects a
prefix processed twice even when the retry succeeds.
[Length fix](https://github.com/ModelCloud/GPTQModel/commit/8ebcd4879dde4bfb2e81efc6736bae7f4737bd56) ·
[whole-call commit fix](https://github.com/ModelCloud/GPTQModel/commit/0740770d7f28ffbd47262a1a346763083db05bd6)

## A grid refresh changes what the next decision reads

- **G2PTQ.** A dynamic group fit reads all its columns. Use 12 input columns,
  group size 4, nonzero gradient and correlated updates. Compare block sizes
  3, 6 and 8 with block size 1, which applies each update before the next
  column. Assert the group indices and zero points, scale agreement and exact
  codes. Before the fix, block sizes 3 and 6 disagreed with the columnwise
  scales, while the aligned block size 8 passed.
  [Fix](https://github.com/ModelCloud/GPTQModel/commit/d1e42b3401e6a37dc5a0f8ae736d28697fe1e08d) ·
  [columnwise control](../../../../tests/test_g2ptq_processor.py)
- **ADMM-Q.** Make iteration two both eligible for early stop and able to
  accept a better grid. Confirm a refresh-disabled control stops there and
  a two-iteration control accepts the refresh. With refresh and continuation
  enabled, require a later iteration and lower independently computed
  Hessian-weighted reconstruction error than the capped run. Checking only
  the refresh flag misses convergence measured on the old iterate.
  [Fix](https://github.com/ModelCloud/GPTQModel/commit/e36c8a309289d92c97e6804cb15e47b2f32c415d) ·
  [early-stop controls](../../../../tests/test_admmq.py)
