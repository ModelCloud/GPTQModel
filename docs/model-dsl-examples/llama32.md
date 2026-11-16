<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama 3.2 1B Instruct

Design sample for `meta-llama/Llama-3.2-1B-Instruct`. Read this with the
[sample conventions and evidence boundaries](README.md). The official
[card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/9213176726f574b556790deb65791e0c5aa438b6/README.md)
is public; its config is gated. The numerical config below is corroborated by
the [public Unsloth mirror](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/blob/5a8abab4a5d6f164389b1079fb721cfab8d7126c/config.json),
not a claim that the gated checkpoint was loaded. The
[reference attention/decoder](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/llama/modeling_llama.py#L200-L318)
provides the computation order.

## Resolved structure

| Item | Declaration |
| --- | --- |
| Decoder | 16 sequential layers, `H=2048`, `I=8192`, vocabulary `128256`. |
| Attention | `Nq=32`, `Nkv=8`, `D=64`; Q width `2048`, K/V width `512` each, logical packed QKV width `3072`. No projection biases or Q/K norm. |
| Norm/MLP | Two RMSNorms per layer, epsilon `1e-5`, stored gain `w`; SiLU(gate) × up, then down. |
| Position | Full-head Llama-3 RoPE, theta `500000`, factor `32`, original context `8192`, low/high frequency factors `1/4`; maximum positions `131072`. |
| Output | Final RMSNorm and linear head; embedding/head have one parameter identity because `tie_word_embeddings=true`. |

```text
request_resources Llama32Request(B, C <= 131072) {
    for i in 0..15: kv[i] = owned_request {
        k: bf16[B,8,C,64], v: bf16[B,8,C,64],
        positions: i64[B,C], valid: bool[B,C], lengths: i64[B],
        init=empty, lifetime=request, memory=device_global
    }
}

block LLAMA32_LAYER(x: A[B,T,2048], pos: i64[B,T], valid: bool[B,T]; w: LayerWeights)
    -> (y: A[B,T,2048])
resources {
    cache: read_append<Llama32KV> when=prefill|decode|verify
        at cache.read..layer.done
} {
    n = RMS_NORM(x; w.input_norm, eps=1e-5, gain=weight)
    past = resource_read(cache) at cache.read
    ctx = CAUSAL_CONTEXT(pos,valid,past.positions,past.valid)
    a,next = ATTENTION(n,ctx,past; w.attn,
        H=2048,Nq=32,Nkv=8,D=64,rope=llama32_rope,dropout=0)
    resource_write(cache,next) at cache.publish
    r = add(x,a)
    z = RMS_NORM(r; w.post_attention_norm, eps=1e-5, gain=weight)
    gate,up = GATE_UP(z; w.gate[8192,2048],w.up[8192,2048])
    m = linear(silu(gate)*up; w.down[2048,8192])
    return add(r,m) at layer.done
}

instantiate decoder[i in 0..15] = LLAMA32_LAYER {
    resources.cache = request.kv[i]
    parameters = explicit_checkpoint_record("model.layers", i)
}
```

`Llama32KV` is precisely the record inside `kv[i]`; the loop is compile-time
expansion into 16 independent records. `explicit_checkpoint_record` abbreviates
the exact binding table below, not a function that guesses weights or code.
`ATTENTION`, `GATE_UP`, `RMS_NORM` and mask operations use the executable pure
bodies in the [main proposal](../model-computation-dsl.md), each with
`resources {}`. The layer alone owns the cache capability. Quantization subsets
inside its MLP therefore do not need a cache; attention replay does.

| Parameter role | Exact checkpoint key for layer `i` |
| --- | --- |
| Input/post-attention norm | `model.layers.{i}.input_layernorm.weight`, `model.layers.{i}.post_attention_layernorm.weight` |
| Q/K/V/O | `model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`, with four explicit emitted bindings |
| Gate/up/down | `model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight`, with three explicit emitted bindings |
| Model boundaries | `model.embed_tokens.weight`, `model.norm.weight`, tied `lm_head.weight` alias/conversion contract |

The request entry point embeds `i64[B,T]` tokens into `bf16[B,T,2048]`, calls
the 16 instances in order, applies final RMSNorm and the head, and returns
`bf16[B,T,128256]` logits for this native activation specialization. Last-token
selection or FP32 output casting is a separate explicit entry-point operation.
For the simplest example all tokens are valid; padding/all-masked-query
behavior must be specified and qualified before admitting padded batches.

Llama-3 RoPE is not interchangeable with default RoPE. For each frequency
`f=theta^(-2*j/D)`, wavelength `lambda=2*pi/f`: keep `f` when `lambda<2048`,
use `f/32` when `lambda>8192`, otherwise interpolate with
`smooth=(8192/lambda-1)/(4-1)` and
`f'=(1-smooth)*f/32+smooth*f`. Frequency generation, trigonometry, application
casts and cache K layout are declared operations. The current chunk's K is
rotated once before append; replay must not rotate cached K again.

## Who needs which request resource and when

| Block | Resource / dtype and shape | Access and interval |
| --- | --- | --- |
| `decoder.i` → its attention cache port | `request.kv[i]`, K/V `bf16[B,8,C,64]` plus metadata | Read old prefix and append current positions during that layer's attention; retain for every future token step. |
| QKV/norm/MLP subsections | None | Explicit empty external-resource interface; values and weights arrive through typed ports. |
| Final norm/head | None | Empty interface; output is a value unless an explicit external output-buffer adapter is selected. |

QKV may share an observation because all three projections consume the exact
same normalized value. The output projection and MLP down projection observe
different values. AWQ must also preserve residual uses and parameter aliases;
quantizing a tied head independently requires an explicit alias-breaking
transformation/policy, never silently replacing one of two names.

The [initial executable compiler](../model-dsl-compiler.md) separately expands
this architecture as a typed Python program and validates pure-Torch inference
against an independent oracle using the pinned public mirror's weights.
The abbreviated textual sample above remains design notation. Gated official
checkpoint reconciliation, padded batches and the broader resource ABI remain
outside that implementation.
