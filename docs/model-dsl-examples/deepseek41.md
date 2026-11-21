<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek V4.1 Flash

Selected checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash`, revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`. Its pinned
[config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json),
[reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
and [kernel bodies](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/kernel.py)
are the evidence. The existing [checkpoint inventory](../deepseek-v41-module-inventory.md)
is not GPT-QModel runtime support. See [sample conventions](README.md).

## Architecture is more than an attention/MLP layer

| Component | Checkpoint declaration |
| --- | --- |
| Backbone | 40 layers, H=5120, vocabulary 129280, maximum context 1048576. |
| Residual stream | Four hyper-connection copies: `bf16[B,T,4,5120]`; carried pre-mix `fp32[B,T,4]`. Sinkhorn uses 20 iterations and epsilon `1e-6`; sublayer norms use `1e-20`. |
| Attention | 64 heads × 512, 64 rotary dimensions, Q low-rank 1280; one shared latent K/V vector of width 512/token. Grouped output: 8 groups, rank 1024/group. Learned per-head attention sink. |
| Long context | Local window 128 plus selected compressed history. Compression ratios: layers 0–1: 0; 2–19: 2; 20–39: 1; draft layers 40–42: 0. |
| MoE | 384 routed, top 6, one shared, expert width 2304; sqrt(softplus(logits)) scoring, normalized unbiased weights × 1.5, SwiGLU clamp limit 10. |
| Additional paths | Engram at layers 1 and 14; 32-layer vision encoder, width 1024; three DSpark draft layers, reading main-layer attention inputs 37/38/39. |
| Native precision | FP8 weights with 32×32 blocks and UE8M0 scales, FP4 experts; activation/cache rounding and scale formats are separately declared. |

Ordinary Q/K/V projection fusion is not a valid description of this latent
attention. The body is Q-down → norm → Q-up and a separate latent-KV projection,
followed by two history sources and grouped low-rank output projection.

## Explicit request resources replace the reference's global slot

The source's `SharedAttentionRuntime` contains `compress_kv`, `index_k`,
`topk_idxs` and `candidates`, handed from producer layers to later layers.
The DSL makes these **request-owned resources with exact producer/reader maps**,
not process-global Python fields or local variables owned by a subset.

```text
request_resources DeepSeek41Request(B,C <= 1048576,Tcap) {
    for i in 0..42: window[i] = request_owned {
        kv: bf16[B,128,512], position: i64[B,128], valid: bool[B,128],
        write_cursor: i64[B], lifetime=request, init=empty
    }
    for s,r in [(2,2),(8,2),(14,2),(20,1)]: compressed[s] = request_owned {
        latent: bf16[B,floor(C/r),512], valid_length: i64[B],
        index_k: bf16[B,floor(C/r),128], index_length: i64[B],
        lifetime=request, init=empty
    }
    for s in [2,8,14]: compressor_tail[s] = request_owned {
        kv: fp32[B,2,512], score: fp32[B,2,512], count: i32[B],
        lifetime=request, init={kv:0,score:-infinity,count:0}
    }
    for s in [2,8,14,20,24,28,32,36]: selection[s] = request_owned {
        ids: i32[B,Tcap,512], count: i32[B,Tcap], step_id: i64,
        lifetime=until_last_reader_of_step, init=producer_write
    }
    candidates20: bool[B,Tcap,C] {
        owner=request, lifetime=until_layer39_done_for_step,
        init=layer20_indexer_done, identity=current_step
    }
    engram_history: {tokens:i64[B,3],valid:bool[B,3]} {
        owner=request,lifetime=request,init=empty
    }
    draft_inputs: bf16[B,Tcap,3*5120] {
        owner=request,lifetime=until_draft_or_verify_done,init=main_capture
    }
}
```

These BF16 cache values describe an explicit **rounded/dequantized reference
representation**, not proof that the source stores packed nibble arrays in
these buffers. Window KV is rounded to FP8 with groups of 32; compressed KV
is rounded to FP4 with groups of 16 and E4M3 scales; index Q/K uses FP4 groups
of 32 with E8M0 scales. A packed-cache lowering must separately declare its
payload dtype, scale tensors, physical shapes and conversion, preserving those
semantics. The much larger logical candidate mask may be compacted only with
a declared equivalent representation; it is not an unbounded hidden workspace.
Draft-window resources are required only when the draft entry is enabled.

| Layers, zero based | Compressed-KV resource | Index-K owner / selection resource | Candidate resource |
| --- | --- | --- | --- |
| 0–1 | None | None | None |
| 2–7 | `compressed[2]` | owner 2 / `selection[2]` | None |
| 8–13 | `compressed[8]` | owner 8 / `selection[8]` | None |
| 14–19 | `compressed[14]` | owner 14 / `selection[14]` | None |
| 20–23 | `compressed[20]` | owner 20 / `selection[20]` | `candidates20` produced at 20 |
| 24–27 | `compressed[20]` | owner 20 / `selection[24]` | Read `candidates20` |
| 28–31 | `compressed[20]` | owner 20 / `selection[28]` | Read `candidates20` |
| 32–35 | `compressed[20]` | owner 20 / `selection[32]` | Read `candidates20` |
| 36–39 | `compressed[20]` | owner 20 / `selection[36]` | Read `candidates20` |

At each source row the producer has write rights; later consumers have read
rights. Layers 24/28/32/36 compute new selections from their own queries and
weights against layer 20's index keys, restricted by its candidates. They
do not recompute or own layer 20's compressed KV. All layers separately update
their own window ring. Exact IDs and valid extents are bound per instance.

## Correlated attention and hyper-connection computation

The layer-24 specialization shows what it consumes without granting write
access to resources it only reads:

```text
block DS41_ATTN24(x: A[B,T,5120], pos: i64[B,T]; w: AttnWeights)
    -> (y: A[B,T,5120])
resources {
    window: read_write<Window512> at window.read..attention.done when=prefill|decode|verify
    compressed: read<CompressedHistory1> at compressed.read..attention.done when=prefill|decode|verify
    index_keys: read<IndexHistory1> at index.read..index.done when=prefill|decode|verify
    candidates: read<bool[B,T,C]> at candidate.read..index.done when=prefill|decode|verify
    selected: write<Selection512> at index.publish..attention.done when=prefill|decode|verify
} {
    qr = RMS_NORM(linear(x;w.wq_a[1280,5120]);w.q_norm,eps=1e-20)
    q = reshape(linear(qr;w.wq_b[32768,1280]),[B,T,64,512])
    q = ROPE_TAIL(q,pos,tail=64,compressed_yarn_profile)
    local = RMS_NORM(linear(x;w.wkv[512,5120]);w.kv_norm,eps=1e-20)
    local = FP8_ROUND(ROPE_TAIL(local,pos,tail=64,compressed_yarn_profile),group=32)
    visible_window = UPDATE_WINDOW_RING(local,pos;window_size=128) resources {ring=window}
    selection = INDEX_SCORE_SELECT(x,qr,resource_read(index_keys),resource_read(candidates);w.indexer)
    resource_write(selected,selection)
    kv = concat(visible_window,resource_read(compressed).latent,axis=token)
    keep = concat(window_indices(pos),ids_with_window_offset(selection.ids))
    a = SPARSE_SINK_ATTENTION(q,kv,keep;w.attn_sink,scale=1/sqrt(512))
    grouped = reshape(a,[B,T,8,4096])
    low = grouped_linear(grouped;w.wo_a[8,1024,4096])
    return linear(flatten_groups(low);w.wo_b[5120,8192]) at attention.done
}
bind decoder.24.attn.resources = {
    window:request.window[24], compressed:request.compressed[20].{latent,valid_length},
    index_keys:request.compressed[20].{index_k,index_length},
    candidates:request.candidates20, selected:request.selection[24]
}
```

`Window512` and `Selection512` are their complete typed manifest records.
`CompressedHistory1` is `{latent:bf16[B,C,512],valid_length:i64[B]}`;
`IndexHistory1` is `{index_k:bf16[B,C,128],index_length:i64[B]}`. Braced
bindings are explicit record views, not a wildcard import. The selection
helper returns IDs, counts and the current step ID as a `Selection512` record.
The window helper explicitly forwards its resource interface;
all value-only projection/norm/RoPE/score/attention helpers declare
`resources {}`. A source-layer attention specialization additionally declares
write access to its compressed/index history, its partial-group tail when
ratio=2 and its candidate output at layer 20. It must not borrow layer 24's
read-only interface and mutate through it.

For ratio=2, the compressor pools pairs with a learned per-feature softmax
gate, retaining incomplete pairs in FP32 tail state. Ratio=1 is projection
plus norm, without the pooling gate. Index keys are derived from the **pre-RoPE
latent before that storage is mutated** for compressed KV. Index scores are
`sum_heads(relu(dot(q_h,k))*weight_h)`; causal completed-group limits precede
top-k 512. Layer 20 first selects up to 2048 candidate blocks of size 8 by
block maximum, pins the newest partial block, and publishes a bool mask.
Later sources filter their own scores with this mask. Selected positions
are sorted into position order, unreachable positions become `-1`, and
window offsets are explicit. Sink attention adds learned per-head sink mass
to the softmax denominator; the sink contributes no value vector.

The layer wrapper forwards the exact attention resource interface; its
hyper-connection helpers each have `resources {}` and operate on stream and
pre-mix values. A concrete layer body is:

```text
block DS41_LAYER24(x:A[B,T,4,5120],incoming:fp32[B,T,4],pos:i64[B,T],image:bool[B,T];w:LayerWeights)
    -> (y:A[B,T,4,5120],next_pre:fp32[B,T,4])
resources {
    window: read_write<Window512> when=prefill|decode|verify at attn.window.read..attn.done
    compressed: read<CompressedHistory1> when=prefill|decode|verify at attn.compressed.read..attn.done
    index_keys: read<IndexHistory1> when=prefill|decode|verify at attn.index.read..attn.index.done
    candidates: read<bool[B,T,C]> when=prefill|decode|verify at attn.candidate.read..attn.index.done
    selected: write<Selection512> when=prefill|decode|verify at attn.index.publish..attn.done
} {
    attn_mix = HC_MIXES(x;w.hc_attn)
    n = RMS_NORM(HC_PRE(x,incoming);w.attn_norm,eps=1e-20)
    a = DS41_ATTN24(n,pos;w.attn) resources {
        window=window,compressed=compressed,index_keys=index_keys,
        candidates=candidates,selected=selected
    }
    r = HC_POST(a,x,attn_mix.post,attn_mix.comb)
    ffn_mix = HC_MIXES(r;w.hc_ffn)
    n = RMS_NORM(HC_PRE(r,attn_mix.pre);w.ffn_norm,eps=1e-20)
    m = DS41_MOE(n,image;w.ffn)
    y = HC_POST(m,r,ffn_mix.post,ffn_mix.comb)
    return y,ffn_mix.pre
}
```

The enclosing computation order is essential: compute `attn_mix` from `x`;
collapse `x` with **incoming pre-mix**; norm and attention; HC_POST; compute
`ffn_mix` from that new stream; collapse it with **attn_mix.pre**; norm and
MoE; HC_POST; pass **ffn_mix.pre to the next layer**. A sublayer does not use
its own newly computed pre-mix. `HC_MIXES` flattens 4×5120 channels, uses FP32
linear/RMS scaling and splits 24 coefficients into pre/post/comb. Pre is
sigmoid plus epsilon, post twice sigmoid; comb starts with row softmax plus
epsilon, then a column normalization and 19 row/column normalization pairs.
`HC_POST` multiplies post into the sublayer output and adds the declared
combination of residual copies, with explicit axis orientation and BF16 cast.
All these bodies require expansion, not an opaque call to the reference.

MoE routing computes FP32 sqrt-softplus scores, chooses with text/image-specific
correction bias, gathers **unbiased** scores, normalizes with epsilon `1e-20`
and scales by 1.5. Experts compute gate/up, clamp gate above 10 and up to
[-10,10], multiply SiLU(gate) × up in FP32 and apply route weight **before**
the BF16 cast/down projection. Moving that weight after down changes rounding.
The shared expert is separate. These value-only regions declare `resources {}`.

## Other request-wide dependencies and admission gaps

| Who | External dependency | When |
| --- | --- | --- |
| Engram hash producer | Token history `i64[B,3]`, validity `bool[B,3]`; immutable model-scope vocabulary/hash tables | Once per chunk; image spans do not contribute text n-grams. Layers 1/14 consume explicit typed hash/value outputs. |
| Vision encoder/aligner → embedding merge | Patch/grid tensors and aligned `bf16[B,Nimage,5120]` features | Multimodal prefill; this reference requires image spans in the first unsplit chunk. |
| Main attention-input capture | `draft_inputs: bf16[B,T,15360]` | Before attention in layers 37/38/39, not their outputs. |
| DSpark draft/verify entry | Draft inputs, its own window caches and token/state branch | Separate entry point, three layers 40–42; draft config has 128 routed/top-3 experts, not the main 384/top-6 policy. |

The full Engram hash/gate body, vision/aligner, draft heads, exact checkpoint
conversions and every portable/kernel lowering still need implementation and
independent qualification. The reference itself does not implement a complete
speculative acceptance loop. These boundaries must reject compiled execution
until supplied; the existence of an inventory or this sample cannot enable it.
