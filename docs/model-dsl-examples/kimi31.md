<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kimi K3.1 Pro

Selected latest open Kimi: `moonshotai/Kimi-K3.1-Pro`, pinned
[config](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json)
and [card](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/README.md).
The [published model list](https://platform.kimi.ai/docs/models) does not name
a Flash checkpoint. Pro/highspeed are not silently treated as different
architectures. See [sample conventions](README.md).

The pinned independent [NeMo AutoModel implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/c57935fc64faea913ce9f5e38ea7e57e6416cc93/nemo_automodel/components/models/kimi_k3/model.py)
is a readable computation reference, not proof of checkpoint-native quantized
inference or cached decode. In particular, its training KDA path passes
`initial_state=None, output_final_state=False`; the request-state interface
below is a proposed executable inference contract requiring separate parity
qualification, not a claim that this reference already implements it.

## Per-layer and subsection specialization

| Item | Declaration |
| --- | --- |
| Backbone | 93 layers, H=7168, vocabulary 163840, context 1048576. |
| Mixer choices | Config layer numbers are **one based**: DSA at `4,8,...,92,93`; all remaining 69 are KDA. Emit zero-based DSA instances `3,7,...,91,92` (24 total), never apply a simple every-fourth rule to the last layer. |
| KDA | 96 heads × 128, separate Q/K/V short convolutions of width 4; vector-valued decay, lower bound -5, full-rank output gate. |
| Gated DSA | Q rank 1536, latent KV rank 512; 96 attention heads, Q/K content width 192 = 128+64, V width 128. `mla_use_nope=true`: no RoPE rotation, but the extra 64 content dimensions remain. |
| DSA indexer | 32 heads × 128, top 2048 individual causal keys, biased LayerNorm on index K, no index RoPE, selection every layer/step. |
| AttnRes | Residual block size 12; learned depth attention over completed block starts plus current prefix. This crosses ordinary decoder-layer boundaries. |
| FFN | Layer 0 dense width 33792; remaining 92 use 896 routed/top-16 experts, expert width 3072 at latent width 3584; two shared experts (combined shared width 6144). SiTU-GLU. |
| Native quantization | Card: MXFP4 weights/MXFP8 activations, QAT. Config: `mxfp4-pack-quantized`, with explicitly ignored paths and native tensors that must retain their own dtype/format. |

## Request resources and a separate depth-residual lifetime

```text
request_resources Kimi31Request(B,C <= 1048576,Tcap) {
    for i in kda_ids: kda[i] = request_owned {
        qconv: bf16[B,12288,4], kconv: bf16[B,12288,4], vconv: bf16[B,12288,4],
        matrix: fp32[B,96,128,128], processed: i64[B],
        init=zeros,lifetime=request
    }
    for i in dsa_ids: dsa[i] = request_owned {
        latent: bf16[B,C,512], extra_k: bf16[B,C,64],
        index_k: bf16[B,C,128], positions: i64[B,C],
        valid: bool[B,C], lengths: i64[B],init=empty,lifetime=request
    }
    attnres = request_owned {
        bank: bf16[B,Tcap,8,7168], counts: i32[B,Tcap],
        token_positions: i64[B,Tcap], step_id: i64,
        init=empty_at_new_token_interval,lifetime=through_final_depth_mix
    }
    vision_features: bf16[B,Nimage,7168] {
        owner=request,init=vision_projector_done,lifetime=until_embedding_merge_done
    }
}
```

The AttnRes bank holds **depth history for the current token interval**, not
an ever-growing token history. Reset its valid extent for each new interval;
preserve it across layers and pipeline stages until final depth mixing. The
eight bank slots correspond to starts `0,12,...,84` in the referenced
zero-based implementation. This memory can be request-owned and reused while
its contents have a shorter semantic lifetime than the request's KDA/DSA
state. Persistent cache precision/physical layout and short-convolution state
conversion must be fully qualified against the chosen inference reference.

```text
block KIMI_KDA(x:A[B,T,7168],valid:bool[B,T];w:KDAWeights)
    -> (y:A[B,T,7168])
resources {
    qconv: read_write<bf16[B,12288,4]> when=prefill|decode|verify at conv.read..conv.done
    kconv: read_write<bf16[B,12288,4]> when=prefill|decode|verify at conv.read..conv.done
    vconv: read_write<bf16[B,12288,4]> when=prefill|decode|verify at conv.read..conv.done
    matrix: read_write<fp32[B,96,128,128]> when=prefill|decode|verify at scan.read..scan.done
    processed: read_write<i64[B]> when=prefill|decode|verify at entry..done
} {
    q,k,v = ATTN_QKV(x;w.q[12288,7168],w.k[12288,7168],w.v[12288,7168])
    q = short_conv_silu(q;w.qconv) resources {history=qconv}
    k = short_conv_silu(k;w.kconv) resources {history=kconv}
    v = short_conv_silu(v;w.vconv) resources {history=vconv}
    raw = reshape(linear(linear(x;w.f_a[128,7168]);w.f_b[12288,128]),[B,T,96,128])
    decay: fp32[B,T,96,128] = -5*sigmoid(exp(w.A_log_fp32)*(cast(raw,fp32)+w.dt_bias_fp32))
    beta: fp32[B,T,96] = sigmoid(cast(linear(x;w.beta[96,7168]),fp32))
    o,next = KDA_SCAN(q,k,v,decay,beta,resource_read(matrix),valid)
    resource_write(matrix,next)
    gate = reshape(linear(x;w.gate[12288,7168]),[B,T,96,128])
    o = RMS_NORM_GATED(o,gate;w.o_norm,gate=sigmoid,eps=1e-5)
    resource_write(processed,advance_valid_counts(resource_read(processed),valid))
    return linear(flatten_heads(o);w.o[7168,12288]) at done
}
```

The short-convolution wrapper declares only its own typed history resource;
its numerical core declares `resources {}`. `KDA_SCAN` has Q/K/V shaped
`A[B,T,96,128]`, FP32 decay/beta/state, validity, and O/state outputs; its
external resource set is empty. Its normalized Q/K delta-rule recurrence has
the same outer-product structure as the [Qwen sample](qwen38.md), but decay
is **per key feature**, so `S_decay=exp(decay_t)[...,key,None]*S` rather than
one scalar/head. The bounded gate, Q/K normalization epsilon convention,
state-axis orientation, all casts and output sigmoid gate are separate
versioned semantics, not inherited from a generic linear-attention label.
Packed documents require explicit reset boundaries in convolution and scan.

## DSA is sparse latent attention, not KDA or ordinary QKV

A DSA block has input/output `A[B,T,7168]` and declares read/append ports for
its layer's latent KV, extra K, index K and typed position metadata, required
in prefill/decode/verify from key projection through attention completion.
Its nested indexer has only the index-K capability. The value-only cores each
declare `resources {}` and expand in this order:

1. Q-down `7168 -> 1536`, RMSNorm, Q-up to `96*192`. Preserve the normalized
   rank-1536 query as the indexer's input.
2. KV-down `7168 -> 512+64`; normalize the 512 latent channels and append
   latent plus extra-64 K with positions. For the reference expansion, KV-up
   maps 512 to `96*(128+128)`: K-content 128 and V 128; append the unchanged
   extra-64 K to make each head's K width 192. No RoPE is applied.
3. Index Q `1536 -> 32*128`; index K `7168 -> 128`, biased LayerNorm; head
   weights `7168 -> 32`, scaled by `1/sqrt(32*128)`. Scores are FP32
   `sum_h(relu(dot(q_h,k))*weight_h)`, then causal top-2048. The independent
   reference computes BF16 projections with FP32 scoring; native activation
   quantization requires its own declared contract.
4. Attend only selected valid positions, with score scale `1/sqrt(192)`;
   multiply flattened V-context by sigmoid of the full-rank output-gate
   projection, then project `12288 -> 7168`.

Selection becomes effectively dense when the valid history is at most 2048;
a test containing only short sequences cannot qualify sparse selection.
The latent-cache lowering and explicit dense expansion must agree on all
output values and cache versions; neither may retain an expanded weight cache.

## AttnRes changes the layer interface and replay boundary

```text
block KIMI_DEPTH_MIX(prefix:A[B,T,7168], bank:A[B,T,M,7168];w:DepthMixWeights)
    -> (mixed:A[B,T,7168]) resources {} pure {
    candidates = concat(bank,prefix[...,None,:],axis=depth)
    score = linear(RMS_NORM(candidates;w.norm,eps=1e-5);w.proj[1,7168])
    p = softmax(score,axis=depth,compute=fp32)
    return cast(sum_depth(p*cast(candidates,fp32)),A)
}
```

The layer wrapper declares a read/write `attnres` port, mapped to
`request.attnres` from layer entry through its final update, plus **only** the
selected mixer's resources. Its ordered body reads the bank, depth-mixes the
incoming prefix for attention when the bank is nonempty, and at layers
`i % 12 == 0` appends the incoming prefix as a completed block start and resets
the current prefix. Norm → selected mixer produces the attention contribution;
add it to the current prefix (or use it as the new prefix at a boundary).
Depth-mix that prefix with the bank using the FFN's distinct norm/projection,
apply post-attention norm and the selected dense/MoE body, then return
prefix+FFN and publish the bank version. Final output performs its own depth
mix before final norm/head. A generic `x + attention(x); x + mlp(x)` changes
this computation and cannot be selected as a fallback.

The latent MoE routes on unpooled H=7168 using an FP32 sigmoid router,
correction-bias selection, top 16, normalized unbiased weights with epsilon
`1e-20`. Project expert inputs `7168 -> 3584`, execute width-3072 SiTU experts,
combine in the declared expert/cast order, latent norm, then project back to
7168. The shared path consumes the original 7168-wide input, with combined
intermediate width 6144, and is added separately. Dense/shared/expert/route
cores have empty request-resource interfaces. SiTU computes in FP32:
`activation=4*tanh(gate/4)*sigmoid(gate)`,
`up'=25*tanh(up/25)`, then casts `activation*up'` to the declared output dtype
before down projection. It is not ordinary SwiGLU.

| Who | What | When |
| --- | --- | --- |
| KDA layer `i` | Only `kda[i]` convolution histories/matrix/progress | Current convolution/scan; retain across token chunks. |
| DSA layer `i` / its indexer | Only `dsa[i]`; indexer sees its index-K view | Current append/selection/attention; retain histories across chunks. |
| Every depth wrapper and final depth mix | `attnres.bank/counts/positions`, with exact read/write views | Current interval across layers/stages; reset for the next interval, after completion. |
| MoonViT-V2/projector → embedding merge | `vision_features:bf16[B,Nimage,7168]` and typed image layout | Multimodal prefill only; the card reports a 401M-parameter vision encoder. |
| Calibration route/recovery stages | Explicit route IDs/counts, bounded replay spool and active expert's statistics | Calibration-only entry points; snapshot associated cache/depth state before replay. No access granted to ordinary inference experts. |

Native MXFP4 packing, MXFP8 activation rounding, ignored native tensors,
complete vision/body mappings and cached inference precision need independent
qualification. GPT-QModel's existing route/recovery calibration keeps natural
routes, recovery observations and identity regularization distinct; these are
method-policy resources, not permission to change inference routes. Replaying
one layer without its bank and mixer-state versions is insufficient, and
failed candidates/verify branches must roll back all of them together.
