<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Muse Spark Open

Latest selected Muse: `meta-models/Muse-Spark-Open-816B-A42B`, revision
`4c0e167c8b7a18e1dd5ca76763ecb5d53c6a9601`. The official public
[card](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B/blob/4c0e167c8b7a18e1dd5ca76763ecb5d53c6a9601/README.md)
describes the October 2026 release. This is newer than Muse Glimmer; the
existing Glimmer definition must not select Spark's computation or bindings.
See [sample conventions and gated-config boundary](README.md).

## Public architecture and reference specialization

The card confirms H=8192, 62 text layers, vocabulary 202048, 128 Q heads,
16 KV heads, D=64, 256 experts/top 8 and a 50-layer perception encoder.
The following schedule/numerical constants are a **concrete specialization of
the pinned public reference configuration**. They must be reconciled with the
gated checkpoint config before treating this as an exact checkpoint manifest.

| Subsection | Explicit specialization |
| --- | --- |
| Attention schedule | From the [config](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/muse_spark/configuration_muse_spark.py#L75-L187): full NoPE at zero-based `1,5,9,...,61`; remaining 46 layers local with window 2048 and RoPE theta 500000. The pattern is counted backward from the last layer. |
| Gated GQA | Q/K scaleless per-head RMSNorm, epsilon `1e-5`; multiply Q by `5.4730064863875` in addition to score scale `1/sqrt(64)`. Per-head output norm, sigmoid attention gate, then output projection. |
| MoE | Router reads full H=8192; experts operate at latent width 4096 after a learned pool projection and centered norm; expert intermediate width 4096; unpool after combine. Reference config defaults all 62 FFNs to sparse. |
| Residual paths | Centered pre-norms; scaleless post-attention norm; input-scaled post-FFN norm; separate learned residual gate after each subsection. Post norms use `1e-8`. |
| Boundary | Scaleless normalization immediately after token embedding; final gain RMSNorm; untied head; logits `20*tanh(raw*0.1767766952966369/20)`. |
| Perception | 50 layers, hidden 1536, 16 heads, FFN 8960, patch 14; local/sparse vs full vision attention is separately instantiated. Learned absolute position + 2D RoPE; merged vision output → adapter → text projection. |
| Audio | Reference config has a four-stage causal convolution pyramid: patches `[16,6,5,4]`, widths `[256,512,1024,2048]`, two conv blocks/stage, kernel 7, output width 8192. |

## Request state and per-block access

```text
request_resources MuseSparkRequest(B,C <= 1048576,Tcap) {
    for i in 0..61: kv[i] = request_owned {
        k: bf16[B,16,Ci,64], v: bf16[B,16,Ci,64],
        positions: i64[B,Ci], valid: bool[B,Ci], lengths: i64[B],
        Ci=(2048 if local[i] else C), init=empty,lifetime=request
    }
    vision_features: bf16[B,Nvision,8192] {
        owner=request,init=vision.norm.done,lifetime=until_vision_scatter_done
    }
    audio_features: bf16[B,Naudio,8192] {
        owner=request,init=audio.norm.done,lifetime=until_audio_scatter_done
    }
    for c in resolved_audio_convolutions: audio_left[c] = request_owned {
        history: bf16[B,Cin[c],Lpad[c]], initialized: bool[B],
        init=zeros,lifetime=audio_stream
    }
}
```

`Ci=2048` requires a checked ring/window resource with logical-position
metadata and overwrite rights. Full NoPE resources are read/append. The two
interfaces are distinct specializations, not one cache operation guessed from
allocated width. `Cin/Lpad` are resolved from each actual audio convolution's
channels and causal padding rule; the emitted manifest has concrete records
even for zero-length padding. Audio history belongs to its request/stream,
not the immutable model or every text decoder block.

```text
block MUSE_ATTN(x: A[B,T,8192],pos:i64[B,T],valid:bool[B,T];w:AttnWeights)
    -> (y:A[B,T,8192])
resources {
    cache: CacheCapability<MuseKV,selected_attention_variant>
        when=prefill|decode|verify at cache.read..attention.done
} {
    q,k,v = ATTN_QKV(x;w.q[8192,8192],w.k[1024,8192],w.v[1024,8192])
    q = SCALELESS_RMS_PER_HEAD(q,D=64,eps=1e-5)*5.4730064863875
    k = SCALELESS_RMS_PER_HEAD(k,D=64,eps=1e-5)
    q,k = selected_rope_or_identity(q,k,pos)
    history = declared_cache_update(resource_read(cache),k,v,pos,valid)
    resource_write(cache,history)
    mask = causal_valid_mask(pos,history)
    mask = mask & selected_window_constraint(pos,history.positions)
    a = GQA(q,history.k,history.v,mask,score_scale=1/sqrt(64))
    a = flatten_heads(SCALELESS_RMS_PER_HEAD(a,D=64,eps=1e-5))
    gated = a*sigmoid(linear(x;w.gate[8192,8192]))
    return linear(gated;w.o[8192,8192]) at attention.done
}

block MUSE_MOE(x:A[N,8192];w:MoEWeights) -> (y:A[N,8192]) resources {} pure {
    logits: fp32[N,256] = linear(cast(x,fp32);cast(w.router,fp32))*0.1767766952966369
    score = sigmoid(logits)
    ids = topk_indices(score+w.selection_bias,8,sorted=false)
    p = gather(score,ids)
    weights = p/(sum(p,expert_slot)+1e-15)
    pooled = CENTERED_RMS(linear(x;w.pool[4096,8192]);w.pool_norm,eps=1e-5)
    each = dispatch(pooled,ids)
    gate,up = GATE_UP(each;w.experts.gate,w.experts.up,I=4096)
    each = linear(silu(gate)*up;w.experts.down)
    each = INPUT_SCALED_RMS(each;w.post_expert_norm,eps=1e-8)
    mixed = weighted_combine(each,weights,reference_expert_order)
    return linear(mixed;w.unpool[8192,4096])
}

block MUSE_LAYER(x:A[B,T,8192],pos:i64[B,T],valid:bool[B,T];w:LayerWeights)
    -> (y:A[B,T,8192])
resources {
    cache: CacheCapability<MuseKV,selected_attention_variant>
        when=prefill|decode|verify at attention.cache.read..attention.done
} {
    n = CENTERED_RMS(x;w.input_norm,eps=1e-5)
    a = MUSE_ATTN(n,pos,valid;w.attn) resources {cache=cache}
    a = SCALELESS_RMS(a,eps=1e-8)
    r = RESIDUAL_GATE(x,a;w.attn_residual_gate,temperature=0.3)
    m = MUSE_MOE(CENTERED_RMS(r;w.pre_ffn_norm,eps=1e-5);w.moe)
    m = INPUT_SCALED_RMS(m;w.post_ffn_norm,eps=1e-8)
    return RESIDUAL_GATE(r,m;w.ffn_residual_gate,temperature=0.3)
}
bind decoder[i].resources.cache = request.kv[i]
```

The [reference norm/MoE bodies](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/muse_spark/modeling_muse_spark.py#L106-L273)
and [attention/residual body](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/muse_spark/modeling_muse_spark.py#L345-L492)
show why these names cannot collapse into one generic norm or residual add:

| Pure block, all with `resources {}` | Executable numerical contract |
| --- | --- |
| `CENTERED_RMS` | Normalize `x.float()`, multiply effective gain `cast(1+w,fp32)`, then cast to A. Preserve the source's dtype of the `1+w` addition. |
| `INPUT_SCALED_RMS` | Multiply `x.float()*cast(1+w,fp32)` **before** normalization; round through A and back to FP32, normalize, then cast to A. |
| `SCALELESS_RMS` | FP32 mean square and power `-0.5`, multiply and cast, no learned gain. Per-head variant reduces over D only. |
| `RESIDUAL_GATE` | `beta=sigmoid(w/0.3)`, `alpha=sqrt(max(sigmoid(-w/0.3)*(1+beta),0.001))`; output `cast(alpha*residual+beta*branch,A)`. Gate parameters/compute have their declared FP32 contract. |

Expert post-normalization occurs **before** weighted combine. Router input is
the unpooled activation. There is no shared-expert branch in this reference.
These are explicit dependencies for calibration/AWQ; an input-dependent norm
or residual gate is not automatically a foldable affine gain.

| Who | What | When |
| --- | --- | --- |
| `decoder.i.attn` | Only `kv[i]`, with local overwrite or full append capability | Cache read/update through attention completion in prefill/decode/verify. |
| Router, pool/norm, experts, unpool and residual gates | None | Empty external-resource interfaces. |
| Vision/adapter/merge | Patch/grid inputs, `vision_features:bf16[B,Nvision,8192]`, typed placeholder map | Image/video prefill; retain encoded features until scatter completes. |
| Audio conv `c` | Only `audio_left[c]:bf16[B,Cin[c],Lpad[c]]` and initialized flags | Streaming audio chunks, read old padding before publishing the retained tail; text decode does not need this port. |
| Audio merge | `audio_features:bf16[B,Naudio,8192]`, `i64` lengths and placeholder positions | After audio projection/norm through scatter completion. |

Vision executes patch embedding/absolute+2D positions, 50 individually selected
attention/MLP layers, token merging, adapter and text projection/norm before
scatter. Audio consumes waveform `fp32[B,Nsamples]`, explicitly casts for the
selected activation profile, traverses the four-stage causal pyramid, then
norm → biased projection → norm → embedding norm → scatter. Each audio
convolution's body concatenates its previous padding with the current input,
performs the declared convolution and publishes the current retained tail;
the [reference](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/muse_spark/modeling_muse_spark.py#L1035-L1257)
keeps this state in a padding cache. No hidden model-level cache is permitted
in the compiled form.

Gated checkpoint reconciliation, complete perception/audio bodies and mappings,
all precision records and numerical/chunk/state qualification remain required.
A future Dflash/assistant graph is a separate entry point with its own resource
contract; a comment mentioning it does not establish a supplied drafter model.
