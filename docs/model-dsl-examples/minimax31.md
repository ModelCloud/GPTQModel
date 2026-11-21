<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniMax M3.1 Flash

Selected checkpoint: `MiniMaxAI/MiniMax-M3.1-Flash`, pinned
[config](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/ce7f4df87d2e05262f5775f373dac4772f134041/config.json).
See [sample conventions](README.md). The pinned Transformers
[configuration adapter](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/minimax_m3_vl/configuration_minimax_m3_vl.py#L115-L165)
explicitly resolves legacy `moe_layer_freq` and `sparse_attention_config` into
per-layer choices; defaults are not a replacement for checkpoint constants.

| Item | Resolved declaration |
| --- | --- |
| Text | 60 layers, `H=6144`, vocabulary `200064`, context `1048576`. |
| Heterogeneous FFN | Layers `0,1,2`: dense width `12288`. Layers `3..59`: 128 routed experts, top 4, width `3072`, plus one shared width-3072 expert. |
| Attention | All 60 sparse; 64 Q heads, 4 KV heads, D=128; Q width `8192` differs from H. Per-head zero-centered Q/K RMSNorm, epsilon `1e-6`; partial RoPE rotates 64 dimensions, theta `10000000`. No attention output gate. |
| Indexer | 4 index heads × 128, one index K vector/token; block size 128, top 16 blocks per index head, current local block always included. |
| Native storage | MXFP8, blocks `[1,32]`, dynamic activation scheme; configured routers/other ignored tensors keep their declared native storage. |
| Vision | 32 layers, width 1280, 16 heads, FFN 5120, patch 14, 3D RoPE; spatial merge 2 and temporal patch 2, projector output 6144. |

## Two attention histories, with separate block requirements

```text
request_resources MiniMaxRequest(B, C <= 1048576) {
    for i in 0..59: attention[i] = owned_request {
        k: bf16[B,4,C,128], v: bf16[B,4,C,128],
        index_k: bf16[B,1,C,128],
        positions: i64[B,C], valid: bool[B,C], lengths: i64[B],
        init=empty, lifetime=request, memory=device_global
    }
    image_features: bf16[B,Nimage,6144] {
        owner=request, init=vision.projector.done,
        lifetime=until_multimodal_scatter_done, memory=device_global
    }
}

block MINIMAX_INDEXER(x: A[B,T,6144], pos: i64[B,T], valid: bool[B,T]; w: IndexWeights)
    -> (selected: i64[B,4,T,Kblocks])
resources {
    keys: read_append<bf16[B,1,C,128]> when=prefill|decode|verify at index.read..index.done
    metadata: read_write<PositionMetadata> when=prefill|decode|verify at index.read..index.done
} {
    q = GEMMA_RMS_NORM(reshape(linear(x;w.q),[B,T,4,128]);w.q_norm)
    k = GEMMA_RMS_NORM(reshape(linear(x;w.k),[B,T,1,128]);w.k_norm)
    q,k = PARTIAL_ROPE(q,k,pos,rotary_dim=64,theta=10000000)
    all_k = append_visible(resource_read(keys),k,pos,valid)
    resource_write(keys,all_k)
    score: fp32[B,4,T,S] = matmul(cast(q,fp32),transpose(cast(all_k,fp32)))
    score = mask_future_and_invalid(score,pos,valid)
    blocks = max_pool_keys(pad(score,-infinity),block_size=128)
    blocks = set_current_local_block(blocks,+infinity)
    values,ids = topk(blocks,min(16,ceil_div(S,128)),policy=qualified_reference_topk)
    selected = where(values == -infinity,-1,ids)
    return selected at index.done
}

block MINIMAX_ATTN(x: A[B,T,6144], pos: i64[B,T], valid: bool[B,T]; w: AttnWeights)
    -> (y: A[B,T,6144])
resources {
    kv: read_append<MiniMaxKV> when=prefill|decode|verify at kv.read..attention.done
    index_keys: read_append<bf16[B,1,C,128]> when=prefill|decode|verify at index.read..index.done
    metadata: read_write<PositionMetadata> when=prefill|decode|verify at entry..attention.done
} {
    q,k,v = ATTN_QKV(x;w.q[8192,6144],w.k[512,6144],w.v[512,6144])
    q,k = per_head_gemma_qk_norm(q,k;w.q_norm,w.k_norm,D=128,eps=1e-6)
    q,k = PARTIAL_ROPE(q,k,pos,rotary_dim=64,theta=10000000)
    next = append_kv(resource_read(kv),k,v,pos,valid)
    resource_write(kv,next)
    selected = MINIMAX_INDEXER(x,pos,valid;w.indexer)
        resources {keys=index_keys,metadata=metadata}
    keep = blocks_to_keys(selected,block_size=128,query_heads_per_index_head=16)
    keep = keep & causal_and_valid(pos,next.positions,valid,next.valid)
    p = softmax(matmul(q,transpose(repeat_kv(next.k,16)))/sqrt(128),mask=keep,compute=fp32)
    a = matmul(cast(p,A),repeat_kv(next.v,16))
    return linear(flatten_heads(a);w.o[6144,8192]) at attention.done
}
```

`MiniMaxKV` contains K/V and their typed metadata from the manifest;
`PositionMetadata` contains `i64` positions/lengths and `bool` validity.
The index-key append and main-KV append publish one consistent logical token
extent. The indexer has no capability to read the main V cache. QKV, norms,
RoPE, block pooling, masks and softmax each declare `resources {}`. Their
values arrive through ports; only the shown wrappers touch external memory.
The pure score/mask equations and sentinel behavior follow the
[indexer reference](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L482-L627).
In particular, block selection does not replace token-level causality.

## Dense and MoE bodies are different layer records

The checkpoint's `swigluoai` is not ordinary SiLU(gate) × up. For **both** dense
and expert FFNs, the [reference](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L141-L264)
defines `g=min(gate,7)`, `u=clamp(up,-7,7)`, then
`product=(u+1)*g*sigmoid(1.702*g)` before down projection.

```text
block MINIMAX_FFN(x: A[N,6144]; w: FFNWeights, I: const) -> (y: A[N,6144]) resources {} pure {
    gate,up = GATE_UP(x;w.gate[I,6144],w.up[I,6144])
    g = minimum(gate,7)
    product = (clamp(up,-7,7)+1)*g*sigmoid(1.702*g)
    return linear(product;w.down[6144,I])
}

block MINIMAX_MOE(x: A[N,6144]; w: MoEWeights) -> (y: A[N,6144]) resources {} pure {
    score: fp32[N,128] = sigmoid(cast(linear(x;w.router),fp32))
    ids = topk_indices(score+w.selection_bias,4,sorted=false)
    p = gather(score,ids)
    weights = p/sum(p,expert_slot)
    routed = dispatch_ffn_combine(x,ids,weights;w.experts,body=MINIMAX_FFN,I=3072)
    shared = MINIMAX_FFN(x;w.shared,I=3072)
    return 2*routed + shared
}
```

Correction bias affects **selection**, not mixture weights. There is no
extra learned shared-expert gate here. Dense `0..2` and sparse `3..59` wrapper
bodies both use norm → attention → residual → norm → chosen FFN → residual,
with exact resource-interface forwarding to attention and none to the FFN.
Stored norm gain uses `1+w`; its cast order remains a separate numerical
contract from Qwen/Muse norms, despite the shared convention name.

| Who | What | When |
| --- | --- | --- |
| `decoder.i.attn` | Layer-i K/V `bf16[B,4,C,128]` and position metadata | Prefill/decode/verify cache interval; retained across steps. |
| `decoder.i.attn.indexer` | Only layer-i index K `bf16[B,1,C,128]` and metadata | Before sparse attention selection; retained across steps. |
| Dense/shared/routed FFN and router | None | Empty request-resource set; weights and routing values are explicit ports. |
| Vision/projector → multimodal merge | `image_features: bf16[B,Nimage,6144]` and typed image/video placement map | Multimodal prefill only, after encoder completion through scatter completion. Decode uses the resulting text caches. |

Multimodal prefill explicitly embeds patches, executes the 32 vision layers,
applies spatial/temporal patch merging and the projector, then scatters image
tokens into text embeddings. Its complete encoder/projector/processor graph
and checkpoint-key conversion still require expansion and parity qualification;
a `vision.forward` callback is not an acceptable implementation.
The GPT-QModel tree covers a current quantization inventory, not proof that
this proposed sparse executor, MXFP8 lowering or multimodal entry point exists.
