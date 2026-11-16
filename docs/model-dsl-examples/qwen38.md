<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen 3.8 MoE

Selected checkpoint: `Qwen/Qwen3.8-2.4T-A95B`, pinned
[config](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/4f885e51d6bbfee5866446d56d3f96f336c39723/config.json).
This is a text MoE config (`qwen3_5_moe_text`), not a license to reuse another
family's forward without checking every semantic attribute. See the
[sample conventions](README.md).

## Exact layer and subsection choices

| Item | Checkpoint declaration |
| --- | --- |
| Backbone | `H=8192`, 92 layers, vocabulary `248320`, context `262144`. |
| Mixer schedule | The explicit `layer_types` list is `[linear,linear,linear,full]` repeated 23 times: full layers `3,7,...,91`, all other indices recurrent. Emit all 92 records. |
| GatedDeltaNet | 16 Q/K heads × 128, 128 V heads × 128, depthwise convolution width 4; QKV widths `2048+2048+16384=20480`, output-gate width `16384`. |
| Full attention | 64 Q heads, 4 KV heads, head dim 256; Q width `16384`, K/V `1024`. Q projection additionally emits a same-width output gate. Partial RoPE rotates 64 of 256 dimensions, theta `10000000`. |
| MoE in every layer | 512 routed experts, top 10, expert width 2048; one shared expert of width 2048 and a separate scalar sigmoid gate. |
| Norm | Zero-centered gain `1+w`, epsilon `1e-6`, FP32 normalization and gain multiplication before the output cast. |

The [candidate reference](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L425-L652)
exposes the recurrent body and its two state resources. This source has separate
`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`; assuming a fused QKVZ
checkpoint projection from a related family would give the wrong bindings.

```text
request_resources Qwen38Request(B, C <= 262144) {
    for i in linear_layer_ids: gdn[i] = owned_request {
        conv: bf16[B,20480,4], matrix: fp32[B,128,128,128],
        initialized: bool[B], processed: i64[B],
        init=zeros, lifetime=request, memory=device_global
    }
    for i in full_layer_ids: kv[i] = owned_request {
        k: bf16[B,4,C,256], v: bf16[B,4,C,256],
        positions: i64[B,C], valid: bool[B,C], lengths: i64[B],
        init=empty, lifetime=request, memory=device_global
    }
}

block QWEN_GDN(x: A[B,T,8192], valid: bool[B,T]; w: GDNWeights)
    -> (y: A[B,T,8192])
resources {
    conv: read_write<bf16[B,20480,4]> when=prefill|decode|verify at conv.read..conv.done
    matrix: read_write<fp32[B,128,128,128]> when=prefill|decode|verify at scan.read..scan.done
    progress: read_write<GDNProgress> when=prefill|decode|verify at entry..done
} {
    x0 = mask_padding(x,valid)
    qkv = linear(x0; w.in_proj_qkv[20480,8192])
    z = reshape(linear(x0; w.in_proj_z[16384,8192]),[B,T,128,128])
    b = linear(x0; w.in_proj_b[128,8192])
    a = linear(x0; w.in_proj_a[128,8192])
    c0 = resource_read(conv) at conv.read
    c1,qkv1 = CAUSAL_DEPTHWISE_CONV_SILU(qkv,c0; w.conv, kernel=4)
    resource_write(conv,c1) at conv.done
    q,k,v = split_reshape(qkv1,widths=[2048,2048,16384],heads=[16,16,128],dim=128)
    beta = cast(sigmoid(b),fp32)
    g = -exp(w.A_log_fp32)*softplus(cast(a,fp32)+w.dt_bias_fp32)
    s0 = resource_read(matrix) at scan.read
    o,s1 = GATED_DELTA_SCAN(q,k,v,g,beta,s0,valid)
    resource_write(matrix,s1) at scan.done
    o = RMS_NORM_GATED(o,z; w.norm,gate=silu,norm_before_gate=true)
    y = linear(flatten_heads(o); w.out_proj[8192,16384])
    resource_write(progress,advance_valid_counts(resource_read(progress),valid)) at done
    return y
}
```

`GDNProgress` is the typed `initialized/processed` record above. The convolution
and scan helpers each declare `resources {}`: their state is an explicit value
input/output; only the wrapper reads/writes request-global memory. The
convolution appends each token to the four-slot causal channel history, takes
the depthwise weighted sum, applies SiLU and returns the retained history.
Document resets and padding behavior are explicit, not inferred from `B`.

`GATED_DELTA_SCAN` has inputs Q/K `A[B,T,16,128]`, V `A[B,T,128,128]`,
g/beta `fp32[B,T,128]`, state `fp32[B,128,128,128]` and validity; outputs O
`A[B,T,128,128]` and the same state type. It declares `resources {}`. Its
portable body repeats Q/K heads eightfold, L2-normalizes them in FP32 with
epsilon `1e-6`, scales Q by `1/sqrt(128)` and scans valid tokens:

```text
S_decay = exp(g_t)[:, :, None, None] * S
delta = beta_t[...,None] * (v_t - sum_key(k_t[...,None] * S_decay))
S_next = S_decay + k_t[...,None] * delta[...,None,:]
o_t = cast(sum_key(q_t[...,None] * S_next), A)
```

The matrix axes are `[batch,value_head,key_feature,value_feature]`. A chunked
kernel and a one-token recurrence are alternative lowerings of this body,
requiring state/chunk/numerical qualification; a full-attention KV cache is
not a substitute for either state resource.

## MoE and full-attention specialization

The [router/shared-expert reference](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L873-L934)
gives this pure correlated region:

```text
block QWEN_MOE(x: A[N,8192]; w: MoEWeights) -> (y: A[N,8192]) resources {} pure {
    logits = linear(x; w.router[512,8192])
    probabilities: fp32[N,512] = softmax(logits,compute=fp32)
    p,ids = topk(probabilities,10,policy=qualified_reference_topk)
    weights: A[N,10] = cast(p/sum(p,expert_slot),A)
    routed = dispatch_swiglu_experts_combine(x,ids,weights; w.experts,I=2048)
    shared = MLP(x; w.shared,I=2048)
    gate: A[N,1] = sigmoid(linear(x; w.shared_gate[1,8192]))
    return routed + gate*shared
}
```

Dispatch/expert/combine expands into gather, `GATE_UP`, SiLU multiplication,
down projection and weighted scatter/reduction, each with an empty external
resource set. Expert order, routing weight casts and reduction precision are
part of the specialization, not a generic "MoE" callback.

Full-attention layers require only their own `kv[i]` resource, read/append at
their cache-update/attention interval. They split Q and gate **within each
head** from `[B,T,64,512]`, apply zero-centered Q/K RMSNorm, rotate the first
64 dimensions, append K/V, perform causal GQA, apply the declared output gate
and project `16384 -> 8192`. A contiguous global half-split of the Q projection
would be incorrect. The surrounding layer is input norm → mixer → residual
add → post-attention norm → MoE → residual add.

**Unresolved semantic guard:** this checkpoint explicitly says
`output_gate_type="swish"`, while the pinned candidate
[full-attention body](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L768-L813)
uses sigmoid. These are different functions. The manifest must bind an exact
verified 3.8 gate body; the compiler must reject this conflicting pairing rather
than quietly execute the borrowed 3.5 body. This sample does not claim that
the candidate reference validates Qwen 3.8 execution.

| Block instance | Required request resources | When |
| --- | --- | --- |
| Linear mixer at `i` | `gdn[i].conv`, `.matrix`, typed progress | Prefill/decode/verify, only around its convolution/scan; retain for future chunks. |
| Full mixer at `i=3+4k` | `kv[i]` | Cache read/append through attention completion. No recurrent matrix. |
| Norm/QKV/expert/router/shared gate | None | Empty interfaces in every phase; their inputs can still be AWQ consumers. |
| Optional MTP entry | Separately declared draft state and aliases | Only if its complete body and parameter mapping are supplied; `mtp_num_hidden_layers=1` alone does not establish executable support. |

Weights bind explicitly under `model.layers.{i}.linear_attn` or `.self_attn`
according to the layer record and under `.mlp` for router/experts/shared expert.
The resolver must reconcile the full checkpoint key map and any packed-expert
conversion before admission. No missing path selects a different mixer.
Calibration snapshots both recurrent and convolution state, not merely hidden
states; all router/shared-gate consumers participate in a proposed AWQ fold.
