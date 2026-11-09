"""Packed V4.1 source linear execution using pinned reference kernels.

Allocate explicit BF16 outputs instead of relying on process-global default
dtype. The kernel module is supplied by the caller from an authenticated source.
"""

import torch
from torch import nn
from transformers.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41Attention,
    DeepseekV41Indexer,
    select_candidate_blocks,
)


def reference_rotary(x, cos, sin, inverse=False):
    """Use the checkpoint's complex multiplication and single final rounding."""
    width = cos.shape[-1] * 2
    result = x.clone()
    pairs = torch.view_as_complex(x[..., -width:].float().reshape(*x.shape[:-1], -1, 2))
    coeff = torch.complex(cos, -sin if inverse else sin)
    if x.ndim == 4:
        coeff = coeff.unsqueeze(2)
    result[..., -width:] = torch.view_as_real(pairs * coeff).flatten(-2)
    return result


class V41NativeIndexer(DeepseekV41Indexer):
    """Bounded query scoring with the checkpoint's BF16 arithmetic and order."""

    def forward(self, hidden_states, q_residual, latent, group_positions,
                position_ids, cache_layer, shared):
        if cache_layer is not None:
            raise ValueError("native calibration indexer requires a full prompt")
        kernels = self.v41_source_kernels
        batch, length, _ = hidden_states.shape
        if self.owns_k:
            shared["index_k"] = None
            if latent is not None:
                key = self.k_norm(self.k_proj(latent))
                cos, sin = self.rotary_emb(key, group_positions, "compress")
                key = reference_rotary(key, cos, sin)
                kernels.fp4_act_quant(key, 32, True)
                shared["index_k"] = key.unsqueeze(1)
        keys = shared.get("index_k")
        if keys is None or keys.shape[2] == 0:
            shared["topk_idx"] = None
            if self.is_candidate_source:
                shared["candidates"] = None
            return
        keys = keys[:, 0].to(hidden_states.device)
        cos, sin = self.rotary_emb(hidden_states, position_ids, "compress")
        query = self.q_b_proj(q_residual).view(batch, length, self.num_heads, self.head_dim)
        query = reference_rotary(query, cos, sin)
        kernels.fp4_act_quant(query, 32, True)
        weights = self.weights_proj(hidden_states) * (self.softmax_scale * self.heads_scaling)
        lengths = shared["compress_lens"].to(keys.device).unsqueeze(-1)
        entries = torch.arange(keys.shape[1], device=keys.device)
        top_k = min(self.index_topk, keys.shape[1])
        candidates = shared.get("candidates") if self.uses_candidates else None
        # 64 MiB score temporary, independent of model context capacity.
        chunk = max(1, min(length, (32 * 1024 * 1024) // (batch * self.num_heads * keys.shape[1])))
        selected, published = [], []
        for start in range(0, length, chunk):
            stop = start + chunk
            scores = torch.einsum("bshd,btd->bsht", query[:, start:stop], keys)
            scores = (scores.relu_() * weights[:, start:stop].unsqueeze(-1)).sum(dim=2)
            scores.masked_fill_(entries >= lengths[:, start:stop], -torch.inf)
            if self.is_candidate_source:
                published.append(select_candidate_blocks(scores, lengths[:, start:stop],
                                                          self.candidate_topk_blocks, self.candidate_block_size))
            elif candidates is not None:
                scores.masked_fill_(~candidates[:, start:stop].to(scores.device), -torch.inf)
            indices = scores.topk(top_k, dim=-1, sorted=False).indices.sort(dim=-1).values
            selected.append(torch.where(indices < lengths[:, start:stop], indices, -1))
        shared["topk_idx"] = torch.cat(selected, dim=1)
        if self.is_candidate_source:
            shared["candidates"] = torch.cat(published, dim=1)


class V41NativeAttention(DeepseekV41Attention):
    """Full-prompt calibrated forward; reuse shared compressed storage directly."""

    def forward(self, hidden_states, shared, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        if past_key_values is not None:
            raise ValueError("V4.1 quantization attention expects independent full prompts")
        kernels = self.v41_source_kernels
        batch, length, _ = hidden_states.shape
        positions = kwargs["position_ids"]
        padding = kwargs.get("padding_mask")
        cos, sin = position_embeddings[self.rope_layer_type]
        qr = self.q_a_norm(self.q_a_proj(hidden_states))
        q = reference_rotary(self.q_b_proj(qr).view(batch, length, self.num_heads, self.head_dim), cos, sin)
        kv = reference_rotary(self.kv_norm(self.kv_proj(hidden_states)), cos, sin)
        kernels.act_quant(kv, 32, "ue8m0", torch.float8_e8m0fnu, True)
        end = torch.arange(length, device=kv.device).unsqueeze(-1)
        indices = (end - self.sliding_window + 1).clamp_min(0) + torch.arange(min(length, self.sliding_window), device=kv.device)
        indices = indices.expand(batch, -1, -1).clone()
        valid = indices <= end
        if attention_mask is not None:
            mask = attention_mask.expand(batch, -1, -1, -1)[:, 0]
            allowed = mask.gather(-1, indices.clamp_max(length - 1))
            valid = valid & (allowed if allowed.dtype == torch.bool else allowed == 0)
        indices = indices.masked_fill(~valid, -1)
        if self.compress_ratio:
            latent = group_positions = None
            if self.is_kv_source:
                shared["compress_kv"] = None
                latent, group_positions, counts, lengths = self.compressor(hidden_states, None, positions, padding)
                shared["group_counts"] = counts
                shared["compress_lens"] = lengths // self.compress_ratio
                if padding is not None:
                    shared["compress_lens"] = shared["compress_lens"].masked_fill(~padding, 0)
            if self.is_index_source:
                self.indexer(hidden_states, qr, latent, group_positions, positions, None, shared)
            if latent is not None:
                cc, cs = self.compress_rotary(latent, position_ids=group_positions, layer_type="compress")
                rotated = reference_rotary(latent, cc, cs)
                kernels.fp4_act_quant(rotated, 16, True, scale_dtype=torch.float8_e4m3fn)
                shared["compress_kv"] = rotated.unsqueeze(1)
            compressed, selected = shared.get("compress_kv"), shared.get("topk_idx")
            if compressed is not None and selected is not None and selected.shape[-1]:
                selected = selected.to(kv.device)
                extra = torch.where(selected >= 0, selected + length, -1)
                indices = torch.cat((indices, extra), dim=-1)
                kv = torch.cat((kv, compressed[:, 0].to(kv.device)), dim=1)
        indices = indices.to(torch.int32).contiguous()
        output = torch.empty_like(q)
        for start in range(0, self.num_heads, 16):
            output[:, :, start:start + 16] = kernels.sparse_attn(
                q[:, :, start:start + 16].contiguous(), kv.contiguous(),
                self.sinks[start:start + 16].contiguous(), indices, self.scaling)
        output = reference_rotary(output, cos, sin, inverse=True)
        grouped = output.reshape(batch, length, self.config.o_groups, -1)
        return self.o_b_proj(self.o_a_proj(grouped).flatten(2)), None




def dspark_joint_ring(main_kv, anchors, window_size):
    """Gather independent reference-order rings from one already-projected prefix."""
    slots = torch.arange(window_size, device=main_kv.device)
    history = anchors[:, None] - ((anchors[:, None] - slots) % window_size)
    valid = history >= 0
    window = main_kv[0, history.clamp_min(0)].masked_fill(~valid[..., None], 0)
    indices = slots[None].expand(anchors.numel(), -1).masked_fill(~valid, -1)
    return window, indices


class V41NativeDSparkAttention(DeepseekV41Attention):
    """Draft block attention over an explicit main-history window and all drafts.

    main_x contains projected main features at absolute positions 0..P. Keeping
    this input explicit makes repeated quantization replay independent of caches.
    The caller must provide at least two main positions, as the source's P=0
    path only seeds caches and never executes routed experts.
    """

    def forward(self, hidden_states, shared, *, main_x, anchor_positions=None, **kwargs):
        if kwargs.get("past_key_values") is not None or self.compress_ratio:
            raise ValueError("dSpark replay requires explicit history and no decode cache")
        batch, drafts, _ = hidden_states.shape
        joint = anchor_positions is not None
        if main_x.ndim != 3 or main_x.shape[0] != (1 if joint else batch) or main_x.shape[1] < 2:
            raise ValueError("dSpark needs matching batch and at least two main-history positions")
        kernels = self.v41_source_kernels
        length = main_x.shape[1]
        win = self.sliding_window
        # Only the last window is needed. Scatter to the reference's ring order
        # so sparse softmax visits keys in exactly the same order.
        if joint and (anchor_positions.shape != (batch,) or anchor_positions.dtype != torch.long
                      or anchor_positions.min() < 1 or anchor_positions.max() >= length
                      or not torch.all(anchor_positions[1:] > anchor_positions[:-1])):
            raise ValueError("invalid joint dSpark anchor positions")
        positions = torch.arange(0 if joint else max(0, length - win), length,
                                 device=hidden_states.device)[None]
        main = main_x if joint else main_x[:, -win:]
        cos, sin = self.compress_rotary(main, position_ids=positions, layer_type="main")
        main_kv = reference_rotary(self.kv_norm(self.kv_proj(main)), cos, sin)
        kernels.act_quant(main_kv, 32, "ue8m0", torch.float8_e8m0fnu, True)
        if joint:
            # Project each main row once, then gather each anchor's ring in the
            # same slot order as the reference cache. No future key is exposed.
            anchors = anchor_positions.to(hidden_states.device)
            window, main_indices = dspark_joint_ring(main_kv, anchors, win)
            draft_positions = anchors[:, None] + 1 + torch.arange(drafts, device=hidden_states.device)
        else:
            window = torch.zeros(batch, win, self.head_dim, device=main_kv.device, dtype=main_kv.dtype)
            window.index_copy_(1, positions[0] % win, main_kv)
            draft_positions = torch.arange(length, length + drafts, device=hidden_states.device)[None]
        cos, sin = self.compress_rotary(hidden_states, position_ids=draft_positions, layer_type="main")
        qr = self.q_a_norm(self.q_a_proj(hidden_states))
        q = reference_rotary(self.q_b_proj(qr).view(batch, drafts, self.num_heads, self.head_dim), cos, sin)
        kv = reference_rotary(self.kv_norm(self.kv_proj(hidden_states)), cos, sin)
        kernels.act_quant(kv, 32, "ue8m0", torch.float8_e8m0fnu, True)
        bank = torch.cat((window, kv), dim=1)
        if joint:
            draft_indices = (win + torch.arange(drafts, device=q.device))[None].expand(batch, -1)
            indices = torch.cat((main_indices, draft_indices), dim=-1)[:, None].expand(-1, drafts, -1)
        else:
            indices = torch.cat((torch.arange(min(win, length), device=q.device),
                                 win + torch.arange(drafts, device=q.device)))
            indices = indices.view(1, 1, -1).expand(batch, drafts, -1)
        indices = indices.to(torch.int32).contiguous()
        output = torch.empty_like(q)
        for start in range(0, self.num_heads, 16):
            output[:, :, start:start + 16] = kernels.sparse_attn(
                q[:, :, start:start + 16].contiguous(), bank,
                self.sinks[start:start + 16].contiguous(), indices, self.scaling)
        output = reference_rotary(output, cos, sin, inverse=True)
        grouped = output.reshape(batch, drafts, self.config.o_groups, -1)
        return self.o_b_proj(self.o_a_proj(grouped).flatten(2)), None


class V41NativeLinear(nn.Module):
    def __init__(self, weight, scale, kernels):
        super().__init__()
        if weight.ndim != 2 or weight.dtype not in (torch.int8, torch.float8_e4m3fn):
            raise ValueError("expected packed FP4 or FP8 V4.1 matrix")
        if scale.dtype != torch.float8_e8m0fnu:
            raise ValueError("V4.1 source linear scales must be E8M0")
        self.packed_fp4 = weight.dtype == torch.int8
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1] * (2 if self.packed_fp4 else 1)
        expected = ((self.out_features, self.in_features // 32) if self.packed_fp4
                    else ((self.out_features + 31) // 32, self.in_features // 32))
        if self.in_features % 32 or tuple(scale.shape) != expected:
            raise ValueError("invalid V4.1 matrix/scale geometry")
        if self.packed_fp4:
            weight = weight.view(torch.float4_e2m1fn_x2)
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.kernels = kernels

    @torch.no_grad()
    def forward(self, x):
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.in_features:
            raise ValueError("native V4.1 GEMM requires BF16 inputs of the declared width")
        rows = x.numel() // self.in_features
        result = torch.empty((*x.shape[:-1], self.out_features), device=x.device, dtype=torch.bfloat16)
        if not rows:
            return result
        a, a_scale = self.kernels.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        if self.packed_fp4:
            kernel = self.kernels.fp4_gemm_kernel(self.out_features, self.in_features,
                                                 act_block_size=32, scale_dtype=self.kernels.FE8M0)
        else:
            kernel = self.kernels.fp8_gemm_kernel(self.out_features, self.in_features,
                                                 block_size=32, scale_dtype=self.kernels.FE8M0)
        kernel(a.reshape(rows, self.in_features), self.weight,
               result.reshape(rows, self.out_features), a_scale.reshape(rows, -1), self.scale)
        return result
