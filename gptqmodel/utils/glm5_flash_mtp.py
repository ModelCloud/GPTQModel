# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-only, non-mHC Flash MTP envelope and decoder.

This uses the existing Glm5NextMTPDecoderLayer contract: normalized target
stream-mean at t, embedding at t+1, and the shifted token's position t+1.
It is deliberately distinct from full-GLM's target-position convention.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from . import glm5_flash_core as core


MTP_CONTRACT = {"schema": "gptqmodel-glm5-flash-mtp-shifted-token-position-v1",
    "target_tap": "post-model.norm-unweighted-mhc-stream-mean",
    "pairs": "target[t]-with-token[t+1]-within-each-prompt",
    "positions": "shifted-token-position[t+1]", "position_zero_embedding": "zero-before-enorm",
    "fusion": "concat-enorm-embedding-then-hnorm-target", "decoder": "non-mHC-MLA-and-clamped-MoE",
    "indexer": "own-full-indexer", "output": "shared_head.norm-once-before-shared-target-head"}


@torch.inference_mode()
def aligned_inputs(target_frontier, token_ids, embedding, target_norm, *, position_ids=None):
    if (target_frontier.ndim != 4 or target_frontier.shape[1] < 2
            or token_ids.dtype != torch.int64 or token_ids.shape != target_frontier.shape[:2]
            or token_ids.device != target_frontier.device):
        raise ValueError("Flash MTP requires matching [B,S,HC,H] and int64 [B,S] tokens")
    if position_ids is None:
        position_ids = torch.arange(token_ids.shape[1], device=token_ids.device).expand_as(token_ids)
    if (position_ids.shape != token_ids.shape or position_ids.dtype != torch.int64
            or position_ids.device != token_ids.device or bool((position_ids < 0).any())
            or bool((position_ids[:, 1:] - position_ids[:, :-1] != 1).any())):
        raise ValueError("Flash MTP requires consecutive native positions")
    shifted = token_ids[:, 1:].contiguous()
    positions = position_ids[:, 1:].contiguous()
    previous = target_norm(target_frontier.mean(dim=2))[:, :-1].contiguous()
    embedded = F.embedding(shifted, embedding) if isinstance(embedding, torch.Tensor) else embedding(shifted)
    if embedded.shape != previous.shape or embedded.dtype != previous.dtype:
        raise ValueError("Flash MTP embedding and target norm geometry differs")
    embedded = torch.where(positions.unsqueeze(-1) == 0, torch.zeros_like(embedded), embedded)
    return {"target_hidden": previous, "inputs_embeds": embedded, "input_ids": shifted,
                "position_ids": positions, "attention_mask": torch.ones_like(shifted, dtype=torch.bool)}


class FlashMTPDecoder(nn.Module):
    """Ordinary residual decoder, with exactly the checkpoint's MTP leaves."""
    def __init__(self, config, layer_idx):
        super().__init__()
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextAttention,
            Glm5NextTextMoE,
            Glm5NextTextRMSNorm,
        )
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = "deepseek_sparse_attention"
        self.input_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Glm5NextTextAttention(config, layer_idx)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Glm5NextTextMoE(config)

    def forward(self, hidden, **context):
        context.pop("use_cache", None)
        attention, _, topk = self.self_attn(self.input_layernorm(hidden), **context)
        residual = hidden + attention
        return residual + self.mlp(self.post_attention_layernorm(residual)), topk


class GlmFlashMTP(nn.Module):
    def __init__(self, decoder, enorm, hnorm, eh_proj, shared_head_norm):
        super().__init__()
        self.decoder = decoder
        self.enorm, self.hnorm, self.eh_proj = enorm, hnorm, eh_proj
        self.shared_head_norm = shared_head_norm
        self.config, self.layer_idx = decoder.config, decoder.layer_idx

    def envelope(self, aligned):
        return torch.cat((self.enorm(aligned["inputs_embeds"]), self.hnorm(aligned["target_hidden"])), dim=-1)

    def forward_aligned(self, aligned, rotary=None):
        hidden = self.eh_proj(self.envelope(aligned))
        output = core.native_forward(self.decoder, {"hidden": hidden, "position_ids": aligned["position_ids"]})
        output["normalized_hidden"] = self.shared_head_norm(output["hidden"])
        return output


@torch.inference_mode()
def load_mtp(source, device="cuda:0", *, include_experts=False):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm
    config = copy.deepcopy(source.native_config())
    layer = config.num_hidden_layers
    if config.num_nextn_predict_layers != 1:
        raise ValueError("Flash MTP requires exactly one checkpoint prediction layer")
    for attr, kind in (("layer_types", "deepseek_sparse_attention"),
                       ("mlp_layer_types", "sparse"), ("indexer_types", "full")):
        values = list(getattr(config, attr))
        if len(values) != layer:
            raise ValueError("Flash native MTP schedule differs")
        setattr(config, attr, values + [kind])
    config.num_hidden_layers += 1
    width, eps = config.hidden_size, config.rms_norm_eps
    with torch.device("meta"):
        decoder = FlashMTPDecoder(config, layer)
        enorm = Glm5NextTextRMSNorm(width, eps)
        hnorm = Glm5NextTextRMSNorm(width, eps)
        eh_proj = nn.Linear(width * 2, width, bias=False)
        shared = Glm5NextTextRMSNorm(width, eps)
    source.materialize_block(decoder, layer, device, include_experts=include_experts)
    prefix = f"{core.PREFIX}layers.{layer}."
    for obj, suffix, shape in ((enorm, "enorm", [width]), (hnorm, "hnorm", [width]),
                               (eh_proj, "eh_proj", [width, width * 2]),
                               (shared, "shared_head.norm", [width])):
        value = source.tensor(prefix + suffix + ".weight", device)
        if value.dtype != torch.bfloat16 or list(value.shape) != shape:
            raise ValueError("Flash MTP checkpoint envelope differs")
        obj.weight = nn.Parameter(value, requires_grad=False)
    source.close()
    return GlmFlashMTP(decoder, enorm, hnorm, eh_proj, shared).eval()
