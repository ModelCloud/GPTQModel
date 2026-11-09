# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Full GLM MLA/DSA MTP calibration without target-state or token aliases.

The checkpoint-only next-token layer consumes target POST-final-norm state
at t and the embedding of token t+1.  Its attention positions remain the
target positions at t, including native position-zero embedding masking.
The final unpaired target row is omitted independently for each prompt.

Primary execution contracts (also used by the full-GLM inference adapters):
* sglang fe1bd108372c9ba2ab959fcf99cf196433d155d5, eagle_worker_v2.py:
  _draft_extend_for_prefill shifts IDs per request while preserving positions.
* The same revision's worker requests post-norm target states, while
  deepseek_nextn.py evaluates enorm(embedding), hnorm(target), eh_proj,
  then shared_head.norm.
* vllm 5f63ea461ccd567ece6f50dd3701430c0f321039,
  vllm/models/deepseek_v32/nvidia/mtp.py uses this full-GLM contract too.

This is the ordinary full-GLM family.  It deliberately has no Flash mHC
stream reduction, does not borrow a Flash decoder and does not register
duplicate target embedding, final norm or vocabulary-head parameters.
"""
from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


MTP_CONTRACT = {
    "schema": "gptqmodel-glm5-mtp-post-norm-shifted-token-target-position-v1",
    "target_tap": "post-model.norm",
    "pairs": "target[t]-with-token[t+1]-within-each-prompt",
    "positions": "target-position[t]",
    "position_zero_embedding": "zero-before-enorm",
    "fusion": "concat-enorm-embedding-then-hnorm-target",
    "indexer": "own-full-indexer-no-target-topk-reuse",
    "output": "shared_head.norm-once-before-shared-target-head",
}


def _positions(token_ids: torch.Tensor, position_ids: torch.Tensor | None) -> torch.Tensor:
    if position_ids is None:
        return torch.arange(token_ids.shape[1], device=token_ids.device).expand_as(token_ids)
    if (position_ids.dtype != torch.int64 or position_ids.shape != token_ids.shape
            or position_ids.device != token_ids.device):
        raise ValueError("MTP positions must be int64 with the token shape and device")
    if bool((position_ids < 0).any()) or bool((position_ids[:, 1:] - position_ids[:, :-1] != 1).any()):
        raise ValueError("MTP cache-free prompts require consecutive nonnegative positions")
    return position_ids


@torch.no_grad()
def aligned_inputs(
    target_frontier: torch.Tensor,
    token_ids: torch.Tensor,
    embedding: torch.Tensor | nn.Embedding | Callable,
    target_norm: nn.Module | Callable | None,
    *,
    position_ids: torch.Tensor | None = None,
    target_tap: str = "pre_final_norm",
) -> dict[str, torch.Tensor]:
    """Build S-1 teacher-forced pairs from one or more unpadded prompts.

    A normal rolling frontier is the raw final decoder residual, so the
    default applies target_norm exactly once.  A caller that already stored
    post-final-norm state must explicitly select ``post_final_norm`` and pass
    no target_norm.  No prompt may borrow the first token of another prompt.
    """
    if (not isinstance(target_frontier, torch.Tensor) or target_frontier.ndim != 3
            or not target_frontier.is_floating_point() or target_frontier.shape[0] < 1
            or target_frontier.shape[1] < 2 or target_frontier.shape[2] < 1):
        raise ValueError("MTP frontier must be floating [batch, sequence>=2, hidden]")
    if (not isinstance(token_ids, torch.Tensor) or token_ids.dtype != torch.int64
            or token_ids.ndim != 2 or tuple(token_ids.shape) != tuple(target_frontier.shape[:2])):
        raise ValueError("MTP tokens must be int64 [batch, sequence] matching the frontier")
    if token_ids.device != target_frontier.device:
        raise ValueError("MTP tokens and target frontier must share a device")
    positions = _positions(token_ids, position_ids)[:, :-1].contiguous()
    shifted = token_ids[:, 1:].contiguous()
    previous = target_frontier[:, :-1].contiguous()
    if target_tap == "pre_final_norm":
        if target_norm is None:
            raise ValueError("raw target frontier requires the native target final norm")
        previous = target_norm(previous)
    elif target_tap == "post_final_norm":
        if target_norm is not None:
            raise ValueError("already normalized target must not receive a second final norm")
    else:
        raise ValueError("unknown MTP target tap")
    if previous.shape != target_frontier[:, :-1].shape or previous.dtype != target_frontier.dtype:
        raise ValueError("native target norm changed the MTP shape or activation dtype")
    if isinstance(embedding, torch.Tensor):
        if (embedding.ndim != 2 or embedding.shape[1] != target_frontier.shape[2]
                or embedding.device != target_frontier.device):
            raise ValueError("MTP embedding geometry or device differs from the frontier")
        if bool((shifted < 0).any()) or bool((shifted >= embedding.shape[0]).any()):
            raise ValueError("MTP next-token ID lies outside the embedding vocabulary")
        embedded = F.embedding(shifted, embedding)
    else:
        embedded = embedding(shifted)
    if (embedded.shape != previous.shape or embedded.dtype != previous.dtype
            or embedded.device != previous.device):
        raise ValueError("native embedding changed the MTP shape or activation dtype")
    embedded = torch.where(positions.unsqueeze(-1) == 0, torch.zeros_like(embedded), embedded)
    length = shifted.shape[1]
    causal = torch.ones((length, length), dtype=torch.bool, device=target_frontier.device).tril()
    return {
        "target_hidden": previous.contiguous(),
        "inputs_embeds": embedded.contiguous(),
        "input_ids": shifted,
        "position_ids": positions,
        "attention_mask": causal[None, None].expand(token_ids.shape[0], 1, length, length),
    }


class GlmMTP(nn.Module):
    """The native full-GLM MTP envelope, decoder and final shared-head norm."""

    def __init__(self, decoder, enorm, hnorm, eh_proj, shared_head_norm, *, layer_idx):
        super().__init__()
        self.decoder = decoder
        self.enorm, self.hnorm, self.eh_proj = enorm, hnorm, eh_proj
        self.shared_head_norm = shared_head_norm
        self.layer_idx = layer_idx
        self.config = decoder.config if hasattr(decoder, "config") else decoder.self_attn.config

    def envelope(self, aligned: dict[str, torch.Tensor]) -> torch.Tensor:
        """Expose the native eh_proj input so K4 capture follows its norms."""
        embedding, target = aligned["inputs_embeds"], aligned["target_hidden"]
        if embedding.shape != target.shape or embedding.dtype != target.dtype:
            raise ValueError("MTP aligned envelope geometry or dtype differs")
        return torch.cat((self.enorm(embedding), self.hnorm(target)), dim=-1)

    def projected_inputs(self, aligned: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.eh_proj(self.envelope(aligned))

    def forward_aligned(self, aligned: dict[str, torch.Tensor], rotary) -> dict[str, torch.Tensor]:
        hidden = self.projected_inputs(aligned)
        positions = aligned["position_ids"]
        hidden, topk = self.decoder(hidden,
            attention_mask=aligned["attention_mask"], position_ids=positions,
            position_embeddings=rotary(hidden, positions), prev_topk_indices=None,
            past_key_values=None, use_cache=False)
        return {"hidden": hidden, "normalized_hidden": self.shared_head_norm(hidden),
                "prev_topk_indices": topk}

    def forward(self, target_frontier, token_ids, embedding, target_norm, rotary, *,
                position_ids=None, target_tap="pre_final_norm"):
        aligned = aligned_inputs(target_frontier, token_ids, embedding, target_norm,
            position_ids=position_ids, target_tap=target_tap)
        return self.forward_aligned(aligned, rotary)


def _weight(module: nn.Module, source, name: str, device, shape) -> nn.Module:
    value = source.tensor(name, device)
    if value.dtype != torch.bfloat16 or tuple(value.shape) != tuple(shape):
        raise ValueError(f"native BF16 MTP geometry/dtype differs: {name}")
    module.weight = nn.Parameter(value, requires_grad=False)
    return module


@torch.no_grad()
def load_mtp(source, device="cuda:0", *, include_experts=False) -> GlmMTP:
    """Load source layer N using a full own indexer and the ordinary GLM block.

    ``source.materialize_block`` is shared with base-layer loading. It owns
    CPU transfers, defuses the source expert leaves when requested, retains
    F32 correction biases and applies the indexer FP32 execution policy.
    """
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaDecoderLayer,
        GlmMoeDsaRMSNorm,
    )
    config = copy.deepcopy(source.native_config())
    layer_idx = int(config.num_hidden_layers)
    if int(getattr(config, "num_nextn_predict_layers", 0)) != 1:
        raise ValueError("full GLM MTP loader requires exactly one native prediction layer")
    for name, final in (("mlp_layer_types", "sparse"), ("indexer_types", "full"),
                        ("layer_types", "deepseek_sparse_attention")):
        schedule = list(getattr(config, name))
        if len(schedule) != layer_idx:
            raise ValueError(f"native GLM MTP {name} schedule differs")
        setattr(config, name, schedule + [final])
    config.num_hidden_layers = layer_idx + 1
    config._attn_implementation = "sdpa"
    config.use_cache = False
    hidden, eps = int(config.hidden_size), float(config.rms_norm_eps)
    with torch.device("meta"):
        decoder = GlmMoeDsaDecoderLayer(config, layer_idx)
        enorm = GlmMoeDsaRMSNorm(hidden, eps=eps)
        hnorm = GlmMoeDsaRMSNorm(hidden, eps=eps)
        eh_proj = nn.Linear(hidden * 2, hidden, bias=False)
        shared_norm = GlmMoeDsaRMSNorm(hidden, eps=eps)
    source.materialize_block(decoder, layer_idx, device=device, include_experts=include_experts)
    if decoder.self_attn.indexer is None:
        raise ValueError("native GLM MTP requires its own full DSA indexer")
    prefix = f"model.layers.{layer_idx}."
    for module, suffix, shape in ((enorm, "enorm", [hidden]), (hnorm, "hnorm", [hidden]),
                                  (eh_proj, "eh_proj", [hidden, hidden * 2]),
                                  (shared_norm, "shared_head.norm", [hidden])):
        _weight(module, source, prefix + suffix + ".weight", device, shape)
    source.close()
    return GlmMTP(decoder, enorm, hnorm, eh_proj, shared_norm, layer_idx=layer_idx).eval()
