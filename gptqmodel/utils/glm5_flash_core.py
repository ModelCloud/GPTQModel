# SPDX-License-Identifier: Apache-2.0
"""Bounded checkpoint-native GLM Flash blocks for causal EXL3 calibration.

The model's mHC and KDA arithmetic is provided by its Transformers decoder.
Only independent KDA heads are sliced to bound the reference's scratch space.
Checkpoint tensors always own their storage; at most one shard stays mapped.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from . import glm5_core as base


PREFIX = "model.language_model."
DenseRawHessian = base.DenseRawHessian
capture_linear_input = base.capture_linear_input
quantize_projection = base.quantize_projection
reconstruct_projection = base.reconstruct_projection
validate_projection = base.validate_projection
router_choices = base.router_choices


def checkpoint_names(key):
    names = {
        "attn_hc.base": "hc_attn_base", "attn_hc.fn": "hc_attn_fn",
        "attn_hc.scale": "hc_attn_scale", "ffn_hc.base": "hc_ffn_base",
        "ffn_hc.fn": "hc_ffn_fn", "ffn_hc.scale": "hc_ffn_scale",
        "self_attn.forget_gate.f_a_proj.weight": "self_attn.f_a_proj.weight",
        "self_attn.forget_gate.f_b_proj.weight": "self_attn.f_b_proj.weight",
        "self_attn.forget_gate.dt_bias": "self_attn.dt_bias",
        "self_attn.forget_gate.A_log": "self_attn.A_log",
    }
    if key == "self_attn.conv1d.weight":
        return [f"self_attn.{part}_conv1d.weight" for part in "qkv"]
    return [names.get(key, key)]


def module(block, relative_name):
    """Resolve checkpoint names without registering parameter aliases."""
    if relative_name.startswith(("self_attn.f_a_proj", "self_attn.f_b_proj")):
        relative_name = relative_name.replace("self_attn.", "self_attn.forget_gate.", 1)
    return block.get_submodule(relative_name)


def bounded_kda(function, heads=4):
    """Slice independent heads; preserve the official chunked recurrence."""
    def forward(query, key, value, g, beta, chunk_size=64, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        outputs, states = [], []
        for start in range(0, query.shape[2], heads):
            end = start + heads
            result, state = function(query[:, :, start:end], key[:, :, start:end], value[:, :, start:end],
                g[:, :, start:end], beta[:, :, start:end], chunk_size=chunk_size,
                initial_state=None if initial_state is None else initial_state[:, start:end],
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                **kwargs)
            outputs.append(result)
            if state is not None:
                states.append(state)
        return torch.cat(outputs, dim=2), torch.cat(states, dim=1) if states else None
    forward._gptqmodel_bounded_heads = heads
    forward._gptqmodel_kda_original = function
    return forward


def install_bounded_kda():
    from transformers.models.glm5_next import modeling_glm5_next as native
    if not getattr(native.chunk_kimi_delta_attention, "_gptqmodel_bounded_heads", None):
        native.chunk_kimi_delta_attention = bounded_kda(native.chunk_kimi_delta_attention)


class GlmFlashBF16Source(base.GlmBF16Source):
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).resolve()
        self.document = json.loads((self.snapshot / "config.json").read_text())
        self.config = self.document["text_config"]
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_text())
        self.weight_map = self.index["weight_map"]
        if self.document.get("model_type") != "glm5_next" or self.document.get("quantization_config"):
            raise ValueError("expected native BF16 GLM Flash checkpoint")
        if self.config.get("qk_rope_head_dim") != 0 or self.config.get("hc_mult") != 4:
            raise ValueError("unreviewed GLM Flash attention or mHC geometry")
        self._reader = self._reader_context = self._reader_shard = None

    def native_config(self):
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
        config = Glm5NextTextConfig.from_dict(self.config)
        config._attn_implementation = "sdpa"
        config._experts_implementation = "eager"
        config.use_cache = False
        return config

    @torch.no_grad()
    def materialize_block(self, block, layer, device="cuda:0", *, include_experts=False):
        prefix = f"{PREFIX}layers.{layer}."
        state = block.state_dict()
        assignments = []
        for key, value in state.items():
            if key.startswith("mlp.experts.") or not value.is_meta:
                continue
            names = [prefix + name for name in checkpoint_names(key)]
            assignments.append((min(self.weight_map[name] for name in names), key, names))
        try:
            for _, key, names in sorted(assignments):
                tensors = [self.tensor(name) for name in names]
                value = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
                if tuple(value.shape) != tuple(state[key].shape):
                    raise ValueError(f"native Flash state shape differs: {prefix + key}")
                if value.dtype not in (torch.bfloat16, torch.float32):
                    raise ValueError(f"non-native Flash tensor dtype: {prefix + key}")
                # Match the official model's keep-in-FP32 execution policy;
                # export still preserves the original checkpoint bytes.
                if any(part in key for part in ("conv1d", "dt_bias", "A_log", "e_score_correction_bias")):
                    value = value.float()
                value = value.to(device)
                parent, _, leaf = key.rpartition(".")
                owner = block.get_submodule(parent) if parent else block
                if leaf in owner._buffers:
                    owner._buffers[leaf] = value
                else:
                    owner._parameters[leaf] = nn.Parameter(value, requires_grad=False)
            block.layer_idx = layer
            if hasattr(block.mlp, "gate"):
                base.bind_sigmoid_grouped_router_recovery(block.mlp.gate)
            if include_experts and hasattr(block.mlp, "experts"):
                self.materialize_routed_experts(block, layer, device)
            for key, value in (*block.named_parameters(), *block.named_buffers()):
                if value.is_meta and (include_experts or not key.startswith("mlp.experts.")):
                    raise ValueError(f"native Flash state remains meta: {key}")
        finally:
            self.close()
        return block.eval()

    @torch.no_grad()
    def materialize_routed_experts(self, block, layer=None, device="cuda:0"):
        if getattr(block, "_gptqmodel_glm5_experts_complete", False):
            return block
        layer = block.layer_idx if layer is None else layer
        experts = block.mlp.experts
        if set(experts.state_dict()) != {"gate_up_proj", "down_proj"}:
            raise ValueError("native Flash fused expert inventory differs")
        shapes = {k: tuple(v.shape) for k, v in experts.state_dict().items()}
        count, doubled, width = shapes["gate_up_proj"]
        inter = doubled // 2
        if doubled % 2 or shapes["down_proj"] != (count, width, inter):
            raise ValueError("native Flash expert geometry differs")
        for key, shape in shapes.items():
            experts._parameters[key] = nn.Parameter(torch.empty(shape, dtype=torch.bfloat16, device=device), requires_grad=False)
        prefix = f"{PREFIX}layers.{layer}.mlp.experts."
        assignments = [(self.weight_map[prefix + f"{e}.{part}.weight"], e, part)
                       for e in range(count) for part in ("gate_proj", "up_proj", "down_proj")]
        try:
            for _, expert, part in sorted(assignments):
                target = (experts.down_proj[expert] if part == "down_proj" else
                          experts.gate_up_proj[expert, :inter] if part == "gate_proj" else experts.gate_up_proj[expert, inter:])
                value = self.tensor(prefix + f"{expert}.{part}.weight")
                if value.dtype != torch.bfloat16 or value.shape != target.shape:
                    raise ValueError("native Flash expert dtype/shape differs")
                target.copy_(value)
            block._gptqmodel_glm5_experts_complete = True
        finally:
            self.close()
        return block

    def load_block(self, layer, device="cuda:0", *, include_experts=False):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
        config = self.native_config()
        if isinstance(layer, bool) or not 0 <= layer < config.num_hidden_layers:
            raise ValueError("Flash target layer outside schedule")
        install_bounded_kda()
        with torch.device("meta"):
            block = Glm5NextTextDecoderLayer(config, layer)
        return self.materialize_block(block, layer, device, include_experts=include_experts)

    def rotary(self, device="cuda:0"):
        return None  # This native MLA has zero rotary channels.

    @staticmethod
    def module(block, relative_name):
        return module(block, relative_name)


@torch.inference_mode()
def initial_hidden(source, ids, device="cuda:0"):
    embedding = source.tensor(PREFIX + "embed_tokens.weight", device)
    hidden = torch.nn.functional.embedding(ids.to(device), embedding)
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    return hidden.unsqueeze(2).expand(-1, -1, source.config["hc_mult"], -1).contiguous()


def prepare_context(hidden, rotary=None, prev_topk=None, *, position_ids=None, attention_mask=None):
    if hidden.ndim not in (3, 4):
        raise ValueError("Flash context requires native token or mHC streams")
    batch, length = hidden.shape[:2]
    if position_ids is None:
        position_ids = torch.arange(length, device=hidden.device).unsqueeze(0)
    if position_ids.dtype != torch.int64 or tuple(position_ids.shape) not in ((1, length), (batch, length)):
        raise ValueError("Flash positions differ from prompt")
    if attention_mask is None:
        attention_mask = torch.ones((batch, length), dtype=torch.bool, device=hidden.device)
    if tuple(attention_mask.shape) != (batch, length):
        raise ValueError("Flash reference requires an unpadded 2D native mask")
    return {"attention_mask": attention_mask, "position_ids": position_ids,
                "prev_topk_indices": prev_topk, "position_embeddings": None}


@torch.inference_mode()
def attention_inputs(block, hidden):
    collapsed = block.attn_hc(hidden)[2] if hasattr(block, "attn_hc") else hidden
    return block.input_layernorm(collapsed)


@torch.inference_mode()
def attention_forward(block, normalized, context):
    if getattr(block, "block_type", None) == "linear_attention":
        value = block.self_attn(normalized, cache_params=None, attention_mask=context["attention_mask"])
        return value, None
    value, _, topk = block.self_attn(normalized, past_key_values=None, **context)
    return value, topk


def _expand(hc, residual, output):
    post, comb, _ = hc(residual)
    return post.to(residual.dtype).unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(
        comb.to(residual.dtype).transpose(-1, -2), residual)


@torch.inference_mode()
def attention_finish(block, residual, output):
    if hasattr(block, "attn_hc"):
        streams = _expand(block.attn_hc, residual, output)
        return streams, block.post_attention_layernorm(block.ffn_hc(streams)[2])
    residual = residual + output
    return residual, block.post_attention_layernorm(residual)


@torch.inference_mode()
def shared_down_input(shared, hidden):
    gate = shared.gate_proj(hidden).clamp(max=shared.swiglu_limit)
    up = shared.up_proj(hidden).clamp(min=-shared.swiglu_limit, max=shared.swiglu_limit)
    return shared.act_fn(gate) * up


@torch.inference_mode()
def finish_layer(block, residual, normalized):
    output = block.mlp(normalized)
    return _expand(block.ffn_hc, residual, output) if hasattr(block, "ffn_hc") else residual + output


@torch.inference_mode()
def native_forward(block, state):
    context = prepare_context(state["hidden"], position_ids=state.get("position_ids"),
                              prev_topk=state.get("prev_topk_indices"))
    hidden, topk = block(state["hidden"], past_key_values=None, use_cache=False, **context)
    out = {"hidden": hidden}
    if topk is not None:
        out["prev_topk_indices"] = topk
    return out


@torch.inference_mode()
def staged_forward(block, state):
    context = prepare_context(state["hidden"], position_ids=state.get("position_ids"),
                              prev_topk=state.get("prev_topk_indices"))
    attention, topk = attention_forward(block, attention_inputs(block, state["hidden"]), context)
    residual, normalized = attention_finish(block, state["hidden"], attention)
    out = {"hidden": finish_layer(block, residual, normalized)}
    if topk is not None:
        out["prev_topk_indices"] = topk
    return out
