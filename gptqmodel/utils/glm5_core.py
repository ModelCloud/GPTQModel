# SPDX-License-Identifier: Apache-2.0
"""Bounded GLM MLA/DSA source loading and calibrated EXL3 core stages.

Every returned checkpoint tensor owns its storage. Only one safetensors reader
is retained; routed experts stay on meta until their current layer needs them.
Calibration orchestration and recovery policy remain with the caller.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from .exl3_router_candidates import bind_sigmoid_grouped_router_recovery, learned_router_ranked_choices
from .kimi_k3_core_format import reconstruct_core_projection, validate_core_tensors
from .kimi_k3_core_stream import DenseRawHessian, capture_linear_input


_HESSIAN_CONTRACT = "signed-block-hadamard-congruence-fp64-v1"


def _quantizer():
    # Header-only tools must not build a CUDA extension at import time.
    from ..exllamav3.modules.quant.exl3_lib import quantize
    return quantize


@torch.no_grad()
def quantize_projection(weight, raw_hessian, *, bits=6, seed=787, sigma=.025, device="cuda:0"):
    """Quantize native [output,input] storage with an owned raw FP32 XTX copy.

    No identity rows are invented here. If routed recovery supplied such rows,
    its declared ``identity_rows`` count is retained in the returned evidence.
    Dense callers omit that key and therefore declare zero synthetic rows.
    """
    if (isinstance(bits, bool) or bits not in (4, 6) or not isinstance(bits, int)
            or isinstance(seed, bool) or not isinstance(seed, int)
            or isinstance(sigma, bool) or not isinstance(sigma, (int, float))
            or not math.isfinite(sigma) or sigma <= 0):
        raise ValueError("GLM projection requires K4/K6, integer seed and positive finite sigma")
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or min(weight.shape) <= 0 or weight.shape[1] % 128):
        raise ValueError("invalid native GLM projection geometry or dtype")
    if (not isinstance(raw_hessian, dict) or raw_hessian.get("finalized") is not False
            or isinstance(raw_hessian.get("count"), bool)
            or not isinstance(raw_hessian.get("count"), int) or raw_hessian["count"] <= 0):
        raise ValueError("calibration requires a fresh raw Hessian and positive sample count")
    identity_rows = raw_hessian.get("identity_rows", 0)
    if (isinstance(identity_rows, bool) or not isinstance(identity_rows, int)
            or not 0 <= identity_rows <= raw_hessian["count"]):
        raise ValueError("invalid declared synthetic row count")
    H = raw_hessian.get("H")
    if (not isinstance(H, torch.Tensor) or H.dtype != torch.float32
            or tuple(H.shape) != (weight.shape[1], weight.shape[1])
            or not bool(torch.isfinite(H).all()) or float(H.diagonal().sum()) <= 0):
        raise ValueError("invalid raw calibration Hessian")
    for chunk in weight.split(4096, dim=0):
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("nonfinite GLM projection")
    shape = list(weight.shape)
    outputs, inputs = shape
    stored_outputs = math.ceil(outputs / 128) * 128
    working = torch.zeros((inputs, stored_outputs), device=weight.device, dtype=torch.float32)
    working[:, :outputs].copy_(weight.T)
    target = torch.device(device)
    raw = {"H": H.to(device=target, copy=True), "count": raw_hessian["count"], "finalized": False}
    args = {"K": bits, "seed": seed, "sigma_reg": float(sigma), "devices": [target], "mcg": True, "apply_out_scales": None}
    _, _, packed = _quantizer().quantize_exl3(working, raw, args, return_weight_q=False)
    metrics = args.get("error_metrics", {})
    required = {"quantizer_path": "hessian_ldlq", "hessian_metric_status": "ok",
        "hessian_numerical_contract": _HESSIAN_CONTRACT, "hessian_transform_compute_dtype": "torch.float64",
        "hessian_storage_dtype": "torch.float32", "hessian_sample_count": raw_hessian["count"]}
    if any(metrics.get(key) != value for key, value in required.items()):
        raise ValueError("projection did not use calibrated FP64-congruence LDLQ")
    for key in ("reported_metric_value", "hessian_weighted_error_numerator",
                "hessian_weighted_reference_denominator", "hessian_weighted_relative_error"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid calibrated projection metric: {key}")
    if metrics["hessian_weighted_reference_denominator"] <= 0:
        raise ValueError("calibrated reference energy must be positive")
    packed = {key: value.detach().cpu() for key, value in packed.items()}
    evidence = validate_core_tensors(packed, shape)
    if evidence["bits"] != bits:
        raise ValueError("quantizer returned a different bitrate")
    evidence.update(hessian_rows=raw_hessian["count"], identity_rows=identity_rows,
                    replay="serialized-exl3-fp16-scales-reconstructed-then-native-output-crop")
    return packed, metrics, evidence


reconstruct_projection = reconstruct_core_projection
validate_projection = validate_core_tensors


class GlmBF16Source:
    """Owned tensor reads with one active shard and deferred native experts."""
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).resolve()
        self.config = json.loads((self.snapshot / "config.json").read_text())
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_text())
        self.weight_map = self.index["weight_map"]
        if self.config.get("model_type") != "glm_moe_dsa" or self.config.get("quantization_config"):
            raise ValueError("expected a native BF16 GLM MLA/DSA checkpoint")
        self._reader = self._reader_context = self._reader_shard = None

    def close(self):
        context = self._reader_context
        self._reader = self._reader_context = self._reader_shard = None
        if context is not None:
            context.__exit__(None, None, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _active_reader(self, name):
        from safetensors import safe_open
        shard = self.weight_map[name]
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("checkpoint shard must be a local safetensors basename")
        if shard != self._reader_shard:
            self.close()
            self._reader_context = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
            self._reader = self._reader_context.__enter__()
            self._reader_shard = shard
        return self._reader

    def tensor(self, name, device="cpu"):
        # Never hand a file mapping directly to CUDA. The synchronous copy
        # retains at most one matrix's owned CPU transfer buffer.
        owned = self._active_reader(name).get_tensor(name).clone()
        return owned.to(device=device)

    def tensor_header(self, name):
        value = self._active_reader(name).get_slice(name)
        return value.get_dtype(), value.get_shape()

    def native_config(self):
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        config = GlmMoeDsaConfig.from_dict(self.config)
        config._attn_implementation = "sdpa"
        config._experts_implementation = "eager"
        config.use_cache = False
        return config

    @torch.no_grad()
    def materialize_block(self, block, layer, device="cuda:0", *, include_experts=False):
        """Load only this native block's state; extra MTP leaves stay with its adapter."""
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError("invalid source layer index")
        prefix = f"model.layers.{layer}."
        state = block.state_dict()
        assignments = []
        for key, value in state.items():
            if key.startswith("mlp.experts.") or not value.is_meta:
                continue
            name = prefix + key
            assignments.append((self.weight_map[name], name, key))
        try:
            for _, name, key in sorted(assignments):
                tensor = self.tensor(name, device)
                if tensor.shape != state[key].shape:
                    raise ValueError(f"native state shape differs: {name}")
                if key.endswith("e_score_correction_bias"):
                    if tensor.dtype != torch.float32:
                        raise ValueError("native router correction bias must remain FP32")
                elif tensor.dtype != torch.bfloat16:
                    raise ValueError(f"native GLM state must be BF16: {name}")
                if key == "self_attn.indexer.weights_proj.weight":
                    tensor = tensor.float()
                parent, _, leaf = key.rpartition(".")
                module = block.get_submodule(parent) if parent else block
                if leaf in module._buffers:
                    module._buffers[leaf] = tensor
                else:
                    module._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
            block.layer_idx = layer
            if hasattr(block.mlp, "gate"):
                bind_sigmoid_grouped_router_recovery(block.mlp.gate)
            if include_experts and hasattr(block.mlp, "experts"):
                self.materialize_routed_experts(block, layer=layer, device=device)
            for key, value in (*block.named_parameters(), *block.named_buffers()):
                if value.is_meta and (include_experts or not key.startswith("mlp.experts.")):
                    raise ValueError(f"native GLM state remains meta: {key}")
        finally:
            self.close()
        return block.eval()

    @torch.no_grad()
    def load_block(self, layer, device="cuda:0", *, include_experts=False):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
        config = self.native_config()
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < config.num_hidden_layers:
            raise ValueError("layer outside native GLM target schedule")
        with torch.device("meta"):
            block = GlmMoeDsaDecoderLayer(config, layer)
        return self.materialize_block(block, layer, device=device, include_experts=include_experts)

    @torch.no_grad()
    def materialize_routed_experts(self, block, layer=None, device="cuda:0"):
        if getattr(block, "_gptqmodel_glm5_experts_complete", False):
            return block
        layer = getattr(block, "layer_idx", None) if layer is None else layer
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("routed materialization requires a source layer index")
        experts = block.mlp.experts
        expected = {"gate_up_proj", "down_proj"}
        if set(experts.state_dict()) != expected:
            raise ValueError("native GLM expert storage differs")
        shapes = {key: tuple(value.shape) for key, value in experts.state_dict().items()}
        count, doubled_inter, width = shapes["gate_up_proj"]
        inter = doubled_inter // 2
        if doubled_inter % 2 or shapes["down_proj"] != (count, width, inter):
            raise ValueError("native fused expert geometry differs")
        block._gptqmodel_glm5_experts_complete = False
        experts.gate_up_proj = nn.Parameter(torch.empty(shapes["gate_up_proj"], dtype=torch.bfloat16,
            device=device), requires_grad=False)
        experts.down_proj = nn.Parameter(torch.empty(shapes["down_proj"], dtype=torch.bfloat16,
            device=device), requires_grad=False)
        prefix = f"model.layers.{layer}.mlp.experts."
        assignments = []
        for expert in range(count):
            for part in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}{expert}.{part}.weight"
                assignments.append((self.weight_map[name], name, expert, part))
        try:
            for _, name, expert, part in sorted(assignments):
                if part == "down_proj":
                    target = experts.down_proj[expert]
                else:
                    start = 0 if part == "gate_proj" else inter
                    target = experts.gate_up_proj[expert, start:start+inter]
                tensor = self.tensor(name)
                if tensor.dtype != torch.bfloat16 or tensor.shape != target.shape:
                    raise ValueError(f"source expert geometry or dtype differs: {name}")
                target.copy_(tensor)
            block._gptqmodel_glm5_experts_complete = True
        finally:
            self.close()
        return block

    def rotary(self, device="cuda:0"):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding
        return GlmMoeDsaRotaryEmbedding(self.native_config()).to(device)


@torch.inference_mode()
def prepare_context(hidden, rotary, prev_topk=None, *, position_ids=None, attention_mask=None):
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or min(hidden.shape) <= 0:
        raise ValueError("GLM context requires nonempty [batch,sequence,hidden] state")
    batch, length, _ = hidden.shape
    if position_ids is None:
        position_ids = torch.arange(length, device=hidden.device).unsqueeze(0)
    if (not isinstance(position_ids, torch.Tensor) or position_ids.dtype != torch.int64
            or tuple(position_ids.shape) not in ((1, length), (batch, length))
            or bool((position_ids < 0).any())):
        raise ValueError("invalid explicit GLM position IDs")
    position_ids = position_ids.to(hidden.device)
    if attention_mask is None:
        attention_mask = torch.ones((1, 1, length, length), device=hidden.device, dtype=torch.bool).tril()
    if (not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 4
            or attention_mask.shape[0] not in (1, batch)
            or tuple(attention_mask.shape[1:]) != (1, length, length)
            or not (attention_mask.dtype == torch.bool or attention_mask.is_floating_point())):
        raise ValueError("invalid explicit GLM causal mask")
    if prev_topk is not None:
        if (not isinstance(prev_topk, torch.Tensor) or prev_topk.ndim != 3
                or tuple(prev_topk.shape[:2]) != (batch, length)
                or prev_topk.dtype not in (torch.int32, torch.int64)):
            raise ValueError("invalid exact shared DSA top-k state")
        prev_topk = prev_topk.to(hidden.device)
    return {"attention_mask": attention_mask.to(hidden.device), "position_ids": position_ids,
                "position_embeddings": rotary(hidden, position_ids), "prev_topk_indices": prev_topk}


@torch.inference_mode()
def attention_inputs(block, hidden):
    return block.input_layernorm(hidden)


attention_entry = attention_inputs


@torch.inference_mode()
def attention_forward(block, normalized, context):
    output, _, topk = block.self_attn(normalized, past_key_values=None, **context)
    return output, topk


@torch.inference_mode()
def attention_finish(block, residual, attention_output):
    residual = residual + attention_output
    return residual, block.post_attention_layernorm(residual)


@torch.inference_mode()
def shared_down_input(shared, hidden):
    return shared.act_fn(shared.gate_proj(hidden)) * shared.up_proj(hidden)


@torch.inference_mode()
def router_choices(moe, hidden, *, rank_max=16):
    logits, weights, selected = moe.gate(hidden)
    if (selected.dtype != torch.int64 or selected.ndim != 2 or selected.shape != weights.shape
            or selected.shape[1] != 8):
        raise ValueError("GLM production recovery requires native top-8 router output")
    if not isinstance(rank_max, int) or isinstance(rank_max, bool) or rank_max != 16:
        raise ValueError("GLM recovery requires corrected router ranks 9 through 16")
    bind_sigmoid_grouped_router_recovery(moe.gate)
    _, ranked = learned_router_ranked_choices(moe.gate, logits, rank_max=rank_max,
                                               selected_indices=selected)
    return selected, weights, ranked[:, 8:16]


@torch.inference_mode()
def complete_moe(residual, normalized, moe):
    """Native BF16 expert reduction, shared add, then the outer residual add."""
    _, weights, selected = moe.gate(normalized)
    shape = normalized.shape
    routed = moe.experts(normalized.reshape(-1, shape[-1]), selected, weights).view(shape)
    return residual + (routed + moe.shared_experts(normalized))


__all__ = ["GlmBF16Source", "DenseRawHessian", "capture_linear_input", "quantize_projection",
           "reconstruct_projection", "validate_projection", "prepare_context", "attention_inputs",
           "attention_entry", "attention_forward", "attention_finish", "shared_down_input",
           "router_choices", "complete_moe"]
