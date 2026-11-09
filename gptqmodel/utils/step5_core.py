# SPDX-License-Identifier: Apache-2.0
"""Bounded Step-5 BF16 source and checkpoint-native calibration stages.

Attention executes the pinned cuteafd Step-5 reference's existing bounded
wrappers, which call the vendor's CSA/indexer/sparse kernels and windowed SDPA.
This adapter adds loading, observation and stage boundaries, never attention math.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from . import glm5_core as base


PREFIX = "model."
REVISION = "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
capture_linear_input = base.capture_linear_input
reconstruct_projection = base.reconstruct_projection
validate_projection = base.validate_projection
HESSIAN_ACCUMULATION_CONTRACT = "step5-native-fp32-XTX-k128-v1"


@torch.inference_mode()
def accumulate_hessian_fp32(H, inputs, *, max_rows=128):
    """Add every actual activation row with a bounded FP32 reduction length.

    A long transposed-input CUDA SGEMM can accumulate thousands of products
    sequentially, destroying the covariance's positive margin. Bound each
    product's K dimension to 128 while retaining FP32 storage and corpus order.
    No padding, synthetic rows, damping or precision fallback is introduced.
    Return the number of actual rows accumulated.
    """
    if (not isinstance(H, torch.Tensor) or H.ndim != 2 or H.shape[0] != H.shape[1]
            or H.shape[0] <= 0 or H.dtype != torch.float32):
        raise ValueError("Step-5 Hessian accumulation requires square FP32 storage")
    if (isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 128):
        raise ValueError("Step-5 Hessian products require 1..128 actual rows")
    if (not isinstance(inputs, torch.Tensor) or inputs.ndim < 2
            or inputs.shape[-1] != H.shape[0] or not inputs.is_floating_point()):
        raise ValueError("invalid Step-5 Hessian activation geometry or dtype")
    rows = inputs.reshape(-1, H.shape[0])
    if not len(rows) or not bool(torch.isfinite(rows).all()):
        raise ValueError("Step-5 Hessian observations must be nonempty and finite")
    for start in range(0, len(rows), max_rows):
        value = rows[start:start + max_rows].to(device=H.device, dtype=torch.float32)
        H.addmm_(value.T, value)
    if not bool(torch.isfinite(H).all()):
        raise ValueError("nonfinite Step-5 Hessian accumulation")
    return len(rows)


class DenseRawHessian(base.DenseRawHessian):
    """Actual dense observations with Step-5's bounded FP32 XTX products."""
    @torch.inference_mode()
    def add(self, inputs):
        self.count += accumulate_hessian_fp32(self.H, inputs, max_rows=min(self.chunk_rows, 128))

    update = add



@torch.no_grad()
def quantize_projection(weight, raw_hessian, *, bits=6, seed=787, sigma=.025, device="cuda:0"):
    """Quantize native [output,input] storage with an owned raw FP32 XTX copy.

    No identity rows are invented here. If routed recovery supplied such rows,
    its declared ``identity_rows`` count is retained in the returned evidence.
    Dense callers omit that key and therefore declare zero synthetic rows.
    """
    if (isinstance(bits, bool) or bits not in (4, 5, 6) or not isinstance(bits, int)
            or isinstance(seed, bool) or not isinstance(seed, int)
            or isinstance(sigma, bool) or not isinstance(sigma, (int, float))
            or not math.isfinite(sigma) or sigma <= 0):
        raise ValueError("Step-5 projection requires K4/K5/K6, integer seed and positive finite sigma")
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
    _, _, packed = base._quantizer().quantize_exl3(working, raw, args, return_weight_q=False)
    metrics = args.get("error_metrics", {})
    required = {"quantizer_path": "hessian_ldlq", "hessian_metric_status": "ok",
        "hessian_numerical_contract": "signed-block-hadamard-congruence-fp64-v1", "hessian_transform_compute_dtype": "torch.float64",
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
    evidence = base.validate_projection(packed, shape)
    if evidence["bits"] != bits:
        raise ValueError("quantizer returned a different bitrate")
    evidence.update(hessian_rows=raw_hessian["count"], identity_rows=identity_rows,
                    replay="serialized-exl3-fp16-scales-reconstructed-then-native-output-crop")
    return packed, metrics, evidence


def _load_reference(root):
    root = Path(root).resolve()
    path = root / "families/step5/golden.py"
    if not path.is_file():
        raise ValueError("Step-5 quantization requires the pinned target reference root")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("gptqmodel_step5_target_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step5BF16Source(base.GlmBF16Source):
    """One mapping at a time; fused expert reads own only one expert matrix."""
    def __init__(self, snapshot, *, reference_root):
        self.snapshot = Path(snapshot).resolve()
        self.document = json.loads((self.snapshot / "config.json").read_text())
        self.config = self.document["text_config"]
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_text())
        self.weight_map = self.index["weight_map"]
        self._reader = self._reader_context = self._reader_shard = None
        if (self.document.get("model_type") != "step5_vl"
                or self.document.get("quantization_config")
                or self.config.get("model_type") != "step5"
                or self.config.get("num_hidden_layers") != 92
                or self.config.get("num_nextn_predict_layers") != 3
                or self.config.get("norm_dtype") != "float32"
                or self.config.get("fp32_residual_connection") is not True):
            raise ValueError("expected the reviewed native BF16 Step-5 checkpoint")
        self.reference = _load_reference(reference_root)
        from shape_invariant import install
        install()  # Exactly the fixed-M128 reference GEMM geometry.
        self.source_identity = self.reference.verify_source(self.snapshot)
        self.release, self.kernel = self.reference.load_release(self.snapshot)
        self._config = self.release.Step5Config(**copy.deepcopy(self.reference.decoder_config(self.document)))
        self.reference.validate_config(self._config)
        self._rope = {}

    def native_config(self):
        return self._config

    def expert_tensor(self, layer, expert, part, device="cpu"):
        if (isinstance(layer, bool) or not isinstance(layer, int)
                or layer not in self.config["moe_layer_list"]
                or isinstance(expert, bool) or not isinstance(expert, int)
                or not 0 <= expert < self.config["moe_num_experts"]
                or part not in ("gate_proj", "up_proj", "down_proj")):
            raise ValueError("Step-5 expert slice outside checkpoint geometry")
        name = f"model.layers.{layer}.moe.{part}.weight"
        owned = self._active_reader(name).get_slice(name)[expert].clone()
        if owned.ndim != 2 or owned.dtype != torch.bfloat16:
            raise ValueError("Step-5 fused expert must contain native BF16 matrices")
        return owned.to(device=device)

    @torch.no_grad()
    def materialize_block(self, block, layer, device="cuda:0", *, include_experts=False):
        prefix = f"model.layers.{layer}."
        state = block.state_dict()
        expert_keys = {f"moe.{part}.weight" for part in ("gate_proj", "up_proj", "down_proj")}
        for name in state:
            if prefix + name not in self.weight_map:
                raise ValueError(f"native Step-5 leaf missing from checkpoint: {prefix + name}")
        try:
            for key in sorted(state, key=lambda k: (self.weight_map[prefix + k], k)):
                if key in expert_keys:
                    continue
                value = self.tensor(prefix + key, device)
                if value.shape != state[key].shape or value.dtype != state[key].dtype:
                    raise ValueError(f"native Step-5 dtype/shape differs: {prefix + key}")
                owner_name, _, leaf = key.rpartition(".")
                owner = block.get_submodule(owner_name) if owner_name else block
                if leaf in owner._buffers:
                    owner._buffers[leaf] = value
                else:
                    owner._parameters[leaf] = nn.Parameter(value, requires_grad=False)
            block._gptqmodel_step5_source = self
            block._gptqmodel_step5_extent = 0
            if include_experts:
                self.materialize_routed_experts(block, layer, device)
            for name, value in (*block.named_parameters(), *block.named_buffers()):
                if value.is_meta and value.numel() and name not in expert_keys:
                    raise ValueError(f"Step-5 parameter remains meta: {name}")
        finally:
            self.close()
        return block.eval()

    def load_block(self, layer, device="cuda:0", *, include_experts=False):
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < 95:
            raise ValueError("Step-5 decoder layer outside target/MTP schedule")
        with torch.device("meta"):
            block = self.release.Step5DecoderLayer(self.native_config(), layer, dtype=torch.bfloat16)
            if layer >= self.config["num_hidden_layers"]:
                width, vocab, eps = (self.config[k] for k in ("hidden_size", "vocab_size", "rms_norm_eps"))
                block.enorm = self.release.Step5RMSNorm(width, eps)
                block.hnorm = self.release.Step5RMSNorm(width, eps)
                block.eh_proj = nn.Linear(width * 2, width, bias=False, dtype=torch.bfloat16)
                block.transformer = nn.Module()
                block.transformer.shared_head = nn.Module()
                block.transformer.shared_head.norm = self.release.Step5RMSNorm(width, eps)
                block.transformer.shared_head.output = nn.Linear(width, vocab, bias=False, dtype=torch.bfloat16)
        return self.materialize_block(block, layer, device, include_experts=include_experts)

    @torch.no_grad()
    def materialize_routed_experts(self, block, layer=None, device="cuda:0"):
        if block.moe is None or getattr(block, "_gptqmodel_step5_experts_complete", False):
            return block
        layer = block.layer_idx if layer is None else layer
        try:
            for part in ("gate_proj", "up_proj", "down_proj"):
                leaf = getattr(block.moe, part)
                shape = tuple(leaf.weight.shape)
                leaf.weight = nn.Parameter(torch.empty(shape, device=device, dtype=torch.bfloat16), requires_grad=False)
                for expert in range(shape[0]):
                    leaf.weight[expert].copy_(self.expert_tensor(layer, expert, part, device))
            block._gptqmodel_step5_experts_complete = True
        finally:
            self.close()
        return block

    def rotary(self, device="cuda:0"):
        return None  # The release owns layer-specific rotary tables.

    @staticmethod
    def module(block, relative_name):
        return module(block, relative_name)


def module(block, relative_name):
    return block.get_submodule(relative_name.removesuffix(".weight"))


def replace_projection(block, relative_name, weight):
    """Invalidate a packed QKV view before replacing one checkpoint matrix."""
    name = relative_name.removesuffix(".weight")
    if name.startswith("self_attn.") and name.split(".")[-1] in ("q_proj", "k_proj", "v_proj"):
        block.self_attn._qkv_weight = torch.empty(0, device=weight.device, dtype=weight.dtype)
    target = module(block, name)
    if target.weight.shape != weight.shape:
        raise ValueError("replacement projection shape differs")
    target.weight = nn.Parameter(weight.detach(), requires_grad=False)


@torch.inference_mode()
def initial_hidden(source, ids, device="cuda:0"):
    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if ids.ndim != 1 or ids.dtype != torch.int64:
        raise ValueError("Step-5 requires one unpadded int64 prompt")
    embedding = source.tensor("model.embed_tokens.weight", device)
    return torch.nn.functional.embedding(ids.to(device), embedding).float()


def prepare_context(hidden, rotary=None, prev_topk=None, *, position_ids=None, attention_mask=None):
    if hidden.ndim != 2 or hidden.dtype not in (torch.float32, torch.bfloat16) or not len(hidden):
        raise ValueError("Step-5 context requires nonempty native [tokens,hidden] operands")
    if rotary is not None or prev_topk is not None or attention_mask is not None:
        raise ValueError("Step-5 reference accepts unpadded prompts with layer-owned RoPE")
    if position_ids is None:
        position_ids = torch.arange(len(hidden), device=hidden.device, dtype=torch.int64)
    if position_ids.ndim == 2 and position_ids.shape[0] == 1:
        position_ids = position_ids[0]
    if (position_ids.shape != (len(hidden),) or position_ids.dtype not in (torch.int32, torch.int64)
            or position_ids.device != hidden.device or bool((position_ids < 0).any())
            or bool((position_ids[1:] != position_ids[:-1] + 1).any())):
        raise ValueError("Step-5 positions must be consecutive and native")
    return {"position_ids": position_ids.to(torch.int32), "extent": ((int(position_ids[-1]) + 128) // 128) * 128}


def _prepare_rope(block, context):
    source = block._gptqmodel_step5_source
    extent = context["extent"]
    if block._gptqmodel_step5_extent < extent:
        # The release builder fills empty tables only; retire shorter tables.
        for name in ("rope_cos", "rope_sin", "sparse_indexer_rope_cos", "sparse_indexer_rope_sin"):
            if hasattr(block.self_attn, name):
                setattr(block.self_attn, name, torch.empty(0, device=context["position_ids"].device))
        block.self_attn.build_rope(max_position=extent, device=context["position_ids"].device,
                                  shared_cache=source._rope)
        block._gptqmodel_step5_extent = extent
    return source


@torch.inference_mode()
def attention_inputs(block, hidden):
    return block.input_layernorm(hidden).to(block.self_attn.q_proj.weight.dtype)


@torch.inference_mode()
def attention_forward(block, normalized, context):
    source = _prepare_rope(block, context)
    attention = block.self_attn
    positions = context["position_ids"]
    query, key, value = attention._project(normalized, positions)
    if attention.is_sparse:
        if int(positions[0]) != 0:
            raise ValueError("native sparse calibration requires the complete prompt")
        output = source.reference.sparse_attention(attention, source.kernel, query, key, value,
                                                   normalized, positions, context["extent"])
    else:
        output = source.reference.sliding_attention(query, key, value, attention.sliding_window)
    return attention._gate_and_project(output, normalized), None


@torch.inference_mode()
def attention_finish(block, residual, output):
    residual = residual + output.float()
    return residual, block.post_attention_layernorm(residual).to(output.dtype)


@torch.inference_mode()
def shared_down_input(shared, hidden):
    # Call the release's clamped-SwiGLU, retaining its F32 activation/product.
    release_name = type(shared).__module__
    release = sys.modules.get(release_name)
    if release is None:
        raise ValueError("native Step-5 MLP module is unavailable")
    return release.clamped_swiglu(shared.gate_proj(hidden), shared.up_proj(hidden), shared.limit)


@torch.inference_mode()
def finish_layer(block, residual, normalized):
    return residual + block._feed_forward(normalized).float()


@torch.inference_mode()
def native_forward(block, state):
    context = prepare_context(state["hidden"], position_ids=state.get("position_ids"))
    source = _prepare_rope(block, context)
    if int(context["position_ids"][0]) != 0:
        # Shifted MTP is sliding attention, and uses the same stage arithmetic.
        return staged_forward(block, state)
    hidden = source.reference.layer_forward(block, source.kernel, state["hidden"], context["extent"])
    return {"hidden": hidden, "position_ids": context["position_ids"].long()}


@torch.inference_mode()
def staged_forward(block, state):
    context = prepare_context(state["hidden"], position_ids=state.get("position_ids"))
    attention, _ = attention_forward(block, attention_inputs(block, state["hidden"]), context)
    residual, normalized = attention_finish(block, state["hidden"], attention)
    return {"hidden": finish_layer(block, residual, normalized), "position_ids": context["position_ids"].long()}


@contextmanager
def capture_inputs(block, names, callback):
    """Observe full checkpoint module names, including packed functional QKV.

    ``callback(full_name, input_tensor)`` receives the actual GEMM operands.
    Packed QKV and sparse-indexer W bypass their child Linear forwards.
    Observe their enclosing native calls without additional matrix products;
    no operator, GEMM dtype, output or reduction order is replaced.
    """
    prefix = f"model.layers.{block.layer_idx}."
    relative = {name.removeprefix(prefix).removesuffix(".weight") for name in names}
    hooks = []
    qkv = {f"self_attn.{part}_proj" for part in "qkv"}
    observed_qkv = relative & qkv
    functional_indexer = "self_attn.sparse_indexer_w"
    observed_indexer = functional_indexer in relative
    original = original_indexer = None
    try:
        for name in sorted(relative - qkv - {functional_indexer}):
            def observe(target, args, name=name):
                callback(prefix + name, args[0])
            hooks.append(module(block, name).register_forward_pre_hook(observe))
        if observed_qkv:
            original = block.self_attn._project_packed_qkv
            def packed(self, hidden):
                for name in sorted(observed_qkv):
                    callback(prefix + name, hidden)
                return original(hidden)
            block.self_attn._project_packed_qkv = types.MethodType(packed, block.self_attn)
        if observed_indexer:
            indexer = block.self_attn.indexer
            if indexer is None:
                raise ValueError("Step-5 functional indexer observation requires a sparse layer")
            original_indexer = indexer.project
            def project(self, hidden, positions):
                # The native function rounds its F32 W weight to hidden.dtype
                # and uses this exact hidden operand in functional.linear.
                callback(prefix + functional_indexer, hidden)
                return original_indexer(hidden, positions)
            indexer.project = types.MethodType(project, indexer)
        yield
    finally:
        for hook in hooks:
            hook.remove()
        if original is not None:
            block.self_attn._project_packed_qkv = original
        if original_indexer is not None:
            block.self_attn.indexer.project = original_indexer


def router_choices(moe, hidden, *, rank_max=16):
    """Actual top-k followed by deterministic lower-rank recovery candidates."""
    if rank_max < moe.top_k or rank_max > moe.gate.weight.shape[0]:
        raise ValueError("Step-5 router rank range differs")
    weights, selected = moe.route(hidden)
    logits = torch.nn.functional.linear(hidden.float(), moe.gate.weight.float())
    scores = torch.sigmoid(logits) + moe.router_bias
    # Native topk chooses tied rows differently; retain its exact first slots.
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, selected, False)
    remainder = scores.masked_fill(~mask, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    return selected, weights, remainder[:, :rank_max - moe.top_k]


def invalidate_attention(block):
    """Retire the packed view after the caller replaces Q/K/V parameters."""
    weight = block.self_attn.q_proj.weight
    block.self_attn._qkv_weight = torch.empty(0, dtype=weight.dtype, device=weight.device)


def expert_weight(block, expert, part):
    if (block.moe is None or part not in ("gate_proj", "up_proj", "down_proj")
            or isinstance(expert, bool) or not isinstance(expert, int)
            or not 0 <= expert < block.moe.n_local_experts):
        raise ValueError("Step-5 native expert address differs")
    return getattr(block.moe, part).weight[expert]


@torch.inference_mode()
def head_input(source, hidden):
    """The target narrows its final residual before zero-centered RMSNorm."""
    norm = source.release.Step5RMSNorm(source.config["hidden_size"], source.config["rms_norm_eps"])
    norm.weight = nn.Parameter(source.tensor("model.norm.weight", hidden.device), requires_grad=False)
    return norm(hidden.bfloat16())


@torch.inference_mode()
def mtp_envelope(block, embedding, previous_hidden):
    if embedding.ndim != 2 or embedding.shape != previous_hidden.shape:
        raise ValueError("MTP requires aligned token embeddings and previous residual rows")
    return torch.cat((block.enorm(embedding.bfloat16()), block.hnorm(previous_hidden.bfloat16())), -1)


@torch.inference_mode()
def mtp_head_input(block, hidden):
    return block.transformer.shared_head.norm(hidden.bfloat16())


@torch.inference_mode()
def aligned_inputs(previous_hidden, token_ids, embedding, *, position_ids=None):
    if (previous_hidden.ndim != 2 or previous_hidden.dtype != torch.float32
            or token_ids.ndim != 1 or token_ids.dtype != torch.int64
            or token_ids.shape[0] != len(previous_hidden) or len(token_ids) < 2):
        raise ValueError("MTP requires a complete aligned teacher-forced prompt")
    context = prepare_context(previous_hidden, position_ids=position_ids)
    shifted = token_ids[1:].contiguous()
    embedded = torch.nn.functional.embedding(shifted, embedding) if isinstance(embedding, torch.Tensor) else embedding(shifted)
    return {"target_hidden": previous_hidden[:-1].contiguous(), "inputs_embeds": embedded,
                "input_ids": shifted, "position_ids": context["position_ids"][1:].long()}


@torch.inference_mode()
def forward_aligned(block, aligned):
    hidden = block.eh_proj(mtp_envelope(block, aligned["inputs_embeds"], aligned["target_hidden"])).float()
    result = staged_forward(block, {"hidden": hidden, "position_ids": aligned["position_ids"]})
    result["normalized_hidden"] = mtp_head_input(block, result["hidden"])
    return result


def load_mtp(source, stage, device="cuda:0"):
    if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage < 3:
        raise ValueError("Step-5 MTP stage outside checkpoint schedule")
    return source.load_block(source.config["num_hidden_layers"] + stage, device)


@torch.inference_mode()
def expert_down_input(block, expert, hidden):
    """Actual Step-5 FP32 clamped activation/product, narrowed once to BF16."""
    if block.moe is None:
        raise ValueError("Step-5 expert input requires a routed layer")
    return block._gptqmodel_step5_source.release.clamped_swiglu(
        block.moe.gate_proj.apply_expert(expert, hidden),
        block.moe.up_proj.apply_expert(expert, hidden), block.moe.limit)
