# SPDX-License-Identifier: Apache-2.0
"""Bounded native/EXL3 source reads and calibrated Kimi K3 core projections.

EXL3 uses fixed 128-channel Hadamard blocks. Logical output widths such as
MLA's 576 channels are zero-padded in storage, then cropped only after the
ordinary serialized weight has been fully reconstructed. No custom packer,
Hadamard convention, or required auxiliary checkpoint metadata is introduced.
"""
from __future__ import annotations

import math
import re

import torch
from torch import nn

from .kimi_k3 import KimiK3Source


_HESSIAN_CONTRACT = "signed-block-hadamard-congruence-fp64-v1"


def _quantizer():
    # Importing the source/header helper must not initialize a CUDA extension.
    from ..exllamav3.modules.quant.exl3_lib import quantize
    return quantize


def validate_core_tensors(packed, logical_shape):
    """Validate ordinary MCG storage against native [output, input] geometry."""
    if (not isinstance(logical_shape, (tuple, list)) or len(logical_shape) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in logical_shape)):
        raise ValueError("invalid native EXL3 weight shape")
    outputs, inputs = logical_shape
    stored_outputs = math.ceil(outputs / 128) * 128
    if inputs % 128 or set(packed) != {"trellis", "suh", "svh", "mcg"}:
        raise ValueError("expected 128-aligned input and standard EXL3 MCG tensors")
    trellis, suh, svh, mcg = (packed[k] for k in ("trellis", "suh", "svh", "mcg"))
    if (trellis.dtype != torch.int16 or trellis.ndim != 3
            or tuple(trellis.shape[:2]) != (inputs // 16, stored_outputs // 16)
            or trellis.shape[2] not in range(16, 129, 16)
            or suh.dtype != torch.float16 or tuple(suh.shape) != (inputs,)
            or svh.dtype != torch.float16 or tuple(svh.shape) != (stored_outputs,)
            or mcg.dtype != torch.int32 or tuple(mcg.shape) not in ((), (1,))):
        raise ValueError("EXL3 stored geometry/dtypes differ from the native projection")
    if int(mcg.reshape(()).view(torch.uint32).item()) != 0xCBAC1FED:
        raise ValueError("unsupported EXL3 MCG multiplier")
    if not bool(torch.isfinite(suh).all()) or not bool(torch.isfinite(svh).all()):
        raise ValueError("nonfinite EXL3 scales")
    if outputs != stored_outputs and bool(torch.count_nonzero(svh[outputs:])):
        raise ValueError("EXL3 padded output channels must have zero scales")
    return {"native_shape": [outputs, inputs], "stored_shape": [stored_outputs, inputs],
                "bits": int(trellis.shape[-1] // 16), "output_padding": stored_outputs - outputs}


@torch.no_grad()
def reconstruct_core_projection(packed, logical_shape, *, device="cuda:0", dtype=torch.bfloat16):
    """Reconstruct serialized FP16 scales/trellis, then crop native output rows."""
    validate_core_tensors(packed, logical_shape)
    weight = _quantizer().reconstruct_exl3_tensors(packed, device=device, dtype=dtype)
    return weight.T[:logical_shape[0], :logical_shape[1]].contiguous()


@torch.no_grad()
def quantize_core_projection(weight, raw_hessian, *, bits=6, seed=787, sigma=.025, device="cuda:0"):
    """Quantize a dense native [output,input] matrix using a fresh raw XTX copy.

    ``raw_hessian`` is the usual H/count/finalized mapping. The corpus count
    is required: no identity augmentation or uncalibrated fallback is accepted.
    The caller owns raw_hessian and weight; neither is mutated.
    """
    if (isinstance(bits, bool) or not isinstance(bits, int) or bits != 6
            or isinstance(seed, bool) or not isinstance(seed, int)
            or isinstance(sigma, bool) or not isinstance(sigma, (int, float))
            or not math.isfinite(sigma) or sigma <= 0):
        raise ValueError("Kimi core requires K6, an integer seed, and positive finite sigma")
    if (weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or weight.shape[0] <= 0 or weight.shape[1] % 128):
        raise ValueError("invalid dense Kimi core projection geometry/dtype")
    if (not isinstance(raw_hessian, dict) or raw_hessian.get("finalized") is not False
            or isinstance(raw_hessian.get("count"), bool)
            or not isinstance(raw_hessian.get("count"), int) or raw_hessian["count"] <= 0):
        raise ValueError("calibration requires a fresh raw Hessian and positive sample count")
    H = raw_hessian.get("H")
    if (not isinstance(H, torch.Tensor) or H.dtype != torch.float32
            or tuple(H.shape) != (weight.shape[1], weight.shape[1])
            or not bool(torch.isfinite(H).all()) or float(H.diagonal().sum()) <= 0):
        raise ValueError("invalid raw calibration Hessian")
    for chunk in weight.split(4096, dim=0):
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("nonfinite core projection")
    native_shape = list(weight.shape)
    outputs, inputs = native_shape
    stored_outputs = math.ceil(outputs / 128) * 128
    # Transpose to the quantizer's [input,output] convention. Keep the source
    # device here: the large-head path controls its own CPU working buffers.
    working_weight = torch.zeros((inputs, stored_outputs), device=weight.device, dtype=torch.float32)
    working_weight[:, :outputs].copy_(weight.T)
    target = torch.device(device)
    working_hessian = {"H": H.to(device=target, copy=True), "count": raw_hessian["count"], "finalized": False}
    args = {"K": bits, "seed": seed, "sigma_reg": float(sigma), "devices": [target],
                "mcg": True, "apply_out_scales": None}
    _, _, packed = _quantizer().quantize_exl3(
        working_weight, working_hessian, args, return_weight_q=False)
    metrics = args.get("error_metrics", {})
    if (metrics.get("quantizer_path") != "hessian_ldlq"
            or metrics.get("hessian_metric_status") != "ok"
            or metrics.get("hessian_numerical_contract") != _HESSIAN_CONTRACT
            or metrics.get("hessian_transform_compute_dtype") != "torch.float64"
            or metrics.get("hessian_storage_dtype") != "torch.float32"
            or metrics.get("hessian_sample_count") != raw_hessian["count"]):
        raise ValueError("core projection did not use calibrated FP64-congruence LDLQ")
    for key in ("reported_metric_value", "hessian_weighted_error_numerator",
                "hessian_weighted_reference_denominator", "hessian_weighted_relative_error"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid calibrated core metric: {key}")
    packed = {key: value.detach().cpu() for key, value in packed.items()}
    evidence = validate_core_tensors(packed, native_shape)
    if evidence["bits"] != bits:
        raise ValueError("core quantizer returned a different bitrate")
    evidence.update(hessian_rows=raw_hessian["count"], identity_rows=0,
                    replay="serialized-exl3-fp16-scales-reconstructed-then-native-output-crop")
    return packed, metrics, evidence


class KimiK3EXL3Source(KimiK3Source):
    """K2/K6 source with one-tensor reads and deferred routed expert storage."""

    def close(self):
        context = getattr(self, "_reader_context", None)
        self._reader_context = self._reader = self._reader_shard = None
        if context is not None:
            context.__exit__(None, None, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def tensor(self, name, device="cpu"):
        """Parse one active shard once; transfer only owned tensor bytes.

        Four EXL3 buffers per projection otherwise reopen the same layer shard
        over ten thousand times. A single bounded reader also avoids retaining
        thousands of separate mapped storages until a Python GC collection.
        """
        from safetensors import safe_open
        shard = self.weight_map[name]
        if getattr(self, "_reader_shard", None) != shard:
            self.close()
            context = safe_open(self.snapshot / shard, framework="pt", device="cpu")
            reader = context.__enter__()
            self._reader_context, self._reader, self._reader_shard = context, reader, shard
        owned = self._reader.get_tensor(name).clone()
        return owned.to(device=device)

    def _logical_shape(self, prefix):
        cfg = self.text_config
        h, intermediate = cfg["hidden_size"], cfg["moe_intermediate_size"]
        latent = cfg["routed_expert_hidden_size"]
        match = re.fullmatch(r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])", prefix)
        if match:
            layer, expert, projection = match.groups()
            if not 1 <= int(layer) < cfg["num_hidden_layers"] or not 0 <= int(expert) < cfg["num_experts"]:
                raise ValueError("routed EXL3 tensor is outside the Kimi architecture")
            return (latent, intermediate) if projection == "w2" else (intermediate, latent)
        if prefix == "language_model.lm_head":
            return cfg["vocab_size"], h
        match = re.fullmatch(r"language_model\.model\.layers\.(\d+)\.(.+)", prefix)
        if not match or not 0 <= int(match[1]) < cfg["num_hidden_layers"]:
            raise ValueError(f"unsupported Kimi EXL3 projection: {prefix}")
        leaf = match[2]
        width = cfg["linear_attn_config"]["num_heads"] * cfg["linear_attn_config"]["head_dim"]
        shared = intermediate * cfg["num_shared_experts"]
        shapes = {
            **{f"self_attn.{name}": (width, h) for name in ("q_proj", "k_proj", "v_proj", "g_proj")},
            "self_attn.o_proj": (h, width),
            "self_attn.q_a_proj": (cfg["q_lora_rank"], h),
            "self_attn.q_b_proj": (cfg["num_attention_heads"] * (cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]), cfg["q_lora_rank"]),
            "self_attn.kv_a_proj_with_mqa": (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"], h),
            "block_sparse_moe.routed_expert_down_proj": (latent, h),
            "block_sparse_moe.routed_expert_up_proj": (h, latent),
            "block_sparse_moe.shared_experts.gate_proj": (shared, h),
            "block_sparse_moe.shared_experts.up_proj": (shared, h),
            "block_sparse_moe.shared_experts.down_proj": (h, shared),
        }
        if leaf not in shapes:
            raise ValueError(f"unsupported Kimi EXL3 projection: {prefix}")
        return shapes[leaf]

    def decoded(self, name, device="cpu"):
        if name in self.weight_map:
            return self.tensor(name, device)
        prefix = name.removesuffix(".weight")
        if not name.endswith(".weight") or prefix + ".trellis" not in self.weight_map:
            raise KeyError(name)
        suffixes = {"trellis", "suh", "svh", "mcg"}
        if (any(prefix + "." + suffix not in self.weight_map for suffix in suffixes)
                or any(prefix + "." + suffix in self.weight_map
                       for suffix in ("su", "sv", "mul1", "weight_packed", "weight_scale", "bias"))):
            raise ValueError(f"unsupported EXL3 companions: {prefix}")
        # Packed tensors are tiny relative to their reconstructed weight and
        # each call owns copies, with no safetensors mapping surviving it.
        packed = {suffix: self.tensor(prefix + "." + suffix, device) for suffix in suffixes}
        return reconstruct_core_projection(packed, self._logical_shape(prefix), device=device)

    @torch.no_grad()
    def load_block(self, layer, device="cuda:0", *, include_experts=False):
        if not 0 <= layer < self.text_config["num_hidden_layers"]:
            raise ValueError("layer outside K3 architecture")
        config = self.reference.KimiLinearConfig.from_dict(self.text_config)
        config._attn_implementation = "sdpa"
        config.use_cache = False
        with torch.device("meta"):
            block = self.reference.KimiDecoderLayer(config, layer)
        self._materialize(block, layer, device, experts_only=False, include_experts=include_experts)
        return block.eval()

    def _materialize(self, block, layer, device, *, experts_only, include_experts):
        prefix = f"language_model.model.layers.{layer}."
        for key, parameter in list(block.named_parameters()):
            routed = key.startswith("block_sparse_moe.experts.")
            if (experts_only and (not routed or not parameter.is_meta)) or (not include_experts and routed):
                continue
            tensor = self.decoded(prefix + key, device)
            if tuple(tensor.shape) != tuple(parameter.shape):
                if not (key == "self_attn.A_log" and tensor.ndim == 1 and tensor.numel() >= parameter.numel()):
                    raise ValueError(f"checkpoint parameter geometry differs: {prefix + key}")
            parent, _, leaf = key.rpartition(".")
            target = block.get_submodule(parent) if parent else block
            target._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
        for key, value in (*block.named_parameters(), *block.named_buffers()):
            if value.is_meta and (include_experts or not key.startswith("block_sparse_moe.experts.")):
                raise ValueError(f"K3 block has unmaterialized state: {key}")

    @torch.no_grad()
    def materialize_routed_experts(self, block, device="cuda:0"):
        """Load exactly the base's serialized K2 experts after attention work."""
        self._materialize(block, block.layer_idx, device, experts_only=True, include_experts=True)
        return block
