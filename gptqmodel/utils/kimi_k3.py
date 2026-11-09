# SPDX-License-Identifier: Apache-2.0
"""Bounded Kimi K3 calibration against the checkpoint's own decoder code.

Only the routed w1/w3/w2 matrices are quantized. Attention, latent projections,
shared experts, SiTU and attention-residual state retain source semantics.
The ordinary whole-model GPTQ looper cannot carry K3's residual bank; callers
must use an explicit layer boundary with both prefix_sum and block_residual.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


def load_calibration_reference(snapshot):
    """Load unmodified decoder math, excluding unused HF whole-model wrappers.

    K3's bundled decoder imports HF4's removed OutputRecorder/check_model_inputs
    solely for its top-level model wrapper. Excluding those wrappers lets HF5
    execute the same decoder without global monkey patches or checkpoint edits.
    """
    snapshot = Path(snapshot).resolve()
    name = "_gptqmodel_kimi_k3_" + hashlib.sha256(str(snapshot).encode()).hexdigest()[:16]
    if name + ".modeling_kimi_linear" in sys.modules:
        return sys.modules[name + ".modeling_kimi_linear"]
    package = types.ModuleType(name)
    package.__path__ = [str(snapshot)]
    sys.modules[name] = package
    config_name = name + ".configuration_kimi_k3"
    spec = importlib.util.spec_from_file_location(config_name, snapshot / "configuration_kimi_k3.py")
    config = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config
    spec.loader.exec_module(config)
    path = snapshot / "modeling_kimi_linear.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    wrappers = {"KimiPreTrainedModel", "KimiLinearModel", "KimiLinearForCausalLM"}
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ClassDef) and node.name in wrappers)
                 and not (isinstance(node, ast.ImportFrom) and node.module == "transformers.utils.generic")]
    module = types.ModuleType(name + ".modeling_kimi_linear")
    module.__package__ = name
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[module.__name__]
        raise
    return module


@torch.no_grad()
def decode_mxfp4(packed, scales, *, dtype=torch.bfloat16):
    """OCP E2M1, interleaved low/high nibbles, E8M0 scales per 32 inputs."""
    if (packed.dtype != torch.uint8 or scales.dtype != torch.uint8
            or packed.ndim != 2 or tuple(scales.shape) != (packed.shape[0], packed.shape[1] // 16)
            or packed.shape[1] % 16):
        raise ValueError("invalid K3 MXFP4 packed/scales geometry")
    if bool((scales == 255).any()):
        raise ValueError("NaN MXFP4 E8M0 scale")
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                       device=packed.device, dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1)
    values = lut[codes.long()].reshape(packed.shape[0], -1, 32)
    scale = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)
    values.mul_(scale.unsqueeze(-1))
    result = values.reshape(packed.shape[0], -1).to(dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite decoded MXFP4 weight")
    return result


class KimiK3Source:
    """Read one tensor at a time; never materialize the 2.8T model in host RAM."""
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).resolve()
        self.config = json.loads((self.snapshot / "config.json").read_text())
        if self.config.get("model_type") != "kimi_k3":
            raise ValueError("expected kimi_k3 checkpoint")
        self.text_config = self.config["text_config"]
        self.weight_map = json.loads((self.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        for shard in set(self.weight_map.values()):
            if Path(shard).name != shard or not (self.snapshot / shard).is_file():
                raise ValueError(f"invalid/missing K3 shard: {shard}")
        self.reference = load_calibration_reference(self.snapshot)

    def tensor(self, name, device="cpu"):
        with safe_open(self.snapshot / self.weight_map[name], framework="pt", device="cpu") as file:
            # Queue GPU copies from owned host memory, never a file mapping.
            owned = file.get_tensor(name).clone()
        return owned.to(device=device)

    def decoded(self, name, device="cpu"):
        if name in self.weight_map:
            return self.tensor(name, device)
        prefix = name.removesuffix(".weight")
        if not name.endswith(".weight") or prefix + ".weight_packed" not in self.weight_map:
            raise KeyError(name)
        return decode_mxfp4(self.tensor(prefix + ".weight_packed", device),
                            self.tensor(prefix + ".weight_scale", device))

    @torch.no_grad()
    def load_block(self, layer, device="cuda:0"):
        if not 0 <= layer < self.text_config["num_hidden_layers"]:
            raise ValueError("layer outside K3 architecture")
        config = self.reference.KimiLinearConfig.from_dict(self.text_config)
        config._attn_implementation = "sdpa"
        config.use_cache = False
        with torch.device("meta"):
            block = self.reference.KimiDecoderLayer(config, layer)
        # Checkpoint KDA A_log is padded to 128 heads; kernels read only the
        # 96 active heads. Preserve those source bytes rather than trimming it.
        prefix = f"language_model.model.layers.{layer}."
        for key, parameter in list(block.named_parameters()):
            tensor = self.decoded(prefix + key, device)
            if tuple(tensor.shape) != tuple(parameter.shape):
                if not (key == "self_attn.A_log" and tensor.ndim == 1
                        and tensor.numel() >= parameter.numel()):
                    raise ValueError(f"checkpoint parameter geometry differs: {prefix + key}")
            parent, _, leaf = key.rpartition(".")
            target = block.get_submodule(parent) if parent else block
            target._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
        if any(t.is_meta for t in (*block.parameters(), *block.buffers())):
            raise ValueError("K3 block has unmaterialized state")
        return block.eval()


@torch.no_grad()
def prepare_moe(block, prefix_sum, block_residual):
    """Exactly the reference decoder's attention half, before routed MoE."""
    hidden = prefix_sum
    b, length, width = hidden.shape
    if block_residual.shape[1]:
        hidden = block.forward.__globals__["_apply_attn_res"](
            prefix_sum.view(-1, width), block_residual,
            block.self_attention_res_proj, block.self_attention_res_norm).view(b, length, width)
    if block.layer_idx % block.attn_res_block_size == 0:
        block_residual = torch.cat((block_residual, prefix_sum.view(-1, width).unsqueeze(1)), 1)
        prefix_sum = None
    hidden = block.input_layernorm(hidden)
    attention = block.self_attn(hidden_states=hidden, attention_mask=None, use_cache=False)
    prefix_sum = attention if prefix_sum is None else prefix_sum + attention
    hidden = block.forward.__globals__["_apply_attn_res"](
        prefix_sum.view(-1, width), block_residual,
        block.mlp_res_proj, block.mlp_res_norm).view(b, length, width)
    hidden = block.post_attention_layernorm(hidden)
    return prefix_sum, block_residual, hidden


@torch.no_grad()
def finish_moe(prefix_sum, block_residual, moe_output):
    return prefix_sum + moe_output, block_residual


def situ(gate, up, beta=4., linear_beta=25.):
    output_dtype = gate.dtype
    gate, up = gate.float(), up.float()
    activation = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (activation * up).to(output_dtype)
