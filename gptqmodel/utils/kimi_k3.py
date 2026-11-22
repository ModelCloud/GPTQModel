# SPDX-License-Identifier: Apache-2.0
"""Bounded Kimi K3 calibration against the package-owned K3 decoder port.

Only the routed w1/w3/w2 matrices are quantized. Attention, latent projections,
shared experts, SiTU and attention-residual state retain source semantics.
The ordinary whole-model GPTQ looper cannot carry K3's residual bank; callers
must use an explicit layer boundary with both prefix_sum and block_residual.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


def load_calibration_reference(snapshot=None, trust_remote_code: bool = False):
    """Return the package-owned K3 decoder (``KimiLinearConfig``/``KimiDecoderLayer``).

    Calibration no longer executes Python from the checkpoint snapshot. The
    arguments are accepted for backward compatibility and ignored.
    """
    del snapshot, trust_remote_code
    from ..models.ports.kimi_k3 import modeling_kimi_linear

    return modeling_kimi_linear


def fit_checkpoint_parameter(key, tensor, parameter):
    """Validate a checkpoint tensor against a block parameter; drop zero KDA ``A_log`` padding."""
    if tuple(tensor.shape) == tuple(parameter.shape):
        return tensor
    if key.endswith("self_attn.A_log") and tensor.ndim == 1 and parameter.ndim == 1:
        from ..models.ports.kimi_k3.checkpoint import ActiveKDAHeads

        return next(iter(ActiveKDAHeads(parameter.numel()).convert({key: tensor}, [key], [key]).values()))
    raise ValueError(f"checkpoint parameter geometry differs: {key}")


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
    # ``pow(2, exponent)`` rather than ``ldexp``: ``torch.ldexp`` launches on
    # the *current* device and faults with an illegal memory access when its
    # inputs live on another GPU (#5263). The exponent is widened to float32
    # so the process default dtype cannot turn most scales into 0/inf (#5297).
    scale = torch.pow(2.0, (scales.int() - 127).to(torch.float32))
    values.mul_(scale.unsqueeze(-1))
    result = values.reshape(packed.shape[0], -1).to(dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite decoded MXFP4 weight")
    return result


class KimiK3Source:
    """Read one tensor at a time; never materialize the 2.8T model in host RAM."""
    def __init__(self, snapshot, *, trust_remote_code: bool = False):
        self.snapshot = Path(snapshot).resolve()
        self.config = json.loads((self.snapshot / "config.json").read_text())
        if self.config.get("model_type") != "kimi_k3":
            raise ValueError("expected kimi_k3 checkpoint")
        self.text_config = self.config["text_config"]
        self.weight_map = json.loads((self.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        for shard in set(self.weight_map.values()):
            if Path(shard).name != shard or not (self.snapshot / shard).is_file():
                raise ValueError(f"invalid/missing K3 shard: {shard}")
        self.reference = load_calibration_reference()

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
        # Checkpoint KDA A_log is zero-padded to 128 heads; the port keeps the
        # 96 active heads, exactly as its checkpoint loader does.
        prefix = f"language_model.model.layers.{layer}."
        for key, parameter in list(block.named_parameters()):
            tensor = fit_checkpoint_parameter(prefix + key, self.decoded(prefix + key, device), parameter)
            parent, _, leaf = key.rpartition(".")
            target = block.get_submodule(parent) if parent else block
            target._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
        if any(t.is_meta for t in (*block.parameters(), *block.buffers())):
            raise ValueError("K3 block has unmaterialized state")
        return block.eval()


def attn_res_function(block):
    """The decoder module's own ``_apply_attn_res``, independent of forward wrappers."""
    return getattr(sys.modules[type(block).__module__], "_apply_attn_res")


@torch.no_grad()
def prepare_moe(block, prefix_sum, block_residual):
    """Exactly the reference decoder's attention half, before routed MoE."""
    hidden = prefix_sum
    b, length, width = hidden.shape
    if block_residual.shape[1]:
        hidden = attn_res_function(block)(
            prefix_sum.view(-1, width), block_residual,
            block.self_attention_res_proj, block.self_attention_res_norm).view(b, length, width)
    if block.layer_idx % block.attn_res_block_size == 0:
        block_residual = torch.cat((block_residual, prefix_sum.view(-1, width).unsqueeze(1)), 1)
        prefix_sum = None
    hidden = block.input_layernorm(hidden)
    attention = block.self_attn(hidden_states=hidden, attention_mask=None, use_cache=False)
    prefix_sum = attention if prefix_sum is None else prefix_sum + attention
    hidden = attn_res_function(block)(
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
