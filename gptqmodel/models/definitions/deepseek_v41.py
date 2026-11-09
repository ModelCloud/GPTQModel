# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 calibration primitives, distinct from V4's layer contract.

V4.1 carries both the mHC pre-mix and shared CSA2 state between blocks.
The production loader/looper must bind those before this family is registered
in the automatic quantization dispatcher.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41DecoderLayer, DeepseekV41RotaryEmbedding

from ...utils.ple_mmap import MappedPLETable
from ..base import BaseQModel


class DeepSeekV41RotaryEmbedding(DeepseekV41RotaryEmbedding):
    """Keep the small rotary coefficients FP32, matching complex reference RoPE."""

    def forward(self, x, position_ids, layer_type=None):
        if self.rope_type[layer_type] not in ("default", "yarn"):
            raise ValueError("V4.1 calibration supports the checkpoint's default/YaRN RoPE")
        inv = getattr(self, f"{layer_type}_inv_freq").to(x.device).float()
        angles = position_ids.float().unsqueeze(-1) * inv
        coeff = torch.polar(torch.ones_like(angles), angles)
        factor = getattr(self, f"{layer_type}_attention_scaling")
        return coeff.real * factor, coeff.imag * factor


class DeepSeekV41HyperConnection(nn.Module):
    def __init__(self, source, kernels):
        super().__init__()
        self.fn, self.base, self.scale = source.fn, source.base, source.scale
        self.hc_mult = source.hc_mult
        self.hc_sinkhorn_iters = source.hc_sinkhorn_iters
        self.hc_eps = source.hc_eps
        self.norm_eps = source.input_norm.eps
        self.kernels = kernels

    def forward(self, hidden):
        flat = hidden.flatten(2).float()
        inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.fn) * inverse_rms
        return self.kernels.hc_split_sinkhorn(mixes, self.scale, self.base,
                                              self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)


class DeepSeekV41DecoderLayer(DeepseekV41DecoderLayer):
    @staticmethod
    def hc_expand(x, residual, post, comb):
        # Match the checkpoint's multiply-then-reduce order. A matrix multiply
        # changes intermediate rounding before the final BF16 store.
        mixed = (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
        return (post.unsqueeze(-1) * x.unsqueeze(-2) + mixed).to(x.dtype)


class DeepSeekV41RMSNorm(nn.Module):
    """Weight multiplication precedes BF16 rounding in the checkpoint reference."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, hidden):
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * value).to(hidden.dtype)


class DeepSeekV41MappedEmbedding(nn.Module):
    """Parameter-free PLE gather: file pages can be reclaimed independently."""

    def __init__(self, path, prefix, *, scale_path=None, max_gather_rows=65536):
        super().__init__()
        self.table = MappedPLETable(path, prefix, scale_path=scale_path, max_gather_rows=max_gather_rows)
        self.num_embeddings = self.table.rows
        self.embedding_dim = self.table.width

    @torch.no_grad()
    def forward(self, ids):
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("PLE indices must be int32 or int64")
        flat = ids.detach().reshape(-1).cpu()
        output = torch.empty((*ids.shape, self.embedding_dim), device=ids.device,
                             dtype=torch.float32)
        out_flat = output.reshape(-1, self.embedding_dim)
        chunk = self.table.max_gather_rows
        for start in range(0, flat.numel(), chunk):
            indices = flat[start:start + chunk].tolist()
            self.table.prefetch(indices)
            raw_weight, raw_scale = self.table.gather(indices)
            # Owned bytearrays avoid read-only-buffer alias warnings; neither tensor
            # can retain a reference to the full memory mapping.
            weight = torch.frombuffer(bytearray(raw_weight), dtype=torch.float8_e4m3fn)
            scale = torch.frombuffer(bytearray(raw_scale), dtype=torch.float8_e8m0fnu)
            values = weight.float().reshape(-1, self.embedding_dim // 32, 32)
            values.mul_(scale.float().reshape(-1, self.embedding_dim // 32, 1))
            out_flat[start:start + len(indices)].copy_(values.reshape(-1, self.embedding_dim))
        self.table.release()
        return output

    def close(self):
        self.table.close()


class DeepSeekV41Expert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, limit, *, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.limit = limit

    def forward(self, hidden, route_weights):
        gate, up = self.gate_proj(hidden).float(), self.up_proj(hidden).float()
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        # Routing belongs BEFORE the down projection, including its dtype rounding.
        activated = F.silu(gate) * up * route_weights.unsqueeze(-1)
        return self.down_proj(activated.to(hidden.dtype))


class DeepSeekV41Experts(nn.ModuleList):
    """Unfused routed experts with the reference's token order and FP32 sum."""

    @classmethod
    def from_fused(cls, fused):
        if fused.gate_up_proj.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("decode packed source experts before unfusing")
        experts = cls()
        for index in range(fused.num_experts):
            expert = DeepSeekV41Expert(fused.hidden_dim, fused.intermediate_dim,
                                      fused.limit, device="meta", dtype=fused.gate_up_proj.dtype)
            gate, up = fused.gate_up_proj[index].split(fused.intermediate_dim, dim=0)
            # Clone releases the fused allocation once the old container is removed.
            expert.gate_proj.weight = nn.Parameter(gate.clone(), requires_grad=False)
            expert.up_proj.weight = nn.Parameter(up.clone(), requires_grad=False)
            expert.down_proj.weight = nn.Parameter(fused.down_proj[index].clone(), requires_grad=False)
            experts.append(expert)
        return experts

    def forward(self, hidden, indices, weights):
        result = torch.zeros_like(hidden, dtype=torch.float32)
        for index, expert in enumerate(self):
            token, slot = torch.where(indices == index)
            if token.numel():
                value = expert(hidden[token], weights[token, slot])
                result.index_add_(0, token, value.float())
        return result


class DeepSeekV41QModel(BaseQModel):
    """V4.1 definition; registration awaits the stateful calibration adapter."""

    require_trust_remote_code = False
    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.norm"
    layer_modules_strict = False
    module_tree = ["model", "layers", "#", {
        "mlp:moe": {"experts": {"#": ("gate_proj:0", "up_proj:0", "down_proj:1")}},
    }]
    out_of_model_tensors = {"prefixes": ["mtp", "vision", "aligner"],
                            "tensors": ["image_start", "image_end", "image_newline", "image_pad"]}

    @staticmethod
    def convert_model_structure(model, cleanup_original=False):
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Experts

        converted = False
        for block in model.model.layers:
            if isinstance(block.mlp.experts, DeepseekV41Experts):
                block.mlp.experts = DeepSeekV41Experts.from_fused(block.mlp.experts)
                converted = True
        return converted
