# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from torch import nn

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class Phi3QModel(BaseQModel):
    shared_input_verified_model_types = frozenset({"phi3"})

    pre_lm_head_norm_module = "model.norm"

    @classmethod
    def after_defuser_conversion(cls, model):
        """Split a fused `Phi3MLP.gate_up_proj` left by remote-code checkpoints.

        Phi-3 128k longrope checkpoints load through their repository's remote
        `modeling_phi3.py`, whose `Phi3MLP` class path is not the installed
        `transformers.models.phi3.modeling_phi3.Phi3MLP` that Defuser registers.
        Defuser therefore leaves the fused projection in place while the
        definition declares `gate_proj`/`up_proj`; with the default strict plan
        quantization raises. Split the fused weight with the same layout Defuser
        produces for the native class so both loaders resolve the same tree.
        """

        for module in list(model.modules()):
            gate_up = getattr(module, "gate_up_proj", None)
            if not isinstance(gate_up, nn.Linear):
                continue
            if module.__class__.__name__ != "Phi3MLP" or not hasattr(module, "down_proj"):
                continue
            if gate_up.out_features % 2:
                continue

            half = gate_up.out_features // 2
            weight = gate_up.weight
            bias = gate_up.bias
            gate_proj = nn.Linear(
                gate_up.in_features,
                half,
                bias=bias is not None,
                device=weight.device,
                dtype=weight.dtype,
            )
            up_proj = nn.Linear(
                gate_up.in_features,
                half,
                bias=bias is not None,
                device=weight.device,
                dtype=weight.dtype,
            )
            gate_weight, up_weight = weight.chunk(2, dim=0)
            gate_proj.weight = nn.Parameter(gate_weight.clone(), requires_grad=False)
            up_proj.weight = nn.Parameter(up_weight.clone(), requires_grad=False)
            if bias is not None:
                gate_bias, up_bias = bias.chunk(2, dim=0)
                gate_proj.bias = nn.Parameter(gate_bias.clone(), requires_grad=False)
                up_proj.bias = nn.Parameter(up_bias.clone(), requires_grad=False)

            module.gate_proj = gate_proj
            module.up_proj = up_proj
            del module.gate_up_proj

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("qkv_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
        }
    ]

class PhiMoEGPTQForCausalLM(BaseQModel):
    fusable = FusableConfig("phimoe").merge(LlamaFusableConfig(), profiles=("qkv",))

    dynamic_expert_index = "num_local_experts"
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp|block_sparse_moe:moe:?": {
                "router": ("router:!",),  # PhimoeTopKRouter.forward() returns two values, skipping its quantization.
                "experts": {
                    "#": ("gate_proj|w1:0", "up_proj|w3:0", "down_proj|w2:1"),
                },
            },
        }
    ]

__all__ = ["Phi3QModel", "PhiMoEGPTQForCausalLM"]
