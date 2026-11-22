# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Phi3QModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    shared_input_verified_model_types = frozenset({"phi3"})

    pre_lm_head_norm_module = "model.norm"

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
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    fusable = FusableConfig("phimoe").merge(LlamaFusableConfig(), profiles=("qkv",))

    dynamic_expert_index = "num_local_experts"
    pre_lm_head_norm_module = "model.norm"

    # PhiMoE stores experts as fused 3-D tensors; Defuser expands them into
    # per-expert ``gate_proj``/``up_proj``/``down_proj`` linears before the
    # module tree is resolved. Without these hooks ``ExpertsRoutingBypass``
    # fell back to native top-2 routing with only a per-forward warning.
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

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
