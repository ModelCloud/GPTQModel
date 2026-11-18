# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class BaiChuanQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.baichuan.configuration_baichuan.BaichuanConfig",
        model_class="gptqmodel.models.ports.baichuan.modeling_baichuan.BaichuanForCausalLM",
        tokenizer_class="gptqmodel.models.ports.baichuan.tokenization_baichuan.BaichuanTokenizer",
    )
    fusable = None  # Fused QKV; no qualified fusion executor.
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("W_pack:0:q:k:v", "o_proj:1", "rotary_emb:!"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # `o_proj` reads the attention output and the packed 3x-wide W_pack
        # output can never fold into it, so it has no scale source. ALiBi layers
        # own no `rotary_emb`, which leaves the generic planner's last-module
        # fallback on `input_layernorm` (same width) and merges `o_proj` into the
        # norm -> W_pack group. Keep every other group and quantize `o_proj`
        # plainly.
        o_proj = getattr(getattr(module, "self_attn", None), "o_proj", None)
        nodes = []
        for node in super().awq_get_modules_for_scaling(module, input_feat, module_kwargs):
            layers = [layer for layer in node.get("layers") or [] if layer is not o_proj]
            if layers:
                node["layers"] = layers
                nodes.append(node)
        return nodes
