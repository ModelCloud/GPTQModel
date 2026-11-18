# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Raon1MiniQModel(BaseQModel):
    """KRAFTON Raon-1 mini hybrid RDA/MLA model with routed MoE layers."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.raon1_mini.configuration_raon1_mini.Raon1MiniConfig",
        model_class="gptqmodel.models.ports.raon1_mini.modeling_raon1_mini.Raon1MiniForCausalLM",
        tokenizer_class="gptqmodel.models.ports.raon1_mini.tokenization_raon1_mini.Raon1MiniTokenizer",
    )
    dynamic_expert_index = "num_experts"
    pre_lm_head_norm_module = "model.norm"
    layer_modules_strict = False
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # RDA and MLA layers have different projection names. Raon-mHC control
    # modules are intentionally left in full precision.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn_raon_mhc": ("self_attn_raon_mhc:!", "res:!"),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "q_a_proj:0",
                "q_b_proj:0",
                "kv_a_proj_with_mqa:0",
                "kv_b_proj:0",
                "g_proj:0",
                "b_proj:0",
                "rda_w_proj:0",
                "f_a_proj:0",
                "f_b_proj:0",
                "o_proj:1",
                "q_a_layernorm:!",
                "kv_a_layernorm:!",
                "o_norm:!",
                "q_conv1d:!",
                "k_conv1d:!",
                "v_conv1d:!",
                "A_log:!",
                "dt_bias:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp_raon_mhc": ("mlp_raon_mhc:!", "res:!"),
            "mlp": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
            "moe:moe": {
                "gate": ("gate:!", "e_score_correction_bias:!"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Only the two decoder norms are fold sources. SiTU-GLU bounds `up_proj`
        # with tanh, so `up_proj -> down_proj` is not neutral, and `o_proj` reads
        # the gated o_norm (RDA) or gated attention context (MLA), never the
        # input norm the generic planner falls back to.
        sources = (module.input_layernorm, module.post_attention_layernorm)
        o_proj = module.self_attn.o_proj
        nodes = []
        for node in super().awq_get_modules_for_scaling(module, input_feat, module_kwargs):
            if not any(node["prev_op"] is source for source in sources):
                continue
            layers = [layer for layer in node["layers"] if layer is not o_proj]
            if layers:
                nodes.append({**node, "layers": layers})
        return nodes

    def awq_module_replay_inputs(self, module, x, module_kwargs):
        x, module_kwargs = super().awq_module_replay_inputs(module, x, module_kwargs)
        # RDA/MLA take the decoder's `cache` and `layer_idx` positionally; the
        # layer capture records neither. Replay runs without a cache.
        for index, layer in enumerate(self.model.model.layers):
            if module is getattr(layer, "self_attn", None):
                return x, {**module_kwargs, "cache": None, "layer_idx": index}
        return x, module_kwargs

    def quantize(self, *args, **kwargs):
        if self.model.config.raon_mxfp8_config:
            raise NotImplementedError(
                "Raon MXFP8 source re-quantization is not qualified; use a dense Raon checkpoint."
            )
        return super().quantize(*args, **kwargs)

    def pre_quantize(self, module):
        if self.model.config.raon_mxfp8_config:
            raise NotImplementedError("Raon MXFP8 source re-quantization is not qualified.")
        return super().pre_quantize(module)

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        # Let the base hook consume the captured per-layer-type mask/rotary state
        # before Raon's cache-specific kwargs are applied.
        additional_inputs = super().prepare_layer_replay_kwargs(layer, layer_input, additional_inputs, target_device)
        # Raon decoder layers take `cache`, not the Transformers `use_cache` flag.
        additional_inputs.pop("use_cache", None)
        additional_inputs.setdefault("cache", None)
        return additional_inputs


__all__ = ["Raon1MiniQModel"]
