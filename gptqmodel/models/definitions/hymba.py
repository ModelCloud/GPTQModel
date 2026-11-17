# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class HymbaQModel(BaseQModel):
    supports_desc_act = [False]
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.hymba.configuration_hymba.HymbaConfig",
        model_class="gptqmodel.models.ports.hymba.modeling_hymba.HymbaForCausalLM",
    )
    write_shared_kv_cache = True
    dynamic_expert_index = "num_experts"
    layer_modules_strict = False  # optional affine Q/K norms, relu2, and no-MLP layouts

    pre_lm_head_norm_module = "model.final_layernorm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # The conv/SSM path between `in_proj` and `out_proj` is not
            # homogeneous, so `in_proj` must never become a fold source:
            # folding a scale into it would scale that path's output. Same
            # guard as Falcon-H1's mixer (#4674). The mixer's dense leaves are
            # declared `:!` so the fold source resolution cannot skip over
            # them and land on `in_proj`.
            "mamba": (
                "in_proj:0:nosrc", "out_proj:1:nosrc", "conv1d:!", "x_proj.0:!", "dt_proj.0:!",
                "A_log:!", "D:!", "dt_layernorm:!", "B_layernorm:!", "C_layernorm:!",
                "pre_avg_layernorm1:!", "pre_avg_layernorm2:!", "self_attn.q_norm:!", "self_attn.k_norm:!",
            ),
            # Hymba's MoE reads `pre_moe_layernorm`; it has no
            # `post_attention_layernorm`. Declaring a norm the layer does not
            # own left the routed-expert gate/up block without a producer of
            # its own, so it was folded into `input_layernorm` and every
            # layer's experts were scaled by the wrong norm (#4891).
            "pre_moe_layernorm": ("pre_moe_layernorm:!",),
            "moe:moe": {
                "router:!": (),
                "experts": {
                    "#": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                }
            }
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # in_proj packs parallel Q/K/V/SSM/gate branches; out_proj consumes an
        # average of two normalized branches. Generic sequential folds cannot
        # score that objective or safely compensate shared KV consumers.
        # Keep all declared projections eligible for plain quantization.
        return []

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        result = super().prepare_layer_replay_kwargs(layer, layer_input, additional_inputs, target_device)
        bank = result.pop("_hymba_kv_sources", {})
        # StageInputsCapture records the first layer's mask policy. Later
        # global/local layers must select their own policy from config.
        result["use_swa"] = (layer.config.sliding_window is not None
                             and layer.layer_idx not in layer.config.global_attn_idx)
        source = layer.config.kv_sources[layer.layer_idx]
        if layer.reuse_kv:
            if source in bank:
                result["kv_last_layer"] = tuple(value.to(target_device) for value in bank[source])
            elif source != layer.layer_idx - 1 or result.get("kv_last_layer") is None:
                raise ValueError(f"Hymba replay is missing KV producer layer {source}")
        return result

    def update_layer_replay_kwargs_from_output(self, layer, layer_output, layer_input_kwargs, target_device):
        # Keep this request/batch's producer frontier in its existing replay
        # kwargs, never on the model or definition. The generic looper only
        # retains the preceding layer, which is insufficient for gapped groups.
        sources = layer.config.kv_sources
        bank = layer_input_kwargs.setdefault("_hymba_kv_sources", {})
        for source in list(bank):
            if source not in sources[layer.layer_idx:]:
                del bank[source]
        if layer.layer_idx in sources[layer.layer_idx + 1:]:
            bank[layer.layer_idx] = tuple(value.detach().to(target_device) for value in layer_output[-1])
        return layer_input_kwargs
