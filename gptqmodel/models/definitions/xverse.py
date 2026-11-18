# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from . import LlamaQModel


class XverseQModel(LlamaQModel):
    fusable = None  # No qualified executor for this definition.

    require_trust_remote_code = False
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.xverse.XverseConfig",
        model_class="gptqmodel.models.ports.xverse.XverseForCausalLM",
        tokenizer_class="gptqmodel.models.ports.xverse.XverseTokenizer",
    )
    dynamic_expert_index = "num_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()
    # `mlp:moe` exists only in the effective tree. Resolve the AWQ capture policy
    # against it so the routed root input is captured: the expert gate/up group
    # then replays the whole MoE block, which tolerates unrouted experts.
    _MODULE_TREE_RUNTIME_METHODS = LlamaQModel._MODULE_TREE_RUNTIME_METHODS | {"awq_input_feature_aggregation"}

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        blocks = tree[-1]
        attention = blocks["self_attn"]
        if not isinstance(attention, dict):
            attention = blocks["self_attn"] = {"": attention}
        attention.setdefault("rotary_emb", ("inv_freq:!",))
        if getattr(getattr(model, "config", None), "num_experts", 0) and "mlp:moe" not in blocks:
            blocks.pop("mlp", None)
            experts = {
                "router": ("router:!",),
                "experts": {"#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down")},
            }
            if model.config.num_shared_experts is not None:
                experts["shared_experts"] = ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down")
            blocks["mlp:moe"] = experts
        return tree

    def pre_quantize(self, module):
        if self.model.config.pretraining_tp > 1:
            raise ValueError("XVERSE pretraining_tp > 1 uses dense weight slices; quantization is not qualified")
        return super().pre_quantize(module)
