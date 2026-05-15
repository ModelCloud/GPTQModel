# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os

from safetensors import safe_open

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ..base import BaseQModel


class MimoV2QModel(BaseQModel):
    # MiMo V2 uses repository-defined configuration/modeling classes.
    require_trust_remote_code = True

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # MiMo V2 supports both split q/k/v and fused qkv checkpoints, and individual
    # layers can be dense MLP or routed MoE according to config.moe_layer_freq.
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("qkv_proj:0", "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    @staticmethod
    def _checkpoint_has_tensor(model_local_path: str, tensor_name: str) -> bool:
        if not model_local_path:
            return True

        index_path = os.path.join(model_local_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as fp:
                weight_map = json.load(fp).get("weight_map", {})
            return tensor_name in weight_map

        tensor_file = os.path.join(model_local_path, "model.safetensors")
        if os.path.isfile(tensor_file):
            with safe_open(tensor_file, framework="pt", device="cpu") as handler:
                return tensor_name in handler.keys()

        return True

    @staticmethod
    def _drop_visual_ln_q_bias_if_checkpoint_omits_it(model, model_local_path: str) -> None:
        visual = getattr(model, "visual", None)
        merger = getattr(visual, "merger", None)
        ln_q = getattr(merger, "ln_q", None)
        if ln_q is None or getattr(ln_q, "bias", None) is None:
            return

        bias_name = "visual.merger.ln_q.bias"
        if MimoV2QModel._checkpoint_has_tensor(model_local_path, bias_name):
            return

        # MiMo V2.5 Base checkpoints omit this default LayerNorm bias; keep
        # the shell parameters aligned so offload-backed save does not chase it.
        ln_q.register_parameter("bias", None)

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)
        self._drop_visual_ln_q_bias_if_checkpoint_omits_it(model, self.model_local_path)
        return model
