# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Dots 3 Note routed-expert model definition for EXL3 quantization."""

import torch
import torch.nn as nn

from ...utils.exl3_router_candidates import bind_sigmoid_grouped_router_recovery
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class _SeparateDots3Experts(nn.ModuleList):
    """Preserve native per-expert checkpoint names as individual Linear modules.

    The upstream Transformers implementation stacks expert projections during
    model construction. That representation cannot expose a separate Hessian or
    EXL3 payload for each of the 256 routed experts. This adapter implements the
    same top-k weighted scatter while keeping `.experts.<id>.<proj>` names.
    """

    def __init__(self, config):
        from transformers.models.dots3_note.modeling_dots3_note import Dots3NoteTextMLP

        super().__init__([
            Dots3NoteTextMLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_local_experts)
        ])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        # The router may omit most experts for a calibration batch. Calling only
        # selected modules keeps forward replay bounded on a 121 GiB Spark.
        for expert_id in torch.unique(top_k_index).tolist():
            if expert_id < 0 or expert_id >= len(self):
                raise IndexError(f"Dots3 router selected invalid expert {expert_id}")
            token_idx, top_k_pos = torch.where(top_k_index == expert_id)
            values = self[expert_id](hidden_states[token_idx])
            values = values * top_k_weights[token_idx, top_k_pos, None]
            output.index_add_(0, token_idx, values.to(output.dtype))
        return output


class Dots3NoteQModel(BaseQModel):
    layer_modules_strict = False  # decoder layer 0 is dense
    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # The text-only calibration shell has no audio, vision, or MTP modules.
    # Retain these original FP8 tensors in the final export for vLLM.
    out_of_model_tensors = {
        "prefixes": ["vision_encoder.", "audio_encoder.", "model.layers.46.", "model.mtp."]
    }

    module_tree = [
        "model", "layers", "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0", "q_a_proj:0", "kv_a_proj_with_mqa:0",
                "q_b_proj:1:in=q_a", "kv_b_proj:1:in=kv_a", "g_proj:1",
                "indexer.wq_b:1", "indexer.wk:1", "indexer.weights_proj:1",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {"#": ("gate_proj:0", "up_proj:0", "down_proj:1")},
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        del model_local_path, load_quantized_model
        from transformers.models.dots3_note import modeling_dots3_note

        modeling_dots3_note.Dots3NoteTextExperts = _SeparateDots3Experts
        pretrained = modeling_dots3_note.Dots3NotePreTrainedModel
        if not getattr(pretrained, "_gptqmodel_separate_expert_init", False):
            original_init = pretrained._init_weights

            def _init_separate_expert_weights(self, module):
                # The native initializer expects stacked gate_up_proj fields.
                # Individual Linear children have already been initialized.
                if isinstance(module, _SeparateDots3Experts):
                    return
                return original_init(self, module)

            pretrained._init_weights = _init_separate_expert_weights
            pretrained._gptqmodel_separate_expert_init = True

    def after_model_load(self, model, load_quantized_model=False):
        del load_quantized_model
        expected = sum(kind == "sparse" for kind in model.config.mlp_layer_types)
        routers = 0
        for name, module in model.named_modules():
            if name.endswith(".mlp.gate") and hasattr(module, "e_score_correction_bias"):
                bind_sigmoid_grouped_router_recovery(module)
                routers += 1
        if routers != expected:
            raise RuntimeError(f"Dots3 router count mismatch: {routers} != {expected}")
        return model

    def zero_route_recovery_context(
        self, *, looper, processor, layer_module, subset, task_names
    ):
        # Dots3 and GLM use the same top-8 sigmoid/grouped-router policy.
        from .deepseek_v4 import DeepSeekV4MTPQuantizationModel

        return DeepSeekV4MTPQuantizationModel.zero_route_recovery_context(
            self,
            looper=looper,
            processor=processor,
            layer_module=layer_module,
            subset=subset,
            task_names=task_names,
        )
