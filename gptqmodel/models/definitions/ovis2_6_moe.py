# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from torch import nn

from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from .ovis2_5 import Ovis2_5QModel


class Ovis2_6_MoeQModel(Ovis2_5QModel):
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "llm.model.norm"
    rotary_embedding = "llm.model.rotary_emb"

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    defuser_module_paths = ("llm",)

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "llm",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_norm:!", "k_norm:!", "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        }
    ]

    @staticmethod
    def _materialize_layernorm_defaults(layernorm: nn.LayerNorm, device: torch.device) -> None:
        if layernorm.weight is not None and (
            getattr(layernorm.weight, "is_meta", False) or layernorm.weight.device.type == "meta"
        ):
            layernorm.weight = nn.Parameter(
                torch.ones(layernorm.normalized_shape, device=device, dtype=layernorm.weight.dtype),
                requires_grad=layernorm.weight.requires_grad,
            )

        if layernorm.bias is not None and (
            getattr(layernorm.bias, "is_meta", False) or layernorm.bias.device.type == "meta"
        ):
            layernorm.bias = nn.Parameter(
                torch.zeros(layernorm.normalized_shape, device=device, dtype=layernorm.bias.dtype),
                requires_grad=layernorm.bias.requires_grad,
            )

    def _materialize_missing_vision_post_layernorm(self, device: torch.device) -> None:
        post_layernorm = getattr(
            getattr(getattr(self.model.visual_tokenizer, "vit", None), "vision_model", None),
            "post_layernorm",
            None,
        )
        if isinstance(post_layernorm, nn.LayerNorm):
            self._materialize_layernorm_defaults(post_layernorm, device)

    def pre_quantize_generate_hook_start(self):
        # Ovis 2.6 checkpoints omit SigLIP2 post_layernorm tensors even though
        # the remote code constructs the LayerNorm. Keep its default init instead of
        # resolving nonexistent checkpoint keys.
        self._materialize_missing_vision_post_layernorm(torch.device(self.quantize_config.device))
        super().pre_quantize_generate_hook_start()
