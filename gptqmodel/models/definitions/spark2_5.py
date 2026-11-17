# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class Spark2_5QModel(BaseQModel):
    """GPTQ definition for Spark-X2.5's hybrid-attention decoder."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.spark2_5.configuration_spark.Spark2_5Config",
        model_class="gptqmodel.models.ports.spark2_5.modeling_spark.Spark2_5ForCausalLM",
    )
    require_trust_remote_code = False
    fusable = None  # The fusion executor has not been qualified for this architecture.

    # Verified by the tiny real-forward case for XHToken/Spark-X2.5-4B.
    shared_input_verified_model_types = frozenset({"spark2_5"})

    pre_lm_head_norm_module = "model.norm"

    # Spark has no rotary-embedding module: it computes RoPE tensors directly
    # in the package-owned model's forward method.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_k_v_proj:0:in=x",
                # The 16-head output gate is small and precision-sensitive.
                "g_proj:0:!",
                "out_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": (
                "gate_proj:0:in=x",
                "up_proj:0:in=x",
                "down_proj:1",
            ),
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs=None):
        """Head gating and packed partial-RoPE attention are not fold-qualified."""
        return []


__all__ = ["Spark2_5QModel"]
