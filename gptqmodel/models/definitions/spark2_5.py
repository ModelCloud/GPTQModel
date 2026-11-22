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
        """Generic AWQ groups without any fold into ``self_attn.out_proj``.

        ``out_proj`` reads the (optionally head-gated) attention output, and the
        packed ``q_k_v_proj`` output is wider than that, so no producer in the
        layer can absorb its scale. Without the head gate the declared
        ``g_proj:!`` is absent and the generic planner falls back to the stale
        ``input_layernorm`` for ``out_proj``, which is not its input.
        """

        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        out_proj = getattr(getattr(module, "self_attn", None), "out_proj", None)
        kept = []
        for node in nodes:
            layers = [layer for layer in node.get("layers") or [] if layer is not out_proj]
            if not layers:
                continue
            if len(layers) != len(node.get("layers") or []):
                node = dict(node, layers=layers)
                search = node.get("_awq_search_layers")
                if search is not None:
                    node["_awq_search_layers"] = [layer for layer in search if layer is not out_proj]
            kept.append(node)
        return kept


__all__ = ["Spark2_5QModel"]
