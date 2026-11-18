# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Llama4QModel(BaseQModel):
    # some bug in the attention_mask of transformers.modeling_llama4,
    # so batch quantization for Llama4 is temporarily not supported.
    fusable = FusableConfig("llama4").merge(LlamaFusableConfig(), profiles=("qkv",))

    support_batch_quantize = False
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForImageTextToText",
    )

    # The conditional-generation wrapper keeps the causal head on its inner
    # `Llama4ForCausalLM`, not at the top level.
    lm_head = "language_model.lm_head"

    pre_lm_head_norm_module = "language_model.model.norm"

    dynamic_expert_index = "num_local_experts"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    @staticmethod
    def _decoder_config(config):
        """The text decoder config: ``Llama4Config`` nests it under ``text_config``."""

        return getattr(config, "text_config", config)

    @classmethod
    def _drop_dense_feed_forward(cls, tree):
        """Remove the dense ``feed_forward`` fallback from every tree variant."""

        for variant in cls._iter_module_tree_variants(tree):
            for item in variant:
                if not isinstance(item, dict):
                    continue
                branch = item.get("feed_forward:moe")
                if isinstance(branch, dict):
                    branch.pop("", None)

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        config = cls._decoder_config(getattr(model, "config", None))
        num_layers = getattr(config, "num_hidden_layers", None)
        moe_layers = getattr(config, "moe_layers", None)
        if num_layers and moe_layers is not None and set(range(num_layers)) <= set(moe_layers):
            # Llama-4 Scout (`interleave_moe_layer_step=1`) has no dense MLP in
            # any decoder layer, so the dense fallback #4245 declared for
            # Maverick's interleaved layers cannot resolve. Drop it instead of
            # relaxing `layer_modules_strict` for the whole family (#4689).
            cls._drop_dense_feed_forward(tree)
        return tree

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "feed_forward:moe": {
                "router": ("router:!",), # Llama4Router.forward() returns two values, skipping its quantization.
                "experts:0": {
                    "#": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                },
                "shared_expert:0": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                # Maverick interleaves dense and sparse layers
                # (`interleave_moe_layer_step=2`), so the dense fallback is a
                # real branch. Without this entry its gate/up/down projections
                # never resolve and stay in the source dtype.
                "": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
            },
        }
    ]


class Llama4TextQModel(Llama4QModel):
    fusable = FusableConfig("llama4_text").merge(Llama4QModel.fusable)

    shared_input_verified_model_types = frozenset({"llama4_text"})

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    # `Llama4ForCausalLM` owns the head at the top level; the multimodal
    # wrapper above keeps it on `language_model`.
    lm_head = "lm_head"
    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "feed_forward:moe": {
                "router": ("router:!",),
                "experts:0": {
                    "#": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                },
                "shared_expert:0": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                "": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
            },
        },
    ]


__all__ = ["Llama4QModel", "Llama4TextQModel"]
