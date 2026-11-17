# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForCausalLM

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader


class _LlamaAliasLoader(LocalModelLoader):
    def __get__(self, instance, owner):
        # Other definitions inherit this module tree, not this alias allowlist.
        return self if owner is LlamaQModel else AutoModelForCausalLM


class LlamaQModel(BaseQModel):
    loader = _LlamaAliasLoader(
        config_class="gptqmodel.models.ports.llama.dispatch.LlamaAliasConfig",
        model_class="gptqmodel.models.ports.llama.dispatch.LlamaAliasForCausalLM",
    )

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        config = getattr(model, "config", None)
        extras = []
        if getattr(config, "model_type", None) == "longllama":
            extras.append("rotary_emb:!")
        if getattr(config, "model_type", None) == "cohere" and getattr(config, "use_qk_norm", False):
            extras.extend(("q_norm:!", "k_norm:!"))
        if getattr(config, "model_type", None) == "stablelm" and getattr(config, "qk_layernorm", False):
            for prefix, heads in (("q", config.num_attention_heads), ("k", config.num_key_value_heads)):
                extras.extend(f"{prefix}_layernorm.norms.{index}:!" for index in range(heads))
        if extras:
            tree[-1]["self_attn"] = (*tree[-1]["self_attn"], *extras)
        return tree

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        config = self.model.config
        if (config.model_type in {"longllama", "stablelm_epoch", "yi"}
                or (config.model_type == "cohere" and config.use_qk_norm)
                or (config.model_type == "stablelm" and config.qk_layernorm)):
            return []
        return super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)

    # Model-wide rotation fuses `input_layernorm` into q/k/v and
    # `post_attention_layernorm` into gate/up, then resets both norms to unit
    # gain. That is only exact for the plain Llama layout: an `x * w` RMSNorm
    # and a post-attention norm that is the MLP input. This class is also the
    # definition for other model types with different norm semantics (gemma,
    # cohere, stablelm, ...), so the opt-in names the verified model types
    # instead of claiming the class. Lookup is non-inheriting; see
    # `gptqmodel.models.base.rotation_supported` and
    # docs/rotation-model-support.md.
    supports_rotation = frozenset({"llama"})

    fusable = FusableConfig((
        "llama", "cohere", "cohere2", "gemma", "granite", "helium", "mistral",
        "olmo", "olmo2", "seed_oss", "smollm3", "stablelm",
    )).merge(LlamaFusableConfig())
    shared_input_verified_model_types = frozenset({"llama", "mistral", "gemma", "granite", "olmo2", "stablelm", "cohere", "cohere2"})


    pre_lm_head_norm_module = "model.norm"

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # the o_proj must match v_proj or else scaling optimizations are skipped (GQA vs MHA)
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!:norm",),
            "self_attn": ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1:o"),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
        }
    ]
