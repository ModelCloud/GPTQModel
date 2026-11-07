# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from dataclasses import replace

from ..base import BaseQModel
from ..fusable import LlamaFusableConfig


class LlamaQModel(BaseQModel):
    fusable = replace(LlamaFusableConfig(),
        model_types=("llama", "cohere", "cohere2", "gemma", "granite", "helium", "mistral", "olmo", "olmo2", "seed_oss", "smollm3", "stablelm"),
        model_profiles={
            "cohere": LlamaFusableConfig().derive("cohere", parent_types={
                "qkv": (
                    "transformers.models.cohere.modeling_cohere.CohereAttention",
                ),
                "gate_up": (
                    "transformers.models.cohere.modeling_cohere.CohereMLP",
                ),
            }).profiles,
            "cohere2": LlamaFusableConfig().derive("cohere2", parent_types={
                "qkv": (
                    "transformers.models.cohere2.modeling_cohere2.Cohere2Attention",
                ),
                "gate_up": (
                    "transformers.models.cohere2.modeling_cohere2.Cohere2MLP",
                ),
            }).profiles,
            "gemma": LlamaFusableConfig().derive("gemma", parent_types={
                "qkv": (
                    "transformers.models.gemma.modeling_gemma.GemmaAttention",
                ),
                "gate_up": (
                    "transformers.models.gemma.modeling_gemma.GemmaMLP",
                ),
            }).profiles,
            "granite": LlamaFusableConfig().derive("granite", parent_types={
                "qkv": (
                    "transformers.models.granite.modeling_granite.GraniteAttention",
                ),
                "gate_up": (
                    "transformers.models.granite.modeling_granite.GraniteMLP",
                ),
            }).profiles,
            "helium": LlamaFusableConfig().derive("helium", parent_types={
                "qkv": (
                    "transformers.models.helium.modeling_helium.HeliumAttention",
                ),
                "gate_up": (
                    "transformers.models.helium.modeling_helium.HeliumMLP",
                ),
            }).profiles,
            "mistral": LlamaFusableConfig().derive("mistral", parent_types={
                "qkv": (
                    "transformers.models.mistral.modeling_mistral.MistralAttention",
                ),
                "gate_up": (
                    "transformers.models.mistral.modeling_mistral.MistralMLP",
                ),
            }).profiles,
            "olmo": LlamaFusableConfig().derive("olmo", parent_types={
                "qkv": (
                    "transformers.models.olmo.modeling_olmo.OlmoAttention",
                ),
                "gate_up": (
                    "transformers.models.olmo.modeling_olmo.OlmoMLP",
                ),
            }).profiles,
            "olmo2": LlamaFusableConfig().derive("olmo2", parent_types={
                "qkv": (
                    "transformers.models.olmo2.modeling_olmo2.Olmo2Attention",
                ),
                "gate_up": (
                    "transformers.models.olmo2.modeling_olmo2.Olmo2MLP",
                ),
            }).profiles,
            "seed_oss": LlamaFusableConfig().derive("seed_oss", parent_types={
                "qkv": (
                    "transformers.models.seed_oss.modeling_seed_oss.SeedOssAttention",
                ),
                "gate_up": (
                    "transformers.models.seed_oss.modeling_seed_oss.SeedOssMLP",
                ),
            }).profiles,
            "smollm3": LlamaFusableConfig().derive("smollm3", parent_types={
                "qkv": (
                    "transformers.models.smollm3.modeling_smollm3.SmolLM3Attention",
                ),
                "gate_up": (
                    "transformers.models.smollm3.modeling_smollm3.SmolLM3MLP",
                ),
            }).profiles,
            "stablelm": LlamaFusableConfig().derive("stablelm", parent_types={
                "qkv": (
                    "transformers.models.stablelm.modeling_stablelm.StableLmAttention",
                ),
                "gate_up": (
                    "transformers.models.stablelm.modeling_stablelm.StableLmMLP",
                ),
            }).profiles,
        },
    )
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
