# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Qwen4ExpQModel(BaseQModel):
    """Qwen3.8-Flash-Next / Qwen4 experimental multimodal MoE."""
    fusable = FusableConfig("qwen4_exp").merge(LlamaFusableConfig(), profiles=("qkv",))


    loader = AutoModelForImageTextToText
    require_load_processor = True
    layer_modules_strict = False

    dynamic_expert_index = "num_experts"

    # The final mixer replaces the usual decoder RMSNorm.
    pre_lm_head_norm_module = "model.language_model.hyper_connection_mixer"
    rotary_embedding = "model.language_model.rotary_emb"

    # Transformers intentionally ignores the auxiliary MTP decoder on load.
    out_of_model_tensors = {"prefixes": ["mtp"]}

    # The outer config reuses the text model's PLE checkpoint mapping.
    hf_conversion_model_type_alias = "qwen4_exp_text"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # GQA makes o_proj shape-incompatible with the Q/K/V AWQ scale group.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    # Only unmarked entries are quantized. PLE and the hyper-connection
    # transforms stay dense on purpose, so they carry `:!` instead of being
    # left undeclared (an undeclared Linear silently stays BF16).
    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            # PLE is optional per layer (`ple_layer_ids`) and injects hashed
            # n-gram features; its projections, depthwise conv and embedding
            # remain in the checkpoint dtype.
            "ple:!": (),
            # The hyper-connection mixers are gated low-rank transforms of the
            # residual stream, not quantization candidates.
            "attn_hyper_connection:!": (),
            "self_attn": (
                "indexer.index_qk_proj:!",
                "indexer.q_layernorm:!",
                "indexer.k_layernorm:!",
                "q_proj:0",
                "q_norm:!",
                "k_proj:0",
                "k_norm:!",
                "v_proj:0",
                # The hyper-connection `attn_hyper_connection` mixes the
                # residual through a gated low-rank transform; `o_proj`'s
                # output is not the MLP input, so it may not be a fold source.
                "o_proj:1:nosrc",
            ),
            "linear_attn": (
                "conv1d:!",
                "in_proj_qkv:0",
                "in_proj_z:1",
                "in_proj_b:!:1",
                # `in_proj_a` gates the DeltaNet decay, not `out_proj`'s
                # input; the gated norm sits between them.
                "in_proj_a:!:1:nosrc",
                "norm:!",
                # `mlp_hyper_connection` mixes the residual with a gated
                # low-rank transform before the MLP, so this output is not the
                # shared-expert input.
                "out_proj:2:nosrc",
            ),
            "mlp_hyper_connection:!": (),
            "mlp:moe": {
                # Keep shared experts separate so placeholder expansion does not duplicate them.
                "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_expert_gate": ("shared_expert_gate:!",),
            },
        },
    ]


class Qwen4ExpTextQModel(Qwen4ExpQModel):
    """Qwen4-Exp text-only checkpoints with the standalone decoder layout."""
    fusable = FusableConfig("qwen4_exp_text").merge(Qwen4ExpQModel.fusable)


    # Text-only checkpoints use the causal-LM loader and have no processor.
    loader = AutoModelForCausalLM
    require_load_processor = False

    # The text model stores its decoder directly under ``model`` rather than
    # under the multimodal wrapper's ``model.language_model``.
    pre_lm_head_norm_module = "model.hyper_connection_mixer"
    rotary_embedding = "model.rotary_emb"

    module_tree = [
        "model",
        "layers",
        "#",
        Qwen4ExpQModel.module_tree[-1],
    ]


__all__ = ["Qwen4ExpQModel", "Qwen4ExpTextQModel"]
