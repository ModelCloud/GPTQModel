# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Native Transformers Kimi-Linear decoder tree.

`kimi_linear` historically selected :class:`KimiLinearQModel`, which describes
the AttnRes / K3 checkpoint layout (``block_sparse_moe.experts.*.w1``) and
requires the bounded K3 runner. Native Transformers 5.22
``KimiLinearForCausalLM`` (Kimi-Linear-48B-A3B) has no AttnRes state and a
different tree: ``self_attn`` plus ``mlp.experts``/``mlp.shared_experts``.
"""

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


_LAYER_MODULES = {
    "input_layernorm": ("input_layernorm:!",),
    "self_attn": (
        # Linear-attention (Kimi Delta Attention) projections.
        "q_proj:0:in=x",
        "k_proj:0:in=x",
        "v_proj:0:in=x",
        "b_proj:0:in=x",
        "g_a_proj:0:in=x",
        "g_b_proj:1:in=g_a",
        "forget_gate.f_a_proj:0:in=x",
        "forget_gate.f_b_proj:1:in=f_a",
        # Linear-attention short convolution (checkpoint-owned, not a
        # quantizable projection).
        "conv1d:!",
        # Full-attention (MLA) projections.
        "q_a_proj:0:in=x",
        "q_b_proj:1:in=q_a",
        "kv_a_proj_with_mqa:0:in=x",
        "kv_b_proj:1:in=kv_a",
        # `o_proj` consumes the gated KDA/MLA output built from `g_b_proj`
        # (linear attention) or `kv_b_proj` (full attention), so it cannot share
        # their subset.
        "o_proj:2",
    ),
    "post_attention_layernorm": ("post_attention_layernorm:!",),
    "mlp:moe:?": {
        "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        "gate": ("gate:!",),
        "experts": {
            "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
        "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
    },
}


class KimiLinearNativeQModel(BaseQModel):
    """Quantize native ``KimiLinearForCausalLM`` with the ordinary GPTQ looper."""

    # auto.py dispatches non-K3 ``kimi_linear`` configs here (not through
    # MODEL_MAP), so this class needs its own package-owned loader: the vendored
    # Transformers 5.23 family, never ``transformers.models.kimi_linear``.
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
        tokenizer_class="gptqmodel.models.ports.kimi_k3.tokenization_kimi.KimiTokenizerFactory",
    )

    # Transformers 5.22 fuses the checkpoint's per-expert projections into 3-D
    # parameters; Defuser has no registry entry for this model type, so the
    # loader runs its generic experts pass explicitly.
    defuser_generic_experts = True

    dynamic_expert_index = "num_experts"
    layer_modules_strict = False  # first decoder layer is dense

    pre_lm_head_norm_module = "model.norm"

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = ["model", "layers", "#", _LAYER_MODULES]


def is_kimi_k3_config(config) -> bool:
    """Return whether a ``kimi_linear`` config still needs the K3 runner."""

    return bool(
        getattr(config, "use_attn_residuals", False)
        or getattr(config, "attn_res_block_size", None) is not None
    )


__all__ = ["KimiLinearNativeQModel", "is_kimi_k3_config"]
