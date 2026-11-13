# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class GraniteMoeSWAQModel(BaseQModel):
    """Quantization definition for the sparse Granite MoE SWA decoder.

    The upstream checkpoint stores every routed expert as packed
    ``gate_up_proj``/``down_proj`` tensors. Defuser exposes those as numbered
    ``gate_proj``/``up_proj``/``down_proj`` Linear modules before GPT-QModel
    walks the tree, and the shared expert MLP stays a plain SwiGLU block.
    """

    # Per-layer sliding-window dispatch and post-attention sink scaling are not
    # modeled by the current fused qkv/gate_up executor.
    fusable = None

    layer_modules_strict = False
    dynamic_expert_index = "num_local_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "block_sparse_moe:moe:?": {
                "router": ("router:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
            "shared_mlp": ("input_linear:0", "output_linear:1"),
        },
    ]


__all__ = ["GraniteMoeSWAQModel"]
