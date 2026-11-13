# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from .llama import LlamaQModel


class GraniteSWAQModel(LlamaQModel):
    """Quantization definition for the dense Granite SWA decoder.

    The decoder layout matches Llama (GQA + SwiGLU + RMSNorm). Each attention
    module additionally owns a learnable per-head attention-sink parameter and
    dispatches a full or sliding-window mask per layer. The sink parameters stay
    in their source dtype; the attention projections and MLP leaves are
    quantized normally.
    """

    # Per-layer sliding-window dispatch and post-attention sink scaling are not
    # modeled by the current fused qkv/gate_up executor.
    fusable = None

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!:norm",),
            # Exclude the direct sink parameter, while quantizing the linears.
            "self_attn:!": ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1:o"),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
        },
    ]


__all__ = ["GraniteSWAQModel"]
