# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from . import LlamaQModel
from .gemma4 import _prepare_gemma4_replay_kwargs


# Gemma 4 unified drops the per-layer-input (PLE) feature that base Gemma 4 carries:
# Gemma4UnifiedTextConfig deletes vocab_size_per_layer_input/hidden_size_per_layer_input,
# so the decoder has no per_layer_input_gate / per_layer_projection / post_per_layer_input_norm
# modules and no project_per_layer_inputs capture hook. The quantizable layout is therefore the
# standard Gemma sandwich-norm decoder. It still uses per-layer-type rotary frequencies
# (sliding vs full attention), so cached-replay must refresh position_embeddings per layer.
_GEMMA4_UNIFIED_DECODER = {
    "input_layernorm": ("input_layernorm:!",),
    "self_attn": (
        "q_norm:!",
        "q_proj:0",
        "k_norm:!",
        "k_proj:0",
        "v_norm:!",
        "v_proj:0",
        "o_proj:1",
    ),
    "post_attention_layernorm": ("post_attention_layernorm:!",),
    "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
    "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
    "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
}


class Gemma4UnifiedTextQModel(LlamaQModel):
    """Quantization definition for text-only Gemma 4 unified checkpoints."""

    # Variants may share or omit KV projections (attention_k_eq_v / num_kv_shared_layers).
    layer_modules_strict = False
    # Per-layer-type rotary means cached replay must refresh position_embeddings per layer.
    support_batch_quantize = False
    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"
    module_tree = ["model", "layers", "#", _GEMMA4_UNIFIED_DECODER]

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Refresh sliding/full rotary kwargs during cached replay (no PLE reconstruction)."""

        return _prepare_gemma4_replay_kwargs(self, layer, layer_input, additional_inputs, target_device)


class Gemma4UnifiedForConditionalGenerationGPTQ(BaseQModel):
    """Quantization definition for composite (multimodal) Gemma 4 unified checkpoints."""

    layer_modules_strict = False
    support_batch_quantize = False
    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"
    module_tree = ["model", "language_model", "layers", "#", _GEMMA4_UNIFIED_DECODER]

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Refresh sliding/full rotary kwargs during cached replay (no PLE reconstruction)."""

        return _prepare_gemma4_replay_kwargs(self, layer, layer_input, additional_inputs, target_device)