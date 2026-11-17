# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPT-QModel definition for Agnes 3.0 Flash.

Agnes uses a hybrid language decoder: most layers use recurrent delta-rule
attention and every fourth layer uses grouped-query global attention.  The
vision tower and the auxiliary MTP tensors are deliberately outside the
quantized module tree.
"""

import torch
from transformers.masking_utils import create_causal_mask

from ...utils.attn_mask import normalize_seq_mask
from ...utils.model import MODALITY
from ..base import BaseQModel
from ..local import LocalModelLoader


class AgnesQModel(BaseQModel):
    """Quantization layout for the multimodal Agnes 3.0 Flash decoder."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.agnes.configuration_agnes.AgnesConfig",
        model_class="gptqmodel.models.ports.agnes.modeling_agnes.AgnesForConditionalGeneration",
        processor_class="gptqmodel.models.ports.agnes.processing_agnes.AgnesProcessor",
    )

    require_load_processor = True
    layer_modules_strict = False
    # Quantization calibration and the module tree cover the language backbone
    # only. The multimodal wrapper/processor still load normally, while the
    # vision tower remains at its native precision.
    modality = [MODALITY.TEXT]

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    out_of_model_tensors = {"prefixes": ["mtp"]}

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # Delta-rule attention has recurrent/convolution state.  Only the
            # learned projections are weight-only quantized; the state update
            # parameters and gated norm remain in their native precision.
            "delta_attn:!": (
                "norm:!",
                "conv1d:!",
                "in_proj_qkv:0:in=x",
                "in_proj_z:1:in=x",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "out_proj:2",
            ),
            # Global attention follows the Qwen3.5 projection grouping.  Q/K
            # RMS norms are helpers rather than quantized Linear modules.
            "global_attn": (
                "q_norm:!",
                "q_proj:0:in=x",
                "k_norm:!",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # Agnes always has the main FFN and may have a parallel FFN.  Keep
            # these branches in separate groups: their activations are formed
            # by separate module calls in the local forward implementation.
            "mlp": {
                "": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                "parallel_ffn": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Use plain AWQ quantization; hybrid/parallel norm folds are unqualified.

        The one-centred RMS norms, delta state/gates and parallel FFN must be
        scored together before any scale optimization can be admitted. Returning
        no groups leaves all declared linear projections eligible for quantization.
        """
        return []

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Refresh only global-attention masks during cached layer replay.

        The first decoder layer is a delta layer and consumes the original
        two-dimensional padding mask (or ``None``).  Global layers need a
        freshly-built causal mask.  Save the input before delegating to the
        base hook so a future base implementation cannot replace the padding
        signal before we rebuild the mask.
        """

        original_attention_mask = additional_inputs.get("attention_mask")
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )

        if not layer_input or not torch.is_tensor(layer_input[0]):
            return additional_inputs

        hidden_states = layer_input[0]
        sequence_length = hidden_states.shape[1] if hidden_states.ndim >= 2 else hidden_states.shape[0]

        # The remote decoder exposes one attention child according to its
        # layer type.  The attribute check keeps synthetic/minimal test layers
        # useful without relying on a remote-only class import.
        is_global = getattr(layer, "layer_type", None) == "agnes_global_attention" or hasattr(layer, "global_attn")

        if torch.is_tensor(original_attention_mask):
            if original_attention_mask.ndim == 2:
                padding_mask = original_attention_mask
            else:
                # Forward capture normally supplies [B, S].  If an older
                # replay path handed us an extended mask, reduce it back to
                # the same two-dimensional keep-mask for both layer types.
                padding_mask = normalize_seq_mask(original_attention_mask, seq_len=sequence_length)
        else:
            padding_mask = None

        if not is_global:
            additional_inputs["attention_mask"] = padding_mask
            return additional_inputs

        global_attn = getattr(layer, "global_attn", None)
        if global_attn is None:
            global_attn = getattr(layer, "self_attn", None)
        layer_config = getattr(global_attn, "config", None) or getattr(layer, "config", None)
        if layer_config is None:
            # A layer without config cannot safely construct the HF mask; keep
            # the original replay value rather than passing a malformed mask.
            additional_inputs["attention_mask"] = padding_mask
            return additional_inputs

        additional_inputs["attention_mask"] = create_causal_mask(
            config=layer_config,
            inputs_embeds=hidden_states,
            attention_mask=padding_mask,
            past_key_values=additional_inputs.get("past_key_values"),
            position_ids=additional_inputs.get("position_ids"),
            layer_idx=getattr(global_attn, "layer_idx", None),
        )
        return additional_inputs


__all__ = ["AgnesQModel"]
