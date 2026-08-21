# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from ...utils.attn_mask import normalize_seq_mask
from ...utils.calibration import batched
from ...utils.model import MODALITY
from ..base import BaseQModel


class MuseGlimmerQModel(BaseQModel):
    loader = AutoModelForImageTextToText

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    # Muse Glimmer alternates sliding and full-attention layers. Keeping batches
    # at one avoids padding-dependent mask reuse while decoder layers are replayed.
    support_batch_quantize = False
    require_load_processor = True
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    # The attention gate consumes the same hidden states as Q/K/V, while o_proj
    # consumes the gated attention result. The vision tower intentionally stays
    # outside this tree and remains in its original precision.
    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0:q",
                "k_proj:0:k",
                "v_proj:0:v",
                "gate_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        },
    ]

    # Muse Glimmer uses GQA, so the attention output projection cannot reuse an
    # AWQ scale unless its predecessor has the exact compatible shape.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=False)

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        return processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.load_processor()
        calibration_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            calibration_data.append(self.prepare_inputs_for_conversations(processor, batch))
        del processor
        return calibration_data

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
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
        batch_size = hidden_states.shape[0] if hidden_states.ndim >= 2 else 1

        position_ids = additional_inputs.get("position_ids")
        if position_ids is None or position_ids.shape[-1] != sequence_length:
            position_ids = torch.arange(sequence_length, device=target_device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            additional_inputs["position_ids"] = position_ids

        attention_mask = additional_inputs.get("attention_mask")
        if torch.is_tensor(attention_mask):
            attention_mask = normalize_seq_mask(attention_mask, seq_len=sequence_length)

        self_attention = getattr(layer, "self_attn", None)
        layer_config = getattr(layer, "config", None) or getattr(self_attention, "config", None)
        if layer_config is None:
            return additional_inputs

        layer_index = getattr(self_attention, "layer_idx", None)
        mask_factory = (
            create_sliding_window_causal_mask
            if getattr(self_attention, "is_local_attention", False)
            else create_causal_mask
        )
        additional_inputs["attention_mask"] = mask_factory(
            config=layer_config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
            layer_idx=layer_index,
        )

        layer_rope_theta = getattr(layer_config, "layer_rope_theta", None)
        if layer_rope_theta is not None and layer_index is not None and not layer_rope_theta[layer_index]:
            additional_inputs["position_embeddings"] = None

        return additional_inputs


__all__ = ["MuseGlimmerQModel"]
