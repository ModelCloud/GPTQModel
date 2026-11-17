# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPT-QModel support for the text/VLM portion of Qwen-Drive.

The released Qwen-Drive checkpoint stores its Qwen3.5 VLM under ``vlm``.  The
planning expert and perception model are separate artifacts, so this
definition deliberately quantizes and delegates only the VLM text path.
"""

from collections.abc import Mapping

import torch

from ..local import LocalModelLoader
from ._vlm_calibration import finalize_conversation_batches, pop_conversation_calibration_options
from .qwen3_5 import Qwen3_5QModel


class QwenDriveQModel(Qwen3_5QModel):
    """Quantization definition for the ``vlm.*`` part of Qwen-Drive."""
    fusable = None  # No qualified executor for this definition.


    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.qwen_drive.QwenDriveConfig",
        model_class="gptqmodel.models.ports.qwen_drive.QwenDriveForConditionalGeneration",
    )

    require_trust_remote_code = False
    require_load_processor = False

    layer_modules_strict = False

    lm_head = "vlm.lm_head"
    pre_lm_head_norm_module = "vlm.model.language_model.norm"
    rotary_embedding = "vlm.model.language_model.rotary_emb"

    # This is the Qwen3.5 decoder topology with the additional ``vlm`` root
    # used by QwenDriveForPlanning. Vision stays dense; the separately released
    # planning and perception heads are outside this VLM-only definition.
    module_tree = [
        "vlm",
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "q_proj:0:in=x",
                "k_norm:!",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "linear_attn:!": (
                "norm:!",
                "conv1d:!",
                "in_proj_qkv:0:in=x",
                "in_proj_z:1:in=x",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
        },
    ]

    def awq_get_modules_for_scaling(self, *args, **kwargs):
        """Keep hybrid recurrent/gated attention on plain AWQ quantization."""
        return []

    def prepare_dataset(self, calibration_dataset, batch_size=1, **kwargs):
        media_keys = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
        if isinstance(calibration_dataset, (str, bytes)):
            return super().prepare_dataset(calibration_dataset, batch_size=batch_size, **kwargs)
        rows = [calibration_dataset] if isinstance(calibration_dataset, Mapping) else list(calibration_dataset)
        if not any(isinstance(row, Mapping) and media_keys.intersection(row) for row in rows):
            return super().prepare_dataset(calibration_dataset if isinstance(calibration_dataset, Mapping) else rows,
                                           batch_size=batch_size, **kwargs)
        if batch_size != 1:
            raise ValueError("Qwen-Drive media calibration requires prebatched native tensor rows and batch_size=1")
        sort, min_length = pop_conversation_calibration_options(kwargs, label="Qwen-Drive")
        allowed = media_keys | {"input_ids", "attention_mask", "position_ids", "mm_token_type_ids"}
        batches = []
        for row in rows:
            if not isinstance(row, Mapping) or "input_ids" not in row or set(row) - allowed:
                raise ValueError("Qwen-Drive media calibration requires native VLM tensor dictionaries")
            if not all(torch.is_tensor(value) for value in row.values()) or row["input_ids"].ndim != 2:
                raise ValueError("Qwen-Drive media calibration requires batched input_ids and native tensor values")
            batches.append(dict(row))
        return finalize_conversation_batches(batches, sort=sort, min_length=min_length, label="Qwen-Drive")

    def forward(self, *args, **kwargs):
        """Run the text/VLM forward path, excluding the absent planner."""

        return self.model.vlm(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Delegate text generation to the wrapped Qwen3.5 VLM."""

        return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.vlm.prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.vlm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.vlm.get_output_embeddings()


__all__ = ["QwenDriveQModel"]
