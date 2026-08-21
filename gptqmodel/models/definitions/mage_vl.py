# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoProcessor

from .base_qwen3_vl import BaseQwen3VLGPTQ


class MageVLQModel(BaseQwen3VLGPTQ):
    """Quantize Mage-VL's Qwen3 decoder while keeping the custom vision tower dense."""

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    # Mage-VL's proactive gate is stored outside the main checkpoint index, and
    # its custom processor does not inherit ProcessorMixin/save_pretrained.
    # Preserve both when writing a quantized checkpoint.
    out_of_model_tensors = {
        "files": [
            "streammind_gate.safetensors",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "chat_template.jinja",
        ],
    }

    require_trust_remote_code = True

    def load_processor(self):
        return AutoProcessor.from_pretrained(
            self.model_local_path,
            trust_remote_code=self.require_trust_remote_code,
        )


__all__ = ["MageVLQModel"]
