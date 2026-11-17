# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..local import LocalModelLoader
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

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.mage_vl.configuration_mage_vl.MageVLConfig",
        model_class="gptqmodel.models.ports.mage_vl.modeling_mage_vl.MageVLForConditionalGeneration",
        processor_class="gptqmodel.models.ports.mage_vl.processing_mage_vl.MageVLProcessor",
    )

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Vision-dependent decoder replay and folds have not been qualified.
        return []

    def load_processor(self):
        return self.loader.load_processor(
            self.model_local_path,
            config=self.model.config,
            tokenizer=getattr(self.tokenizer, "tokenizer", self.tokenizer),
        )


__all__ = ["MageVLQModel"]
