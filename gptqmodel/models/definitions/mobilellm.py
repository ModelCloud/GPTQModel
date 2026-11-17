# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..local import LocalModelLoader
from . import LlamaQModel


class MobileLLMQModel(LlamaQModel):
    fusable = None  # No qualified executor for this definition.

    require_trust_remote_code = False
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.mobilellm.MobileLLMConfig",
        model_class="gptqmodel.models.ports.mobilellm.MobileLLMForCausalLM",
    )

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Repeated shared decoder uses have not been qualified for AWQ folding.
        return []

    def pre_quantize(self, module):
        if self.model.config.pretraining_tp > 1:
            raise ValueError("MobileLLM pretraining_tp > 1 uses dense weight slices; quantization is not qualified")
        return super().pre_quantize(module)
