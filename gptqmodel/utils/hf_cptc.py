# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Transformers load hooks for GPT-QModel CPTC safetensors checkpoints."""

from transformers.quantizers import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin


class HFCPTCConfig(QuantizationConfigMixin):
    def __init__(self, quant_method="cptc", **payload):
        self.quant_method = quant_method
        self.__dict__.update(payload)


class HFCPTCQuantizer(HfQuantizer):
    @property
    def is_trainable(self):
        return False

    def is_serializable(self, safe_serialization=None):
        return safe_serialization is not False

    def validate_environment(self, *args, **kwargs):
        if not self.pre_quantized:
            raise ValueError("Quantize CPTC weights with GPTQModel.quantize() before loading with Transformers")

    def _process_model_before_weight_loading(self, model, **kwargs):
        from .hf_load import hf_gptqmodel_prepare_model_for_load

        self.context = hf_gptqmodel_prepare_model_for_load(
            model, checkpoint_files=kwargs.get("checkpoint_files"), device_map=kwargs.get("device_map"),
        )
        if self.context is None or self.context.quantize_config.method.value != "cptc":
            raise ValueError("Cannot load CPTC without a valid GPT-QModel CPTC safetensors checkpoint")

    def _process_model_after_weight_loading(self, model, **kwargs):
        from .hf_load import hf_gptqmodel_post_init_for_load

        return hf_gptqmodel_post_init_for_load(model, self.context)
