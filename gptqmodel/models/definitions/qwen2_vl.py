# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from .base_qwen2_vl import BaseQwen2VLGPTQ


class Qwen2VLQModel(BaseQwen2VLGPTQ):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.qwen2_vl.Qwen2VLConfigLoader",
        model_class="gptqmodel.models.ports.qwen2_vl.Qwen2VLModelLoader",
        processor_class="gptqmodel.models.ports.qwen2_vl.Qwen2VLProcessor",
        tokenizer_class="gptqmodel.models.ports.qwen2_vl.Qwen2Tokenizer",
    )
    fusable = FusableConfig("qwen2_vl").merge(LlamaFusableConfig())

    pre_lm_head_norm_module = ["model.language_model.norm", "language_model.norm", "norm"]

    def __init__(self, model, *args, **kwargs):
        if model.config.model_type == "qwen2_vl_text":
            self.require_load_processor = False
            self.modality = [self.modality[0]]
        super().__init__(model, *args, **kwargs)

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        if getattr(getattr(model, "config", None), "model_type", None) == "qwen2_vl_text":
            import copy

            return ["layers", "#", copy.deepcopy(cls.module_tree[-1])]
        return super()._resolve_effective_module_tree(model, quantize_config, detector=detector)

    @classmethod
    def extract_layers_node(cls):
        paths = BaseQModel.extract_layers_node.__func__(cls)
        return paths if paths == ["layers"] else super().extract_layers_node()

    @classmethod
    def get_base_modules(cls, model):
        if model.config.model_type == "qwen2_vl_text":
            return [name for name, _ in model.named_children() if name != "layers"]
        return super().get_base_modules(model)

    def pre_quantize_generate_hook_start(self):
        if self.model.config.model_type != "qwen2_vl_text":
            return super().pre_quantize_generate_hook_start()

    def pre_quantize_generate_hook_end(self):
        if self.model.config.model_type != "qwen2_vl_text":
            return super().pre_quantize_generate_hook_end()

    def run_input_capture(self, example, use_cache, data_device):
        if self.model.config.model_type == "qwen2_vl_text":
            return BaseQModel.run_input_capture(self, example, use_cache, data_device)
        return super().run_input_capture(example, use_cache, data_device)

    def prepare_dataset(self, calibration_dataset, batch_size=1, **kwargs):
        if self.model.config.model_type == "qwen2_vl_text":
            return BaseQModel.prepare_dataset(self, calibration_dataset, batch_size=batch_size, **kwargs)
        return super().prepare_dataset(calibration_dataset, batch_size=batch_size, **kwargs)

    def load_processor(self):
        # BaseQModel already selected the native processor with the normalized
        # tokenizer. Conversation calibration must reuse that owned instance.
        return self.processor

    def generate(self, inputs=None, **kwargs):
        if self.model.config.model_type == "qwen2_vl_text":
            raise NotImplementedError(
                "qwen2_vl_text is the native headless Qwen2VLTextModel for feature extraction; "
                "generation requires a qwen2_vl conditional-generation checkpoint with its trained LM head."
            )
        return super().generate(inputs=inputs, **kwargs)
