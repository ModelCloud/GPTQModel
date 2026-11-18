# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...utils.model import MODALITY
from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from .base_qwen2_vl import BaseQwen2VLGPTQ


class Qwen2_5_VLQModel(BaseQwen2VLGPTQ):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLConfigLoader",
        model_class="gptqmodel.models.ports.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModelLoader",
    )
    fusable = FusableConfig("qwen2_5_vl").merge(LlamaFusableConfig())

    pre_lm_head_norm_module = ["model.language_model.norm", "language_model.norm", "norm"]

    @staticmethod
    def _is_text_model(model):
        return getattr(getattr(model, "config", None), "model_type", None) == "qwen2_5_vl_text"

    @property
    def require_load_processor(self):
        return not self._is_text_model(self.model)

    @property
    def modality(self):
        return [MODALITY.TEXT] if self._is_text_model(self.model) else BaseQwen2VLGPTQ.modality

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        if cls._is_text_model(model):
            # Standalone native TextModel owns layers/embeddings at its root.
            # Never mutate the shared composite class tree.
            tree = tree[2:]
        return tree

    @classmethod
    def extract_layers_node(cls):
        paths = BaseQModel.extract_layers_node.__func__(cls)
        if paths == ["layers"]:
            return paths
        return super().extract_layers_node()

    @classmethod
    def get_base_modules(cls, model):
        if cls._is_text_model(model):
            return ["embed_tokens", "norm", "rotary_emb"]
        return super().get_base_modules(model)

    def pre_quantize_generate_hook_start(self):
        if not self._is_text_model(self.model):
            super().pre_quantize_generate_hook_start()

    def pre_quantize_generate_hook_end(self):
        if not self._is_text_model(self.model):
            super().pre_quantize_generate_hook_end()

    def prepare_dataset(self, calibration_dataset, batch_size=1, **kwargs):
        if self._is_text_model(self.model):
            return BaseQModel.prepare_dataset(self, calibration_dataset, batch_size=batch_size, **kwargs)
        return super().prepare_dataset(calibration_dataset, batch_size=batch_size, **kwargs)

    def run_input_capture(self, example, use_cache, data_device):
        if self._is_text_model(self.model):
            return BaseQModel.run_input_capture(self, example, use_cache=use_cache, data_device=data_device)
        return super().run_input_capture(example, use_cache=use_cache, data_device=data_device)

    def generate(self, inputs=None, **kwargs):
        if self._is_text_model(self.model):
            raise NotImplementedError(
                "qwen2_5_vl_text is a native headless backbone and does not support generation; "
                "use qwen2_5_vl for conditional generation"
            )
        return super().generate(inputs=inputs, **kwargs)
