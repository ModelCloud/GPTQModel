# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import ProcessorMixin

from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from .base_qwen2_5_omni import BaseQwen2_5_OmniGPTQ


class Qwen2_5_OmniGPTQ(BaseQwen2_5_OmniGPTQ):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForTextToWaveform",
        processor_class="gptqmodel.models.ports.native.auto.NativeProcessor",
        tokenizer_class="gptqmodel.models.ports.native.auto.NativeTokenizer",
    )
    fusable = FusableConfig("qwen2_5_omni").merge(LlamaFusableConfig())

    def load_processor(self) -> ProcessorMixin:
        processor = getattr(self, "processor", None)
        if processor is not None:
            return processor
        return self.loader.load_processor(
            self.model_local_path,
            config=self.model.config,
            tokenizer=getattr(self.tokenizer, "tokenizer", self.tokenizer),
        )
