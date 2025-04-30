# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ...utils import BACKEND
from ...utils.logger import setup_logger
from . import LlamaGPTQ

log = setup_logger()

SUPPORT_ERR = "Currently, only vLLM/SGLang with flashinfer enabled can correctly inference a quantized Gemma2-27B model. Pre-quantized model with sample vLLM code: https://huggingface.co/ModelCloud/gemma-2-27b-it-gptq-4bit ."


class Gemma2GPTQ(LlamaGPTQ):
    layer_type = "Gemma2DecoderLayer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # There is an issue with duplicate outputs in the quantized gemma-2 model 27b with transformers.
        if hasattr(self.model.config, "num_hidden_layers"):
            num_hidden_layers = getattr(self.model.config, "num_hidden_layers")
            # The gemma-2 model 9b has 42 hidden layers, while the gemma-2 model 27b has 46 hidden layers.
            if num_hidden_layers > 42:
                if not self.quantized:
                    log.warn(SUPPORT_ERR)
                    return

                # quantized gemma-2 27b model only support vLLM/SGLang load.
                from ...utils.vllm import VLLM_AVAILABLE
                if VLLM_AVAILABLE:
                    from vllm import LLM
                    if isinstance(self.model, LLM):
                        backend = BACKEND.VLLM

                from ...utils.sglang import SGLANG_AVAILABLE
                if SGLANG_AVAILABLE:
                    from sglang.srt.server import Runtime
                    if isinstance(self.model, Runtime):
                        backend = BACKEND.SGLANG

                if backend not in [BACKEND.VLLM,  BACKEND.SGLANG]:
                    raise ValueError(SUPPORT_ERR)

