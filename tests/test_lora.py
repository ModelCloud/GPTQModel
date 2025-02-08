# Copyright 2025 ModelCloud
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

# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.adapter.adapter import Lora
# -- end do not touch
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402


class Test(ModelTest):
    @parameterized.expand([
        BACKEND.TORCH,
        BACKEND.CUDA,
        BACKEND.TRITON,
        BACKEND.EXLLAMA_V1,
        # (BACKEND.EXLLAMA_V2), <-- adapter not working yet
        BACKEND.MARLIN,
        # (BACKEND.IPEX), <-- not tested yet
        # (BACKEND.BITBLAS, <-- not tested yet
    ])
    def test_load(self, backend: BACKEND):
        quant_model_path = "sliuau/llama3.2-1b-4bit-group128"
        lora_path = "adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc/blob/main/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc"

        # TODO, use local path before merge
        # quant_model_path = "/monster/data/model/sliuau-llama3.2-1b-4bit-group128"
        # lora_path = "/monster/data/model/sliuau-llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc/blob/main/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc"

        adapter = Lora(path=lora_path, rank=128)

        model = GPTQModel.load(
            quant_model_path,
            adapter=adapter,
            backend=backend,
            device_map="auto",
        )

        # print(model)
        tokens = model.generate("Capital of France is")[0]
        result = model.tokenizer.decode(tokens)
        print(f"Result: {result}")
        assert "paris" in result.lower()
