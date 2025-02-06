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

from gptqmodel import QuantizeConfig, GPTQModel, BACKEND
from gptqmodel.quantization import EoRA

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

def test_load():
    quant_model_path = "sliuau/llama3.2-1b-4bit-group128"
    lora_path = "sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc"

    eora_config = EoRA(lora_path=lora_path, rank=128)


    model = GPTQModel.load(
        quant_model_path,
        extension=eora_config,
        backend=BACKEND.EORA_TORCH,
        device_map="auto",
    )

    # print(model)
    tokens = model.generate("Uncovering deep insights begins with")[0]
    result = model.tokenizer.decode(tokens)
    print(f"Result: {result}")
