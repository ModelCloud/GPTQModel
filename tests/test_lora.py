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
# -- end do not touch

from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402


class Test(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/sliuau-llama3.2-1b-4bit-group128"
    lora_path = "/monster/data/model/sliuau-llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora_test-rank128-arc/blob/main/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora_test-rank128-arc"

    NATIVE_ARC_CHALLENGE_ACC = 0.3567
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3805
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.36

    @classmethod
    def setUpClass(cls):
        cls.adapter = Lora(path=cls.lora_path, rank=128)

    @parameterized.expand([
        # BACKEND.EXLLAMA_V2V,
        #BACKEND.TORCH,
        # BACKEND.CUDA,
        # BACKEND.TRITON,
        # BACKEND.EXLLAMA_V1,
        BACKEND.EXLLAMA_V2,
        # BACKEND.MARLIN,
        # # (BACKEND.IPEX), <-- not tested yet
        # # (BACKEND.BITBLAS, <-- not tested yet
    ])
    def test_load(self, backend: BACKEND):
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            adapter=self.adapter,
            backend=backend,
            device_map="auto",
        )

        # print(model)
        tokens = model.generate("Capital of France is")[0]
        result = model.tokenizer.decode(tokens)
        print(f"Result: {result}")
        self.assertIn("paris", result.lower())

    @parameterized.expand([
        BACKEND.EXLLAMA_V2,
    ])
    def test_download(self, backend: BACKEND):
        adapter = Lora(path="https://huggingface.co/sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc/blob/main/adapter_model.safetensors", rank=128)

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            adapter=adapter,
            backend=backend,
            device_map="auto",
        )

        tokens = model.generate("Capital of France is")[0]
        result = model.tokenizer.decode(tokens)
        print(f"Result: {result}")
        self.assertIn("paris", result.lower())

    def test_lm_eval_from_path(self):
        adapter = Lora(path=self.lora_path, rank=128)
        task_results = self.lm_eval(self.NATIVE_MODEL_ID, extra_args={"adapter": adapter.to_dict()}) # "backend":"exllama_v2",
        self.check_results(task_results)

    def test_lm_eval_from_model(self):
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            adapter=self.adapter,
            # backend=BACKEND.EXLLAMA_V2V,
        )
        task_results = self.lm_eval(model)
        self.check_results(task_results)
