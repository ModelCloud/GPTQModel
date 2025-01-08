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

import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestsQ4CUDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)

    @parameterized.expand(
        [
            (torch.bfloat16, "cuda:0"),
            (torch.float16, "cuda:0"),
        ]
    )
    def test_generation_desc_act_true(self, torch_dtype, device):
        model_q = GPTQModel.from_quantized(
            self.model_id,
            revision="desc_act_true",
            device=device,
            backend=BACKEND.CUDA,
            torch_dtype=torch_dtype,
        )
        input = self.tokenizer("The capital of France is is", return_tensors="pt").to(device)

        # This one uses Autocast.
        self.generate(model_q, input)

        # This one does not.
        self.generate(model_q.model, input)

    @parameterized.expand(
        [
            (torch.bfloat16, "cuda:0"),
            (torch.float16, "cuda:0"),
        ]
    )
    def test_generation_desc_act_false(self, torch_dtype, device):
        model_q = GPTQModel.from_quantized(
            self.model_id,
            device=device,
            backend=BACKEND.CUDA,
            torch_dtype=torch_dtype,
        )

        input = self.tokenizer("The capital of France is is", return_tensors="pt").to(device)

        # This one uses Autocast.
        self.generate(model_q, input)

        # This one does not.
        self.generate(model_q.model, input)

    def generate(self, model, input):
        generate_str = self.tokenizer.decode(model.generate(**input, max_new_tokens=2)[0])
        print(f"generate_str: {generate_str}")
        self.assertIn("paris", generate_str.lower())
