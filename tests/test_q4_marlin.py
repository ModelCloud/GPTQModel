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
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestQ4Marlin(unittest.TestCase):

    @parameterized.expand(
        [
            # act_order==False, group_size=128
            ("/monster/data/model/Llama-2-7B-GPTQ"),

            # act_order==True, group_size=128
            ("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"),
            # act_order==True, group_size=64
            ("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit/gptq-4bit-64g-actorder_True"),
            # act_order==True, group_size=32
            ("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit/gptq-4bit-32g-actorder_True"),

            # # 8-bit, act_order==True, group_size=channelwise
            ("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit/gptq-8bit--1g-actorder_True"),
            # # 8-bit, act_order==True, group_size=128
            ("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit/gptq-8bit-128g-actorder_True"),
            # # 8-bit, act_order==True, group_size=32
            ("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit/gptq-8bit-32g-actorder_True"),

            # # 4-bit, act_order==True, group_size=128
            ("/monster/data/model/gemma-1.1-2b-it-GPTQ")
        ]
    )
    def test_generation(self, model_id):
        prompt = "The capital of France is"
        device = torch.device("cuda:0")

        try:
            model_q = GPTQModel.load(model_id, device="cuda:0", backend=BACKEND.MARLIN)
        except ValueError as e:
            raise e

        has_marlin = False
        for _, module in model_q.named_modules():
            linear = MarlinQuantLinear
            if isinstance(module, linear):
                has_marlin = True
                break
        self.assertTrue(has_marlin)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0]).lower()

        self.assertTrue("paris" in predicted_text or "city" in predicted_text)

    def test_bias(self):
        # TheBloke/Llama-2-7B-Chat-GPTQ has bias, but they are all zeros, use a checkpoint which really uses bias.
        model_id = "/monster/data/model/starcoderbase-1b-GPTQ"
        try:
            model_q = GPTQModel.load(model_id, device="cuda:0", backend=BACKEND.MARLIN)
        except ValueError as e:
            raise e

        for _, param in model_q.named_parameters():
            self.assertNotEqual(param.device, torch.device("meta"))

        for _, param in model_q.named_buffers():
            self.assertNotEqual(param.device, torch.device("meta"))

        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_proj.bias) > 0)
        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_attn.bias) > 0)

        model_id = "/monster/data/model/starcoderbase-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        prompt = "The capital of France is"
        inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertIn("paris", predicted_text.lower())

