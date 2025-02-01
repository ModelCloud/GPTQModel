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
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear  # noqa: E402
from gptqmodel.quantization import FORMAT  # noqa: E402
from gptqmodel.utils.importer import select_quant_linear  # noqa: E402
from gptqmodel.utils.model import gptqmodel_post_init  # noqa: E402
from test_q4_exllama_v1 import REFERENCE, get_diff  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

GENERATE_EVAL_SIZE = 100

class TestsQ4ExllamaV2(unittest.TestCase):
    def test_exllamav2(self):
        group_size = 128

        m = 1
        k = 1024
        n = 1024
        device = torch.device("cuda:0")
        pack_dtype = torch.int32

        linear_class = select_quant_linear(
            bits=4,
            group_size=group_size,
            desc_act=False,
            sym=True,
            backend=BACKEND.EXLLAMA_V2,
            format=FORMAT.GPTQ,
            pack_dtype=pack_dtype,
        )

        linear = linear_class(
            bits=4,
            group_size=group_size,
            desc_act=False,
            sym=True,
            in_features=k,
            out_features=n,
            bias=False,
            pack_dtype=pack_dtype,
        )

        self.assertTrue(isinstance(linear, ExllamaV2QuantLinear))

        torch.manual_seed(42)

        linear.qweight = torch.randint(-100, 100, size=linear.qweight.shape, dtype=torch.int32)
        linear.scales = linear.scales + 0.002
        linear.qzeros += 0b00010001000100010001000100010001  # for new weight format

        linear = linear.eval()
        linear = linear.to(device)

        linear = gptqmodel_post_init(linear, use_act_order=False)

        inp = torch.rand(1, m, k, dtype=torch.float16).to(device)

        with torch.no_grad():
            res = linear(inp)[0][0]

        reference = REFERENCE.to(device)

        self.assertTrue(
            torch.allclose(res, reference, rtol=3e-5, atol=2e-2),
            get_diff(res, reference),
        )

    def test_generation_desc_act_false(self):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\n(The stage is now dark, but the audience can see the characters walking around the stage.)\n\n(The stage is now lit up, but the audience can only see the characters' silhouettes.)\n\n("

        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

        model_q = GPTQModel.load(model_id, device="cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])

    def test_generation_desc_act_true(self):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\n(The stage is now dark, but the audience can see the characters walking around the stage.)\n\n(The stage is now lit up, but the audience can see the characters walking around the stage.)\n\n(The"

        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        revision = "desc_act_true"

        model_q = GPTQModel.load(
            model_id,
            rivision=revision,
            device="cuda:0",
            backend=BACKEND.EXLLAMA_V2,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])

    def test_exllama_v2_buffer_size(self):
        # prompt = "I'm in Paris and" * 450
        prompt = "I'm in Paris and" * 500
        device = torch.device("cuda:0")

        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        # revision = "desc_act_true"

        model_q = GPTQModel.from_quantized(
            model_id,
            # revision=revision,
            device="cuda:0",
            backend=BACKEND.EXLLAMA_V2,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        self.assertGreater(inp["input_ids"].shape[1], 2048)  # 2048 is the default max_input_length for LLama

        _ = model_q.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)
