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
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import tempfile  # noqa: E402
import unittest  # noqa: E402

import transformers  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from packaging.version import Version  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig  # noqa: E402


class TestIntegration(unittest.TestCase):
    INFERENCE_PROMPT = "Which city is the capital of France? The city name is "
    INFERENCE_RESULT_KEYWORDS = ["paris", "eiffel", "country"]


    @classmethod
    def setUpClass(cls):
        assert Version(transformers.__version__) > Version("4.48.99")

    def _test_load_quantized_model_gptq_v1(self, device_map):
        model_id_or_path = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map=device_map)

        self.assertInference(model=model, tokenizer=tokenizer)

        del model
        torch_empty_cache()

    def _test_load_quantized_model_gptq_v2(self, device_map):
        model_id_or_path = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map=device_map)

        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

        self.assertInference(model=model, tokenizer=tokenizer)

        del model
        torch_empty_cache()

    def _test_quantize(self, device_map):
        model_id = "/monster/data/model/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = ["gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
        gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
        quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, quantization_config=gptq_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantized_model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)
            del quantized_model

            model = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map=device_map)

            generate_str = tokenizer.decode(model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(model.device))[0])

            self.assertIn("is a good", generate_str.lower())

            del model
            torch_empty_cache()

    def test_load_quantized_model_gptq_v1_ipex(self):
        self._test_load_quantized_model_gptq_v1(device_map="cpu")

    def test_load_quantized_model_gptq_v1_cuda(self):
        self._test_load_quantized_model_gptq_v1(device_map="cuda")

    def test_load_quantized_model_gptq_v2_ipex(self):
        self._test_load_quantized_model_gptq_v2(device_map="cpu")

    def test_load_quantized_model_gptq_v2_cuda(self):
        self._test_load_quantized_model_gptq_v2(device_map="cuda")

    def test_quantize_ipex(self):
        self._test_quantize(device_map="cpu")

    def test_quantize_cuda(self):
        self._test_quantize(device_map="cuda")

    def assertInference(self, model, tokenizer=None, keywords=None, prompt=INFERENCE_PROMPT):
        # gptqmodel can auto init tokenizer internally
        if keywords is None:
            keywords = self.INFERENCE_RESULT_KEYWORDS
        if tokenizer is None:
            tokenizer = model.tokenizer

        generated = self.generate(model, tokenizer, prompt).lower()
        for k in keywords:
            if k.lower() in generated:
                self.assertTrue(True)
                return
        self.assertTrue(False, f"none of keywords were found in generated: {generated}")

    def generate(self, model, tokenizer, prompt=None):
        if prompt is None:
            prompt = self.INFERENCE_PROMPT
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=10, max_new_tokens=30)
        output = tokenizer.decode(res[0])
        print(f"Result is: >>\n{output}\n<<")
        return output
