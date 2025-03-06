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

from gptqmodel import GPTQModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import tempfile  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402
from peft import AdaLoraConfig, get_peft_model  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from datasets import load_dataset  # noqa: E402

import transformers  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from packaging.version import Version  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig  # noqa: E402


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

    def test_peft(self):
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
        model_kwargs["quantization_config"] = GPTQConfig(bits=4, dataset=['/monster/data/model/dataset/c4-train.00000-of-01024.json.gz'])

        model = AutoModelForCausalLM.from_pretrained("/monster/data/model/opt-125m", **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained("/monster/data/model/opt-125m")
        dataset = load_dataset("json", data_files="/monster/data/model/dataset/c4-train.00000-of-01024.json.gz", split="train")

        config = AdaLoraConfig(
            total_step=20,
        )

        peft_model = get_peft_model(model, config)
        training_args = SFTConfig(dataset_text_field="text", max_seq_length=128)
        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
        )
        trainer.train()
        with tempfile.TemporaryDirectory() as tmpdir:
            peft_model.save_pretrained(tmpdir)
            model = GPTQModel.load(tmpdir)
            generated = self.generate(model, tokenizer, "Which city is the capital of France? The city name is ").lower()
            self.assertIn("paris", generated)