# -- do not touch
import os
import tempfile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from gptqmodel.integration.optimum.quantizer import GPTQConfig  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

GENERATE_EVAL_SIZE = 100

from gptqmodel.nn_modules.qlinear.qlinear_exllama import ExllamaQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_exllamav2 import ExllamaV2QuantLinear  # noqa: E402


class TestTransformersIntegration(unittest.TestCase):

    def setUp(self):
        from gptqmodel.integration.optimum import monkey_patch_gptqmodel_into_transformers

        monkey_patch_gptqmodel_into_transformers()

        self.device = torch.device("cuda:0")
        self.prompt = "The International Space Station (ISS) is a large"

    @parameterized.expand(
        [
            1,
            2,
        ]
    )
    def test_load_gptq_model_with_exllama(self, exllama_version):
        model_id = "LnL-AI/opt-125M-autoround-lm_head-false-symTrue"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        gptq_config = GPTQConfig(bits=4, exllama_config={
            "version": exllama_version,
        })
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=gptq_config
        )
        reference_output = "</s>The International Space Station (ISS) is a large space station that is located in the International"
        self.assertResult(model, tokenizer, True, exllama_version, reference_output)

    @parameterized.expand(
        [
            1,
            2,
        ]
    )
    def test_quant_and_load(self, exllama_version):
        model_id = "facebook/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        calibration_dataset = [example["text"] for example in traindata.select(range(1024))]

        gptq_config = GPTQConfig(bits=4,
                                 group_size=128,
                                 sym=True,
                                 model_seqlen=2048,
                                 desc_act=False,
                                 exllama_config={
                                     "version": exllama_version,
                                 }, dataset=calibration_dataset)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=gptq_config
        )

        print("model", model)

        reference_output = "</s>The International Space Station (ISS) is a large space station that is located in the International Space Station"
        self.assertResult(model, tokenizer, False, exllama_version, reference_output)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(
                tmp_dir,
            )

            del model

            gptq_config = GPTQConfig(bits=4, exllama_config={
                "version": exllama_version,
            })
            model = AutoModelForCausalLM.from_pretrained(
                tmp_dir, device_map="auto", quantization_config=gptq_config,
            )

            self.assertResult(model, tokenizer, True, exllama_version, reference_output)

    def assertResult(self, model, tokenizer, load_quant_model, exllama_version, reference_output):
        if exllama_version == 1:
            self.assertIsInstance(model.model.decoder.layers[0].self_attn.k_proj, ExllamaQuantLinear)
        elif exllama_version == 2:
            if load_quant_model:
                self.assertIsInstance(model.model.decoder.layers[0].self_attn.k_proj, ExllamaV2QuantLinear)
            else:
                self.assertIsInstance(model.model.decoder.layers[0].self_attn.k_proj, ExllamaQuantLinear)
        inp = tokenizer(self.prompt, return_tensors="pt").to(self.device)
        res = model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)
        predicted_text = tokenizer.decode(res[0])
        print("predict", predicted_text)
        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])
