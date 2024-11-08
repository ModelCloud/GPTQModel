import unittest

from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

pretrained_model_id = "facebook/opt-125m"
quantized_model_id = "facebook-opt-125m"

class Test_save_load_pt_weight(unittest.TestCase):
    def test_pt(self):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
        calibration_dataset = [
            tokenizer(
                "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]

        reference_output = "</s>gptqmodel is an easy-to-use model quantization library that can be used"

        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
        )

        model = GPTQModel.load(pretrained_model_id, quantize_config)

        model.quantize(calibration_dataset)

        model.save(quantized_model_id, use_safetensors=False)

        model = GPTQModel.load(quantized_model_id, device="cuda:0", use_safetensors=False)

        result = tokenizer.decode(model.generate(**tokenizer("gptqmodel is an easy-to-use model quantization library", return_tensors="pt").to(model.device))[0])

        self.assertEqual(result, reference_output)
