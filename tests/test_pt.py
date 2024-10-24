import torch
import unittest

from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig

pretrained_model_id = "facebook/opt-125m"
quantized_model_id = "facebook-opt-125m"

class TestsQ4ExllamaV2(unittest.TestCase):
    def test_pt(self):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
        calibration_dataset = [
            tokenizer(
                "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]

        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
        )

        model = GPTQModel.from_pretrained(pretrained_model_id, quantize_config)

        model.quantize(calibration_dataset)

        model.save_quantized(quantized_model_id, use_safetensors=False)

        model = GPTQModel.from_quantized(quantized_model_id, device="cuda:0", use_safetensors=False)

        result = tokenizer.decode(model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(model.device))[0])

        assert len(result) > 0

