import os
import sys
import unittest

from gptqmodel import BACKEND, GPTQModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"




class TestMlxGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/Qwen2.5-0.5B-Instruct/gptq_4bits_01-07_14-18-11_maxlen1024_ns1024_descFalse_damp0.1/"

    def test_mlx_generate(self):
        mlx_model = GPTQModel.load(
            self.pretrained_model_id,
            backend=BACKEND.MLX
        )

        prompt = "Tell me the city name. Which city is the capital of France?"

        messages = [{"role": "user", "content": prompt}]
        prompt = mlx_model.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        text = mlx_model.generate(prompt=prompt)
        assert "paris" in text.lower()



