import os
import unittest

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from gptqmodel import GPTQModel, BACKEND

class TestMLX(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct-gptq-converted-mlx"


    def test_mlx(self):
        self.mlx_model = GPTQModel.load(self.MODEL_ID, backend=BACKEND.MLX)
        prompt = "The capital of France is"

        messages = [{"role": "user", "content": prompt}]
        prompt = self.mlx_model.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        text = self.mlx_model.generate(prompt=prompt)
        assert "paris" in text.lower()


