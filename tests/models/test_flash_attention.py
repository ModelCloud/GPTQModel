# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402



class Test(unittest.TestCase):

    MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1"

    def test(self):
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        model = GPTQModel.load(self.MODEL_ID)

        self.assertEqual(model.config._attn_implementation, "flash_attention_2")

        generate_str = tokenizer.decode(model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(model.device), max_new_tokens=2)[0])

        print(f"generate_str: {generate_str}")

        self.assertIn("paris", generate_str.lower())



