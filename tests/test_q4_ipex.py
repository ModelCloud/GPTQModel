# -- do not touch
import os

import torch

from gptqmodel import BACKEND

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

GENERATE_EVAL_SIZE = 100


class TestsIPEX(unittest.TestCase):

    def test_ipex_format(self):
        prompt = "I am in Paris and"
        expected_output = "<s> I am in Paris and I am in love with"
        device = torch.device("cpu")

        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"  # "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

        model_q = GPTQModel.load(
            model_id,
            backend=BACKEND.IPEX,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        input = tokenizer(prompt, return_tensors="pt").to(device)

        result = model_q.generate(**input, num_beams=1, max_new_tokens=5)
        output = tokenizer.decode(result[0])
        self.assertEqual(output, expected_output)
