# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


GENERATE_EVAL_SIZE = 100


class TestsQ4CUDA(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.bfloat16, "cuda:0"),
            (torch.float16, "cuda:0"),
        ]
    )
    def test_generation_desc_act_true(self, torch_dtype, device):

        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        revision = "desc_act_true"

        model_q = GPTQModel.from_quantized(
            model_id,
            revision=revision,
            device=device,
            backend=BACKEND.CUDA,
            torch_dtype=torch_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer("The capital of France is is", return_tensors="pt").to(device)

        # This one uses Autocast.
        generate_str = tokenizer.decode(model_q.generate(**inp, max_new_tokens=2)[0])
        print(f"generate_str: {generate_str}")
        self.assertIn("paris", generate_str.lower())

        # This one does not.
        generate_str = tokenizer.decode(model_q.model.generate(**inp, max_new_tokens=2)[0])
        print(f"generate_str: {generate_str}")
        self.assertIn("paris", generate_str.lower())

    @parameterized.expand(
        [
            (torch.bfloat16, "cuda:0"),
            (torch.float16, "cuda:0"),
        ]
    )
    def test_generation_desc_act_false(self, torch_dtype, device):
        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

        model_q = GPTQModel.from_quantized(
            model_id,
            device=device,
            backend=BACKEND.CUDA,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer("The capital of France is is", return_tensors="pt").to(device)

        # This one uses Autocast.
        generate_str = tokenizer.decode(model_q.generate(**inp, max_new_tokens=2)[0])
        print(f"generate_str: {generate_str}")
        self.assertIn("paris", generate_str.lower())

        # This one does not.
        generate_str = tokenizer.decode(model_q.model.generate(**inp, max_new_tokens=2)[0])
        print(f"generate_str: {generate_str}")
        self.assertIn("paris", generate_str.lower())

