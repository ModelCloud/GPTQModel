# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import sys  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

GENERATE_EVAL_SIZE = 100


class TestsQ4Torch(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.float16, "mps"),
            (torch.bfloat16, "cpu"),
            (torch.float16, "cuda"),
        ]
    )
    def test_generation_desc_act_true(self, torch_dtype, device):
        if sys.platform == "darwin" and device not in ["cpu","mps"]:
            self.skipTest(f"MacOS env skipping unsupported device `{device}`")
        elif sys.platform == "linux" and device not in ["cpu", "cuda", "xpu"]:
            self.skipTest(f"MacOS env skipping unsupported device `{device}`")

        prompt = "I am in Paris and"

        # CPU implementation is extremely slow.
        new_tokens = 5
        reference_output = "<s> I am in Paris and I am in love with"

        model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        revision = "desc_act_true"

        model_q = GPTQModel.from_quantized(
            model_id,
            revision=revision,
            device=device,
            backend=BACKEND.TORCH,
            torch_dtype=torch_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        # This one uses Autocast.
        res = model_q.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])
        print("predicted_text", predicted_text)
        print("reference_output", reference_output)
        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])

        # This one does not.
        res = model_q.model.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])
        print("predicted_text", predicted_text)
        print("reference_output", reference_output)
        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])

    @parameterized.expand(
        [
            (torch.bfloat16, "cpu"),
            (torch.float16, "mps"),
            (torch.float16, "cuda"),
            # TODO: pending pytorch fix https://github.com/pytorch/pytorch/issues/100932
            # (torch.float16, "cpu"),
        ]
    )
    def test_generation_desc_act_false(self, torch_dtype, device):
        if sys.platform == "darwin" and device not in ["cpu","mps"]:
            self.skipTest(f"MacOS env skipping unsupported device `{device}`")
        elif sys.platform == "linux" and device not in ["cpu", "cuda", "xpu"]:
            self.skipTest(f"MacOS env skipping unsupported device `{device}`")

        prompt = "I am in Paris and"

        # CPU implementation is extremely slow.
        new_tokens = 5
        reference_output = "<s> I am in Paris and I am in love with"

        model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

        model_q = GPTQModel.from_quantized(
            model_id,
            device=device,
            backend=BACKEND.TORCH,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        # This one uses Autocast.
        res = model_q.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])

        # This one does not.
        res = model_q.model.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])
