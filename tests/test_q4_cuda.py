# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import BACKEND  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

GENERATE_EVAL_SIZE = 100


class TestsQ4CUDA(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.float32, "cpu"),
            (torch.float32, "cuda:0"),
            (torch.float16, "cuda:0"),
        ]
    )
    def test_generation_desc_act_true(self, torch_dtype, device):
        prompt = "I am in Paris and"

        # Reference generated with the cuda-old kernel
        if device == "cpu":
            # CPU implementation is extremely slow.
            new_tokens = 2
            reference_output = "<s> I am in Paris and I am"
        else:
            reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\n(The stage is now dark, but the audience can see the characters walking around the stage.)\n\n(The stage is now lit up, but the audience can only see the characters' silhouettes.)\n\n("
            new_tokens = 60

        model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        revision = "desc_act_true"

        model_q = GPTQModel.from_quantized(
            model_id,
            revision=revision,
            device=device,
            backend=BACKEND.CUDA,
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
            (torch.float32, "cpu"),
            (torch.float32, "cuda:0"),
            (torch.float16, "cuda:0"),
        ]
    )
    def test_generation_desc_act_false(self, torch_dtype, device):
        prompt = "I am in Paris and"

        # Reference generated with the cuda-old kernel
        if device == "cpu":
            # CPU implementation is extremely slow.
            new_tokens = 3
            reference_output = "<s> I am in Paris and I am in"
        else:
            reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\n(The stage is now dark, but the audience can see the characters walking around the stage.)\n\n(The stage is now lit up, but the audience can see the characters walking around the stage.)\n\n(The"
            new_tokens = 60

        model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

        model_q = GPTQModel.from_quantized(
            model_id,
            device=device,
            backend=BACKEND.CUDA,
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
