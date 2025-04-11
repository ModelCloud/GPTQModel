import unittest

import torch
from gptqmodel import BACKEND, GPTQModel


class TestQwen2Moe(unittest.TestCase):
    def test_inference(self):
        model = GPTQModel.load("Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
                device=torch.device("cuda:0"),
                backend=BACKEND.MARLIN)

        tokenizer = model.tokenizer

        prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=128
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Response: `{response}`")
