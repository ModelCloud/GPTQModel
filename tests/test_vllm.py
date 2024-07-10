import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from vllm import SamplingParams  # noqa: E402


class TestLoadVLLM(unittest.TestCase):
    MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    def test_load_vllm(self):
        model = GPTQModel.from_quantized(
            self.MODEL_ID,
            device="cuda:0",
            backend=BACKEND.VLLM,
        )
        outputs = model.generate(
            prompts=self.prompts,
            sampling_params=self.sampling_params,
        )
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        outputs_param = model.generate(
            prompts=self.prompts,
            temperature=0.8,
            top_p=0.95,
        )
        for output in outputs_param:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        self.assertTrue(outputs is not None)
