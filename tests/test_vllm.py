import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import subprocess  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402

class TestLoadVLLM(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.5.1"])
        from vllm import SamplingParams  # noqa: E402
        self.MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        self.prompts = [
            "The capital of France is",
        ]
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

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
            self.assertEquals(generated_text, " Paris, which is also the capital of France.")
        outputs_param = model.generate(
            prompts=self.prompts,
            temperature=0.8,
            top_p=0.95,
        )
        for output in outputs_param:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            self.assertEquals(generated_text, " Paris. 2. Name the capital of the United States. 3.")

