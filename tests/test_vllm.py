import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402

class TestLoadVLLM(unittest.TestCase):
    MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def test_load_vllm(self):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        model = GPTQModel.from_quantized(
            self.MODEL_ID,
            device="cuda:0",
            backend=BACKEND.VLLM,
            trust_remote_code=True,
        )
        outputs = model.generate(
            load_format="vllm",
            prompts=prompts,
            sampling_params=sampling_params,
        )
        print(f"pzs-----{model}")
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        self.assertTrue(model is not None)

    # def test_load_sglang(self):
    #     model = GPTQModel.from_quantized(
    #         self.MODEL_ID,
    #         device="cuda:0",
    #         backend=BACKEND.SGLANG,
    #         trust_remote_code=True,
    #     )
    #
    #     self.assertTrue(model is not None)