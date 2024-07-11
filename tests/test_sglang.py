import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402
import subprocess
import sys
from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestLoadSglang(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sglang>=0.1.19"])
        self.MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        self.prompt = "Hello, my name is"

    def test_load_sglang(self):
        model = GPTQModel.from_quantized(
            self.MODEL_ID,
            device="cuda:0",
            backend=BACKEND.SGLANG,
            disable_flashinfer=True,
        )
        output = model.generate(
            prompts=self.prompt,
            temperature=0.8,
            top_p=0.95,
        )
        print(f"Prompt: {self.prompt!r}, Generated text: {output!r}")

        self.assertTrue(output is not None)
        model.shutdown()
        del model

