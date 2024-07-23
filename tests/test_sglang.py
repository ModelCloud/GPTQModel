import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import subprocess  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402
import torch  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402

torch_version = '.'.join(torch.__version__.split('.')[:2])
cuda_version = torch.version.cuda.replace('.', '')
pip_index_url = f"https://flashinfer.ai/whl/cu{cuda_version}/torch{torch_version}"

class TestLoadSglang(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # sglang set disable_flashinfer=True still import flashinfer
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flashinfer", "-i", pip_index_url])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sglang[srt]>=0.1.19"])

        self.MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        self.prompt = "The capital of France is"

    def test_load_sglang(self):
        model = GPTQModel.from_quantized(
            self.MODEL_ID,
            device="cuda:0",
            backend=BACKEND.SGLANG,
        )
        output = model.generate(
            prompts=self.prompt,
            temperature=0.8,
            top_p=0.95,
        )
        print(f"Prompt: {self.prompt!r}, Generated text: {output!r}")

        self.assertTrue(len(output)>5)
        model.shutdown()
        del model

