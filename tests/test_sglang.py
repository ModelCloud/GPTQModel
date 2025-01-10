# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestLoadSglang(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # sglang set disable_flashinfer=True still import flashinfer
        if importlib.util.find_spec("flashinfer") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flashinfer", "-i", f"https://flashinfer.ai/whl/cu{torch.version.cuda.replace('.', '')}/torch{'.'.join(torch.__version__.split('.')[:2])}"])
        if importlib.util.find_spec("sglang") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sglang[srt]>=0.3.2"])

        self.MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        self.prompt = "The capital of France is"

    def test_load_sglang(self):
        model = GPTQModel.load(
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

