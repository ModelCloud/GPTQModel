# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import importlib.util  # noqa: E402

from models.model_test import ModelTest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestLoadSglang(ModelTest):

    @classmethod
    def setUpClass(self):
        # sglang set disable_flashinfer=True still import flashinfer
        if importlib.util.find_spec("flashinfer") is None or importlib.util.find_spec("sglang") is None:
            raise RuntimeError("flashinfer and sglang are required by this test. you can install them by `pip install gptqmodel['sglang']`")

        self.MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def test_load_sglang(self):
        model = GPTQModel.load(
            self.MODEL_ID,
            device="cuda:0",
            backend=BACKEND.SGLANG,
        )
        output = model.generate(
            prompts=self.INFERENCE_PROMPT,
            temperature=0.8,
            top_p=0.95,
        )
        print(f"Prompt: {self.INFERENCE_PROMPT!r}, Generated text: {output!r}")

        self.assertTrue(len(output)>5)
        model.shutdown()
        del model

