# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import importlib.util  # noqa: E402
import unittest  # noqa: E402

import pytest  # noqa: E402
from models.model_test import ModelTest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.utils.sglang import SGLANG_AVAILABLE, SGLANG_INSTALL_HINT  # noqa: E402


pytestmark = [pytest.mark.model, pytest.mark.slow]

pytestmark = [pytest.mark.model, pytest.mark.slow]


class TestLoadSglang(ModelTest):

    @classmethod
    def setUpClass(self):
        # sglang set disable_flashinfer=True still import flashinfer
        if importlib.util.find_spec("flashinfer") is None:
            raise unittest.SkipTest(
                "flashinfer is required by this test. install via `pip install gptqmodel['sglang']`"
            )
        if importlib.util.find_spec("sglang") is None or not SGLANG_AVAILABLE:
            raise unittest.SkipTest(SGLANG_INSTALL_HINT)

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
