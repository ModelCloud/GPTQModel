# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear.bitblas import BitBLASLinear  # noqa: E402


pytestmark = [pytest.mark.model, pytest.mark.slow]


class TestQ4BitBLAS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load one small BitBLAS-backed fixture for both coverage checks."""
        cls.model_id = "/monster/data/model/opt-125M-autoround-lm_head-false-symTrue"
        cls.model_q = GPTQModel.load(cls.model_id, device="cuda:0", backend=BACKEND.BITBLAS)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)

    def test_generation(self):
        prompt = "The capital city of France is named"

        has_bitblas = False
        for _, module in self.model_q.named_modules():
            if isinstance(module, BitBLASLinear):
                has_bitblas = True
                break
        self.assertTrue(has_bitblas)

        predicted_text = ModelTest.generate_stable_with_limit(
            self.model_q,
            self.tokenizer,
            prompt,
            min_new_tokens=60,
            max_new_tokens=60,
            skip_special_tokens=False,
        )

        self.assertIn("paris", predicted_text.lower())

    def test_bias(self):
        for _, param in self.model_q.named_parameters():
            self.assertNotEqual(param.device, torch.device("meta"))

        for _, param in self.model_q.named_buffers():
            self.assertNotEqual(param.device, torch.device("meta"))

        self.assertGreater(
            torch.count_nonzero(self.model_q.model.model.decoder.layers[0].self_attn.q_proj.bias),
            0,
        )
        self.assertGreater(
            torch.count_nonzero(self.model_q.model.model.decoder.layers[0].fc1.bias),
            0,
        )

        prompt = "The capital city of France is named"
        predicted_text = ModelTest.generate_stable_with_limit(
            self.model_q,
            self.tokenizer,
            prompt,
            min_new_tokens=60,
            max_new_tokens=60,
            skip_special_tokens=False,
        )
        self.assertIn("paris", predicted_text.lower())
