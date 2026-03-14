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
from gptqmodel.nn_modules.qlinear.bitblas import BitBLASQuantLinear  # noqa: E402

pytestmark = [pytest.mark.model, pytest.mark.slow]


class TestQ4BitBLAS(unittest.TestCase):
    def test_generation(self):
        prompt = "The capital city of France is named"
        device = torch.device("cuda:0")

        model_id = "/monster/data/model/opt-125M-autoround-lm_head-false-symTrue"

        try:
            model_q = GPTQModel.load(model_id, device="cuda:0", backend=BACKEND.BITBLAS)
        except ValueError as e:
            raise e

        has_bitblas = False
        for _, module in model_q.named_modules():
            if isinstance(module, BitBLASQuantLinear):
                has_bitblas = True
                break
        self.assertTrue(has_bitblas)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        predicted_text = ModelTest.generate_stable_with_limit(
            model_q,
            tokenizer,
            prompt,
            min_new_tokens=60,
            max_new_tokens=60,
            skip_special_tokens=False,
        )

        self.assertIn("paris", predicted_text.lower())

    def test_bias(self):
        # TheBloke/Llama-2-7B-Chat-GPTQ has bias, but they are all zeros, use a checkpoint which really uses bias.
        model_id = "/monster/data/model/starcoderbase-1b-GPTQ"

        model_q = GPTQModel.load(model_id, device="cuda:0", backend=BACKEND.BITBLAS)

        for _, param in model_q.named_parameters():
            self.assertNotEqual(param.device, torch.device("meta"))

        for _, param in model_q.named_buffers():
            self.assertNotEqual(param.device, torch.device("meta"))

        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_proj.bias) > 0)
        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_attn.bias) > 0)

        model_id = "/monster/data/model/starcoderbase-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        prompt = "The capital city of France is named"
        predicted_text = ModelTest.generate_stable_with_limit(
            model_q,
            tokenizer,
            prompt,
            min_new_tokens=60,
            max_new_tokens=60,
            skip_special_tokens=False,
        )
        self.assertIn("paris", predicted_text.lower())
