# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os

import pytest
import torch
from transformers import AutoTokenizer


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

from models.model_test import ModelTest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


pytestmark = [pytest.mark.model, pytest.mark.slow]


class TestMultiGPUInference(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_PATH = "/monster/data/model/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH, trust_remote_code=True)

    def test_multi_gpu_inference(self):
        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count < 5:
            self.skipTest(f"Need at least 5 visible CUDA devices, got {cuda_device_count}.")
        model = GPTQModel.load(
            self.MODEL_PATH,
            backend=BACKEND.TORCH,
            trust_remote_code=True,
            device_map="auto"
        )

        messages = [
            {"role": "user", "content": "How many p's are in the word \"apple\"? Please only respond with a number."},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        input_ids = model_inputs["input_ids"]
        result = ModelTest.generate_stable_with_limit(
            model,
            self.tokenizer,
            inputs=model_inputs,
            max_new_tokens=512,
            decode_start_idx=input_ids.shape[1],
            skip_special_tokens=False,
        )

        self.assertIn("2<|im_end|>", result.lower(), "The generated result should contain '2<|im_end|>'")
