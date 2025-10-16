# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os

import torch
from transformers import AutoTokenizer


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestMultiGPUInference(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_PATH = "/monster/data/model/LongCat-Flash-Chat/gptq_4bits_groupsize128_maxlen2048_ns256_descTrue_damp0.025_mse2.4_09-03_15-48-35"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH, trust_remote_code=True)

    def test_multi_gpu_inference(self):
        cuda_device_count = torch.cuda.device_count()
        self.assertGreaterEqual(cuda_device_count, 5, f"Expected CUDA device count to be greater than or equal to 5, but got {cuda_device_count}")
        model = GPTQModel.load(
            self.MODEL_PATH,
            backend=BACKEND.TORCH,
            trust_remote_code=True,
            device_map="auto"
        )

        messages = [
            {"role": "user", "content": "How many p's are in the word \"apple\"? Please only respond with a number."},
        ]
        input_tensor = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        outputs = model.generate(
            inputs=input_tensor.to(model.device),
            max_length=512
        )

        result = self.tokenizer.decode(
            outputs[0][input_tensor.shape[1]:],
            skip_special_tokens=False
        )

        self.assertIn("2</longcat_s>", result.lower(), "The generated result should contain '2</longcat_s>'")
