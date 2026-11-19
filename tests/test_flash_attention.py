# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch


import pytest  # noqa: E402
from models.model_test import ModelTest, is_flash_attn_2_available  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402


# FlashAttention-2 is an optional CUDA-only package. Skip the test instead of
# failing the suite when it is not installed or no CUDA device is visible.
pytestmark = pytest.mark.skipif(
    not is_flash_attn_2_available(),
    reason="FlashAttention-2 requires the optional flash_attn package and a CUDA device",
)


class Test(ModelTest):

    MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"

    def test(self):
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        model = GPTQModel.load(self.MODEL_ID, attn_implementation="flash_attention_2", device="cuda")

        self.assertEqual(model.config._attn_implementation, "flash_attention_2")

        self.assertInference(model=model,tokenizer=tokenizer)


