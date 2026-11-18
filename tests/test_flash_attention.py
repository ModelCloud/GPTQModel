# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch


import pytest  # noqa: E402
import torch  # noqa: E402


# FlashAttention-2 is an optional CUDA-only package.
pytest.importorskip("flash_attn")

from models.model_test import ModelTest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash_attention_2 runs on CUDA")
class Test(ModelTest):

    MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"

    def test(self):
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        model = GPTQModel.load(self.MODEL_ID, attn_implementation="flash_attention_2", device="cuda")

        self.assertEqual(model.config._attn_implementation, "flash_attention_2")

        self.assertInference(model=model,tokenizer=tokenizer)


