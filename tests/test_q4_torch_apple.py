# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys  # noqa: E402

import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestsQ4Torch(ModelTest):
    GENERATE_EVAL_SIZE_MIN = 5
    GENERATE_EVAL_SIZE_MAX = 10

    model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    @parameterized.expand(
        [
            (torch.float16, "mps"),
            (torch.bfloat16, "cpu"),
        ]
    )
    def test_generation_desc_act_true(self, dtype, device):
        if sys.platform != "darwin":
            self.skipTest("This test is macOS only")


        revision = "desc_act_true"

        model_q = GPTQModel.from_quantized(
            self.model_id,
            revision=revision,
            device=device,
            backend=BACKEND.TORCH,
            dtype=dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # This one uses Autocast.
        self.assertInference(model=model_q,tokenizer=tokenizer)
        # This one does not.
        self.assertInference(model=model_q.model,tokenizer=tokenizer)

    @parameterized.expand(
        [
            (torch.bfloat16, "cpu"),
            (torch.float16, "mps"),
        ]
    )
    def test_generation_desc_act_false(self, dtype, device):
        if sys.platform != "darwin":
            self.skipTest("This test is macOS only")

        model_q = GPTQModel.from_quantized(
            self.model_id,
            device=device,
            backend=BACKEND.TORCH,
            dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # This one uses Autocast.
        self.assertInference(model=model_q,tokenizer=tokenizer)
        # This one does not.
        self.assertInference(model=model_q.model,tokenizer=tokenizer)
