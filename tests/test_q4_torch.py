# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestsQ4Torch(ModelTest):
    GENERATE_EVAL_SIZE_MIN = 20
    GENERATE_EVAL_SIZE_MAX = 20
    model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    @parameterized.expand(
        [
            (torch.bfloat16, "cpu"),
            (torch.float16, "cuda"),
        ]
    )
    def test_generation_desc_act_true(self, dtype, device):
        revision = "desc_act_true"

        qmodel = GPTQModel.from_quantized(
            self.model_id,
            revision=revision,
            device=device,
            backend=BACKEND.TORCH,
            dtype=dtype,
        )

        # gptqmodel
        self.assertInference(model=qmodel)

        # hf model
        self.assertInference(model=qmodel.model)

    @parameterized.expand(
        [
            (torch.bfloat16, "cpu"),
            (torch.float16, "cuda"),
            # TODO: pending pytorch fix https://github.com/pytorch/pytorch/issues/100932
            # (torch.float16, "cpu"),
        ]
    )
    def test_generation_desc_act_false(self, dtype, device):
        qmodel = GPTQModel.from_quantized(
            self.model_id,
            device=device,
            backend=BACKEND.TORCH,
            dtype=dtype,
        )

        # gptqmodel
        self.assertInference(model=qmodel)

        # hf model
        self.assertInference(model=qmodel.model)
