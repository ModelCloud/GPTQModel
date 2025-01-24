# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestsQ4Torch(ModelTest):
    GENERATE_EVAL_SIZE_MIN = 5
    GENERATE_EVAL_SIZE_MAX = 10
    model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    @parameterized.expand(
        [
            (torch.bfloat16, "cpu"),
            (torch.float16, "cuda"),
        ]
    )
    def test_generation_desc_act_true(self, torch_dtype, device):
        revision = "desc_act_true"

        qmodel = GPTQModel.from_quantized(
            self.model_id,
            revision=revision,
            device=device,
            backend=BACKEND.TORCH,
            torch_dtype=torch_dtype,
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
    def test_generation_desc_act_false(self, torch_dtype, device):
        qmodel = GPTQModel.from_quantized(
            self.model_id,
            device=device,
            backend=BACKEND.TORCH,
            torch_dtype=torch_dtype,
        )

        # gptqmodel
        self.assertInference(model=qmodel)

        # hf model
        self.assertInference(model=qmodel.model)
