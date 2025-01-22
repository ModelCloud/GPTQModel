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
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear  # noqa: E402


class TestsQ4Triton(ModelTest):
    model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def test_generation_desc_act_false(self):

        model_q = GPTQModel.load(
            self.model_id,
            device="cuda:0",
            backend=BACKEND.TRITON,
            torch_dtype=torch.float16,
        )
        for _, submodule in model_q.named_modules():
            if isinstance(submodule, TritonV2QuantLinear):
                break
        else:
            raise ValueError("Did not find a tritonv2 linear layer")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # This one uses Autocast.
        self.assertInference(model=model_q,tokenizer=tokenizer)
        # This one does not.
        self.assertInference(model=model_q.model,tokenizer=tokenizer)

    def test_generation_desc_act_true(self):
        revision = "desc_act_true"

        model_q = GPTQModel.load(
            self.model_id,
            device="cuda:0",
            backend=BACKEND.TRITON,
            revision=revision,

        )
        for _, submodule in model_q.named_modules():
            if isinstance(submodule, TritonV2QuantLinear):
                break
        else:
            raise ValueError("Did not find a tritonv2 linear layer")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # This one uses Autocast.
        self.assertInference(model=model_q,tokenizer=tokenizer)
        # This one does not.
        self.assertInference(model=model_q.model,tokenizer=tokenizer)
