# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
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
from gptqmodel import BACKEND  # noqa: E402
from models.model_test import ModelTest  # noqa: E402


class TestsTorchFused(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"  # "bigscience/bloom-560m"
    NATIVE_ARC_CHALLENGE_ACC = 0.28
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.31
    TORCH_DTYPE = torch.float16
    LOAD_BACKEND = BACKEND.TORCH_FUSED
    DELETE_QUANTIZED_MODEL = False
    USE_VLLM = False

    def test_torch_fused(self):
        self.quant_lm_eval()
