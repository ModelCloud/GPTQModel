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

import torch  # noqa: E402
from model_test import ModelTest


class TestGpt2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gpt2" # "openai-community/gpt2"
    NATIVE_ARC_CHALLENGE_ACC = 0.1903
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2270
    TORCH_DTYPE = torch.float16
    TRUST_REMOTE_CODE = True
    INPUTS_MAX_LENGTH = 1024

    def test_gpt2(self):
        self.quant_lm_eval()

