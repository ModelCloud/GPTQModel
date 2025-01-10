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

import torch  # noqa: E402from tests.model_test import ModelTest
from model_test import ModelTest


class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/falcon-7b-instruct" # "tiiuae/falcon-7b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3993
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4292
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True
    TORCH_DTYPE = torch.float16
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.52
    BATCH_SIZE = 6
    USE_VLLM = False

    def test_falcon(self):
        self.quant_lm_eval()
