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

from model_test import ModelTest


class TestInternlm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/internlm-7b" # "internlm/internlm-7b"
    NATIVE_ARC_CHALLENGE_ACC = 0.4164
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4309
    TRUST_REMOTE_CODE = True
    USE_VLLM = False
    DISABLE_FLASH_ATTN = True

    def test_internlm(self):
        # transformers<=4.44.2 run normal
        self.quant_lm_eval()
