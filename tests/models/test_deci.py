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

from model_test import ModelTest


class TestDeci(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/DeciLM-7B-instruct" # "Deci/DeciLM-7B-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.5239
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5222
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.8
    TRUST_REMOTE_CODE = True
    USE_VLLM = False
    BATCH_SIZE = 6

    def test_deci(self):
        self.quant_lm_eval()
