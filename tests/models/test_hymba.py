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


class TestHymba(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Hymba-1.5B-Instruct/"  # "baichuan-inc/Baichuan2-7B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.2073
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2713
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.75
    MODEL_MAX_LEN = 8192
    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = True
    # Hymba currently only supports a batch size of 1.
    # See https://huggingface.co/nvidia/Hymba-1.5B-Instruct
    BATCH_SIZE = 1

    # Hymba currently tests that DESC_ACT=False to get better results.
    # If DESC_ACT=False, the output will be terrible.
    DESC_ACT = False


    def test_hymba(self):
        self.quant_lm_eval()
