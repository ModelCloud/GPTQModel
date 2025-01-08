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

import transformers
from model_test import ModelTest
from packaging.version import Version


class TestMiniCpm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/MiniCPM-2B-128k"  # "openbmb/MiniCPM-2B-128k"
    NATIVE_ARC_CHALLENGE_ACC = 0.3848
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4164
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 4

    def test_minicpm(self):
        args = {}
        # if flash_attn was installed and _attn_implementation_autoset was None, flash attention would be loaded
        # but device map is cpu, it will trow non-supported device error
        if Version(transformers.__version__) >= Version("4.46.0"):
            args["_attn_implementation_autoset"] = True

        self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE, **args)
