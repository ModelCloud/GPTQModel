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

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from gptqmodel.utils import BACKEND
from parameterized import parameterized
from inference_speed import InferenceSpeed


class TestInferenceSpeedIpex(InferenceSpeed):
    @parameterized.expand(
        [
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.IPEX, 12),
        ]
    )
    def test_inference_speed_ipex(self, model_path, backend, tokens_per_second):
        self.inference(model_path=model_path, backend=backend, tokens_per_second=tokens_per_second)