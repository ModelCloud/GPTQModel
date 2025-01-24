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
from parameterized import parameterized

from gptqmodel.utils import BACKEND
from inference_speed import InferenceSpeed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class TestInferenceSpeed(InferenceSpeed):
    @parameterized.expand(
        [
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.MARLIN, 748),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.CUDA, 493),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.EXLLAMA_V1, 717),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.EXLLAMA_V2, 775),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.TRITON, 296),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.TORCH, 295),
            (InferenceSpeed.BITBLAS_NATIVE_MODEL_ID, BACKEND.BITBLAS, 1474), # Second time running bitblas, there is cache
        ]
    )
    def test_inference_speed(self, model_path, backend, tokens_per_second):
        if backend == BACKEND.BITBLAS:
            self.inference(model_path=model_path, backend=backend, tokens_per_second=tokens_per_second, assert_result=False)

        self.inference(model_path=model_path, backend=backend, tokens_per_second=tokens_per_second)