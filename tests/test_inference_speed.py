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
from gptqmodel.utils import BACKEND  # noqa: E402
# -- end do not touch
from inference_speed import InferenceSpeed  # noqa: E402
from parameterized import parameterized  # noqa: E402

'''
NATIVE_MODEL_ID = /monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1
BITBLAS_NATIVE_MODEL_ID = /monster/data/model/opt-125M-autoround-lm_head-false-symTrue
GPU: 4090

(InferenceSpeed.NATIVE_MODEL_ID, BACKEND.MARLIN, 748),
(InferenceSpeed.NATIVE_MODEL_ID, BACKEND.CUDA, 493),
(InferenceSpeed.NATIVE_MODEL_ID, BACKEND.EXLLAMA_V1, 717),
(InferenceSpeed.NATIVE_MODEL_ID, BACKEND.EXLLAMA_V2, 775),
(InferenceSpeed.NATIVE_MODEL_ID, BACKEND.TRITON, 296),
(InferenceSpeed.NATIVE_MODEL_ID, BACKEND.TORCH, 295),
(InferenceSpeed.BITBLAS_NATIVE_MODEL_ID, BACKEND.BITBLAS, 1474),
(InferenceSpeed.NATIVE_MODEL_ID, BACKEND.IPEX, 48),
'''

class TestInferenceSpeed(InferenceSpeed):

    @parameterized.expand(
        [
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.MARLIN, 224.39),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.CUDA, 50.67),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.EXLLAMA_V1, 221.48),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.EXLLAMA_V2, 225.14),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.TRITON, 163.94),
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.TORCH, 50.65),
            (InferenceSpeed.BITBLAS_NATIVE_MODEL_ID, BACKEND.BITBLAS, 1129.44), # Second time running bitblas, there is cache
        ]
    )
    def test_inference_speed(self, model_path, backend, tokens_per_second):
        # There are differences between the results of the first and second runs of bitblas
        # (there is a cache when running bitblas for the second time),
        # so only the results of the second run of bitblas are asserted.
        # The first run of bitblas only prints relevant information
        self.inference(model_path=model_path, backend=backend, tokens_per_second=tokens_per_second, compile=True, warmup_runs=1)
