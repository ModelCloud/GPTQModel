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

# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestVerifyHashFunction(unittest.TestCase):
    MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
    EXPECTED_MD5_HASH = "md5:7725c72bc217bcb57b3f1f31d452d871"
    EXPECTED_SHA256_HASH = "sha256:2680bb4d5c977ee54f25dae584665641ea887e7bd8e8d7197ce8ffd310e93f2f"


    def test_verify_md5_hash_function(self):
        # Load the model with MD5 verify_hash parameter
        model = GPTQModel.load(self.MODEL_ID, device="cuda:0", backend=BACKEND.MARLIN,
                                         verify_hash=self.EXPECTED_MD5_HASH)
        self.assertIsNotNone(model)

    def test_verify_sha256_hash_function(self):
        # Load the model with SHA-256 verify_hash parameter
        model = GPTQModel.load(self.MODEL_ID, device="cuda:0", backend=BACKEND.MARLIN,
                                         verify_hash=self.EXPECTED_SHA256_HASH)
        # Add additional checks to ensure the model is loaded correctly
        self.assertIsNotNone(model)

