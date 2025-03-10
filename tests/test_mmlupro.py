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

import os
import unittest
import tempfile

from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class TestMMLUPRO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/QwQ-32B-4bit-gp32-verify-data"


    def test_mmlupro(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = GPTQModel.eval(self.MODEL_ID, framework=EVAL.MMLUPRO, tasks=EVAL.MMLUPRO.MATH, output_path=tmp_dir, batch_size=1, ntrain=1)
