# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import tempfile
import unittest

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
            GPTQModel.eval(self.MODEL_ID, framework=EVAL.MMLU_PRO, tasks=EVAL.MMLU_PRO.MATH, output_path=tmp_dir, batch_size=10, ntrain=5)
