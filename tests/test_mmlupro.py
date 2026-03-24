# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import tempfile
import unittest

from tests.eval import evaluate, get_eval_task_metrics


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class TestMMLUPRO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/QwQ-32B-4bit-gp32-verify-data"


    def test_mmlupro(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = evaluate(
                self.MODEL_ID,
                tasks="mmlu_pro:math",
                output_path=f"{tmp_dir}/result.json",
                batch_size=2,
                suite_kwargs={"num_fewshot": 1, "max_rows": 2},
            )
            metrics = get_eval_task_metrics(result, "mmlu_pro:math")
            self.assertTrue(metrics)
