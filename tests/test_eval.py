# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import tempfile

import pytest
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from tests.eval import evaluate, get_eval_task_metrics  # noqa: E402


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

pytestmark = [pytest.mark.model, pytest.mark.slow]


class TestEval(ModelTest):
    @classmethod
    def setUpClass(cls):
        cls.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"
        cls.model = GPTQModel.load(cls.MODEL_ID)

    @parameterized.expand(
        [
            ("arc_challenge", False),
            ("mmlu_pro:stem.math", True),
        ]
    )
    def test_evalution_gptqmodel(self, task: str, apply_chat_template: bool):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = evaluate(
                model_or_id_or_path=self.MODEL_ID,
                tasks=[task],
                batch_size=1,
                output_path=f"{tmp_dir}/result.json",
                apply_chat_template=apply_chat_template,
                suite_kwargs={"max_rows": 2, "num_fewshot": 1},
            )
            if task == "mmlu_pro:stem.math":
                metrics = get_eval_task_metrics(results, "mmlu_pro_stem_math")
            else:
                metrics = get_eval_task_metrics(results, task)
            self.assertTrue(metrics, f"Expected Evalution metrics for task {task}")
