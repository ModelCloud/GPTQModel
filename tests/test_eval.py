# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import tempfile  # noqa: E402
from typing import (
    Type,  # noqa: E402
    Union,  # noqa: E402
)

import pytest  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.utils.eval import EVAL, evaluate, get_eval_task_metrics  # noqa: E402


pytestmark = [pytest.mark.model, pytest.mark.slow]


class TestEval(ModelTest):
    @classmethod
    def setUpClass(cls):
        cls.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"
        cls.model = GPTQModel.load(cls.MODEL_ID)

    @parameterized.expand(
        [
            (EVAL.LM_EVAL, EVAL.LM_EVAL.ARC_CHALLENGE, 'gptqmodel'),
            (EVAL.EVALPLUS, EVAL.EVALPLUS.HUMAN, 'gptqmodel'),
            (EVAL.EVALPLUS, EVAL.EVALPLUS.HUMAN, 'vllm'),
        ]
    )
    def test_eval_gptqmodel(self, framework: Union[Type[EVAL.LM_EVAL],Type[EVAL.EVALPLUS]], task: Union[EVAL.LM_EVAL, EVAL.EVALPLUS], llm_backend: str):
        if llm_backend == "vllm":
            try:
                import vllm._C  # noqa: F401,E402
            except Exception as exc:
                self.skipTest(f"vllm runtime unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/result.json"
            model_args = {}
            if task == EVAL.LM_EVAL.GPQA:
                model_args["gpu_memory_utilization"]=0.7

            results = evaluate(model_or_id_or_path=self.MODEL_ID,
                               framework=framework,
                               tasks=[task],
                               batch_size=1,
                               output_path=output_path,
                               llm_backend=llm_backend,
                               model_args=model_args,
                               )

            if framework == EVAL.LM_EVAL:
                metrics = get_eval_task_metrics(results, task)
                acc_score = metrics.get("accuracy,loglikelihood")
                acc_norm_score = metrics.get("accuracy,loglikelihood_norm")

                self.assertGreaterEqual(acc_score, 0.279, "acc score does not match expected result")
                self.assertGreaterEqual(acc_norm_score, 0.30, "acc_norm score does not match expected result")
            elif framework == EVAL.EVALPLUS:
                result = results.get(task.value)
                base_formatted, plus_formatted, _ = float(result.get("base tests")), float(
                    result.get("base + extra tests")), result.get("results_path")
                self.assertGreaterEqual(base_formatted, 0.26, "Base score does not match expected result")
                self.assertGreaterEqual(plus_formatted, 0.23, "Plus score does not match expected result")
