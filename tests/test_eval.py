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

from lm_eval.tasks import TaskManager  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402


class TestEval(ModelTest):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"
        self.model = GPTQModel.load(self.MODEL_ID)

    @parameterized.expand(
        [
            (EVAL.LM_EVAL, EVAL.LM_EVAL.ARC_CHALLENGE, 'gptqmodel'),
            (EVAL.LM_EVAL, EVAL.LM_EVAL.ARC_CHALLENGE, 'vllm'),
            (EVAL.EVALPLUS, EVAL.EVALPLUS.HUMAN, 'gptqmodel'),
            (EVAL.EVALPLUS, EVAL.EVALPLUS.HUMAN, 'vllm'),
            (EVAL.LM_EVAL, EVAL.LM_EVAL.GPQA, 'vllm'),
        ]
    )
    def test_eval_gptqmodel(self, framework: Union[Type[EVAL.LM_EVAL],Type[EVAL.EVALPLUS]], task: Union[EVAL.LM_EVAL, EVAL.EVALPLUS], llm_backend: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/result.json"
            model_args = {}
            if task == EVAL.LM_EVAL.GPQA:
                model_args["gpu_memory_utilization"]=0.7

            results = GPTQModel.eval(model_or_id_or_path=self.MODEL_ID,
                                     framework=framework,
                                     tasks=[task],
                                     batch_size=1,
                                     output_path=output_path,
                                     llm_backend=llm_backend,
                                     model_args=model_args,
                                     task_manager=TaskManager(include_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks"), include_defaults=False)
                                     )

            if llm_backend == EVAL.LM_EVAL:
                if task == EVAL.LM_EVAL.GPQA:
                    gpqa_main_n_shot = results['results'].get('gpqa_main_n_shot', {}).get('acc,none')
                    gpqa_main_zeroshot = results['results'].get('gpqa_main_zeroshot', {}).get('acc,none')

                    self.assertGreaterEqual(gpqa_main_n_shot, 0.21, "acc score does not match expected result")
                    self.assertGreaterEqual(gpqa_main_zeroshot, 0.25, "acc_norm score does not match expected result")
                else:
                    acc_score = results['results'].get(task.value, {}).get('acc,none')
                    acc_norm_score = results['results'].get(task.value, {}).get('acc_norm,none')

                    self.assertGreaterEqual(acc_score, 0.28, "acc score does not match expected result")
                    self.assertGreaterEqual(acc_norm_score, 0.32, "acc_norm score does not match expected result")
            elif llm_backend == EVAL.EVALPLUS:
                result = results.get(task.value)
                base_formatted, plus_formatted, _ = float(result.get("base tests")), float(
                    result.get("base + extra tests")), result.get("results_path")
                self.assertGreaterEqual(base_formatted, 0.26, "Base score does not match expected result")
                self.assertGreaterEqual(plus_formatted, 0.23, "Plus score does not match expected result")
