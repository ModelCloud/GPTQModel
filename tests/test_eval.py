import os
from typing import Union

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import unittest
from parameterized import parameterized

from gptqmodel.utils import EVAL, LM_EVAL_TASK, EVALPLUS_TASK
from gptqmodel import GPTQModel


class TestEval(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    @parameterized.expand(
        [
            (EVAL.LM_EVAL, LM_EVAL_TASK.ARC_CHALLENGE),
            (EVAL.EVALPLUS, EVALPLUS_TASK.HUMAN)
        ]
    )
    def test_eval(self, eval_backend: EVAL, task: Union[LM_EVAL_TASK, EVALPLUS_TASK]):
        results = GPTQModel.eval(self.MODEL_ID, backend=eval_backend, tasks=[task], model_backend="hf")

        if eval_backend == EVAL.LM_EVAL:
            task_results = {
                metric: value for metric, value in results['results'].get(task.value, {}).items()
                if metric != 'alias' and 'stderr' not in metric
            }

            acc_key = "acc"
            acc_norm_key = "acc_norm"
            acc_score = 0
            acc_norm_score = 0

            for key, value in task_results.items():
                if acc_key in key and acc_norm_key not in key:
                    acc_score = float(value)
                elif acc_norm_key in key:
                    acc_norm_score = float(value)

            self.assertGreaterEqual(acc_score, 0.31, f"{acc_key} score does not match expected result")
            self.assertGreaterEqual(acc_norm_score, 0.35, f"{acc_norm_key} score does not match expected result")
        elif eval_backend == EVAL.EVALPLUS:
            result = results.get(task)
            base_formatted, plus_formatted, _ = float(result.get("base tests")), float(result.get("base + extra tests")), result.get("results_path")
            self.assertGreaterEqual(base_formatted, 0.34, "Base score does not match expected result")
            self.assertGreaterEqual(plus_formatted, 0.29, "Plus score does not match expected result")





