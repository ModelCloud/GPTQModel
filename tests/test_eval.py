import os
import unittest
from typing import Union

from gptqmodel import GPTQModel
from gptqmodel.utils import EVAL, EVALPLUS_TASK, LM_EVAL_TASK
from parameterized import parameterized
import tempfile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = f"{tmp_dir}/result.json"
            results = GPTQModel.eval(self.MODEL_ID, framework=eval_backend, tasks=[task], batch=32, output_file=output_file)
            if eval_backend == EVAL.LM_EVAL:
                acc_score = results['results'].get(task.value, {}).get('acc,none')
                acc_norm_score = results['results'].get(task.value, {}).get('acc_norm,none')

                self.assertGreaterEqual(acc_score, 0.31, "acc score does not match expected result")
                self.assertGreaterEqual(acc_norm_score, 0.35, "acc_norm score does not match expected result")
            elif eval_backend == EVAL.EVALPLUS:
                result = results.get(task.value)
                base_formatted, plus_formatted, _ = float(result.get("base tests")), float(
                    result.get("base + extra tests")), result.get("results_path")
                self.assertGreaterEqual(base_formatted, 0.29, "Base score does not match expected result")
                self.assertGreaterEqual(plus_formatted, 0.26, "Plus score does not match expected result")





