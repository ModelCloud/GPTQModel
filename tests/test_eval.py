import os
import tempfile
import unittest
from typing import Union

from gptqmodel import GPTQModel
from gptqmodel.utils import EVAL
from parameterized import parameterized

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TestEval(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1"

    @parameterized.expand(
        [
            (EVAL.LM_EVAL, EVAL.LM_EVAL.ARC_CHALLENGE, 'gptqmodel'),
            (EVAL.EVALPLUS, EVAL.EVALPLUS.HUMAN, 'gptqmodel'),
            (EVAL.LM_EVAL, EVAL.LM_EVAL.ARC_CHALLENGE, 'vllm'),
            (EVAL.EVALPLUS, EVAL.EVALPLUS.HUMAN, 'vllm')
        ]
    )
    def test_eval_gptqmodel(self, eval_backend: EVAL, task: Union[EVAL.LM_EVAL, EVAL.EVALPLUS], backend: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = f"{tmp_dir}/result.json"
            results = GPTQModel.eval(self.MODEL_ID, framework=eval_backend, tasks=[task], batch=32,
                                     output_file=output_file, backend=backend)
            if eval_backend == EVAL.LM_EVAL:
                acc_score = results['results'].get(task.value, {}).get('acc,none')
                acc_norm_score = results['results'].get(task.value, {}).get('acc_norm,none')

                self.assertGreaterEqual(acc_score, 0.28, "acc score does not match expected result")
                self.assertGreaterEqual(acc_norm_score, 0.32, "acc_norm score does not match expected result")
            elif eval_backend == EVAL.EVALPLUS:
                result = results.get(task.value)
                base_formatted, plus_formatted, _ = float(result.get("base tests")), float(
                    result.get("base + extra tests")), result.get("results_path")
                self.assertGreaterEqual(base_formatted, 0.27, "Base score does not match expected result")
                self.assertGreaterEqual(plus_formatted, 0.24, "Plus score does not match expected result")





