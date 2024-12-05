import os
from typing import List, Union

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import unittest
from parameterized import parameterized
from pathlib import Path
import shutil

from gptqmodel.utils import EVAL, LM_EVAL_TASK, EVALPLUS_TASK
from gptqmodel import GPTQModel


class TestEval(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"

    @parameterized.expand(
        [
            # (EVAL.LM_EVAL, [LM_EVAL_TASK.ARC_CHALLENGE]),
            (EVAL.EVALPLUS, [EVALPLUS_TASK.HUMANEVAL])
        ]
    )
    def test_eval(self, eval_backend: EVAL, tasks: Union[List[LM_EVAL_TASK], List[EVALPLUS_TASK]]):
        results = GPTQModel.eval(self.MODEL_ID, backend=eval_backend, tasks=tasks, model_backend="gptqmodel")

        if eval_backend == EVAL.LM_EVAL:
            pass
        elif eval_backend == EVAL.EVALPLUS:
            result = results.get(tasks[0])
            base_formatted, plus_formatted, result_path = result.get("base tests"), result.get("base + extra tests"), result.get("results_path")
            full_path = Path(result_path)
            root_dir = full_path.parts[0]
            shutil.rmtree(root_dir)
            self.assertEqual(base_formatted, "0.299", "Base score does not match expected result")
            self.assertEqual(plus_formatted, "0.274", "Plus score does not match expected result")





