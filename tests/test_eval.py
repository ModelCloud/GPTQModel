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
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    @parameterized.expand(
        [
            # (EVAL.LM_EVAL, [LM_EVAL_TASK.ARC_CHALLENGE]),
            (EVAL.EVALPLUS, [EVALPLUS_TASK.HUMAN])
        ]
    )
    def test_eval(self, eval_backend: EVAL, tasks: Union[List[LM_EVAL_TASK], List[EVALPLUS_TASK]]):
        results = GPTQModel.eval(self.MODEL_ID, backend=eval_backend, tasks=tasks, model_backend="hf")

        if eval_backend == EVAL.LM_EVAL:
            pass
        elif eval_backend == EVAL.EVALPLUS:
            result = results.get(tasks[0])
            base_formatted, plus_formatted, result_path = result.get("base tests"), result.get("base + extra tests"), result.get("results_path")
            self.assertGreaterEqual(base_formatted, "0.34", "Base score does not match expected result")
            self.assertGreaterEqual(plus_formatted, "0.29", "Plus score does not match expected result")
            full_path = Path(result_path)
            root_dir = full_path.parts[0]
            shutil.rmtree(root_dir)





