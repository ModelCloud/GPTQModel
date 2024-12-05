import os
from pathlib import Path
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import unittest

from gptqmodel.utils.evalplus import evalplus

class TestEvalplus(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    def test_evalplus(self):
        base_formatted, plus_formatted, result_path = evalplus(model=self.MODEL_ID, dataset='humaneval', backend='hf')
        self.assertGreaterEqual(base_formatted, "0.34", "Base score does not match expected result")
        self.assertGreaterEqual(plus_formatted, "0.29", "Plus score does not match expected result")
        full_path = Path(result_path)
        root_dir = full_path.parts[0]
        shutil.rmtree(root_dir)


