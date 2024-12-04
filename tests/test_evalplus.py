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
        self.MODEL_ID = "ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"

    def test_evalplus(self):
        base_formatted, plus_formatted, result_path = evalplus(model=self.MODEL_ID, dataset='humaneval', backend='gptqmodel')
        self.assertEqual(base_formatted, "0.299", "Base score does not match expected result")
        self.assertEqual(plus_formatted, "0.274", "Plus score does not match expected result")

        full_path = Path(result_path)
        root_dir = full_path.parts[0]
        shutil.rmtree(root_dir)


