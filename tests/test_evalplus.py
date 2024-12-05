import os
import unittest

from gptqmodel.utils.eval import evalplus

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class TestEvalplus(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    def test_evalplus(self):
        base_formatted, plus_formatted, _ = evalplus(model=self.MODEL_ID, dataset='humaneval')
        self.assertGreaterEqual(float(base_formatted), 0.31, "Base score does not match expected result")
        self.assertGreaterEqual(float(plus_formatted), 0.29, "Plus score does not match expected result")


