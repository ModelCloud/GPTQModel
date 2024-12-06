import os
import unittest
import tempfile

from gptqmodel.utils.eval import evalplus

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class TestEvalplus(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    def test_evalplus(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = f"{tmp_dir}/result.json"
            base_formatted, plus_formatted, _ = evalplus(model=self.MODEL_ID, dataset='humaneval', output_file=output_file)
            self.assertGreaterEqual(float(base_formatted), 0.29, "Base score does not match expected result")
            self.assertGreaterEqual(float(plus_formatted), 0.26, "Plus score does not match expected result")


