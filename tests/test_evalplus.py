# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
import unittest  # noqa: E402

from gptqmodel.utils.eval import evalplus  # noqa: E402


class TestEvalplus(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1"

    def test_evalplus(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = f"{tmp_dir}/result.json"
            base_formatted, plus_formatted, _ = evalplus(model=self.MODEL_ID, dataset='humaneval', output_file=output_file)
            self.assertGreaterEqual(float(base_formatted), 0.27, "Base score does not match expected result")
            self.assertGreaterEqual(float(plus_formatted), 0.24, "Plus score does not match expected result")


