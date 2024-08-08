# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
import unittest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from lm_eval.utils import make_table


class TestLmEval(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def test_lm_eval(self):
       with tempfile.TemporaryDirectory() as tmp_dir:
           model_args = f"pretrained={self.MODEL_ID},parallelize={False},device_map={'auto'},gptqmodel={True}"
           results = GPTQModel.lm_eval(
                model_path=self.MODEL_ID,
                model_args=model_args,
                output_path=tmp_dir,
                tasks='arc_challenge',
            )

           print('--------Eval Result---------')
           print(make_table(results))
           if "groups" in results:
               print(make_table(results, "groups"))
           print('--------Eval Result End---------')

           stored_files = os.listdir(tmp_dir)
           self.assertTrue(len(stored_files) > 0)

