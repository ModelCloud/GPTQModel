# -- do not touch
import os

# -- end do not touch
import tempfile  # noqa: E402
import unittest  # noqa: E402

from lm_eval.utils import make_table  # noqa: E402

from gptqmodel.utils.eval import lm_eval  # noqa: E402


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class TestLmEval(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1"
        self.random_seed = 1234
        self.task = 'arc_challenge'

    def test_lm_eval(self):
       with tempfile.TemporaryDirectory() as tmp_dir:
           results = lm_eval(
                model_name='hf',
                model_args=f'pretrained={self.MODEL_ID},gptqmodel=True',
                apply_chat_template=True,
                output_path=tmp_dir,
                tasks=self.task,
                numpy_random_seed=self.random_seed,
                torch_random_seed=self.random_seed,
                fewshot_random_seed=self.random_seed
            )

           print('--------lm_eval Eval Result---------')
           print(make_table(results))
           if "groups" in results:
               print(make_table(results, "groups"))
           print('--------lm_eval Result End---------')

           acc_score = results['results'].get(self.task, {}).get('acc,none')
           acc_norm_score = results['results'].get(self.task, {}).get('acc_norm,none')

           self.assertGreaterEqual(acc_score, 0.28, "acc score does not match expected result")
           self.assertGreaterEqual(acc_norm_score, 0.32, "acc_norm score does not match expected result")

