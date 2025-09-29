# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import tempfile  # noqa: E402
import unittest  # noqa: E402

from lm_eval.utils import make_table  # noqa: E402

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.utils.eval import EVAL  # noqa: E402


class TestLmEval(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"
        self.random_seed = 1234
        self.task = EVAL.LM_EVAL.ARC_CHALLENGE

        # self.acc_score = 0.3183
        # self.acc_norm_score = 0.3515


    def test_eval_direct(self):
       with tempfile.TemporaryDirectory() as tmp_dir:
           results = GPTQModel.eval(
                model_or_id_or_path=self.MODEL_ID,
                apply_chat_template=True,
                output_path=tmp_dir,
                tasks=[self.task],
            )

           print('--------lm_eval Eval Result---------')
           print(make_table(results))
           if "groups" in results:
               print(make_table(results, "groups"))
           print('--------lm_eval Result End---------')

           results['results'].get(self.task.value, {}).get('acc,none')
           acc_norm_score = results['results'].get(self.task.value, {}).get('acc_norm,none')

           # self.assertGreaterEqual(acc_score, self.acc_score, "acc score does not match expected result")
           self.assertGreaterEqual(acc_norm_score, 0.3400, "acc_norm score does not match expected result")

    def test_eval_path(self):
       with tempfile.TemporaryDirectory() as tmp_dir:
           results = GPTQModel.eval(
                model_or_id_or_path=self.MODEL_ID,
                backend = BACKEND.EXLLAMA_V2, # for path loading, can override backend
                output_path=tmp_dir,
                tasks=[self.task],
            )

           print('--------lm_eval Eval Result---------')
           print(make_table(results))
           if "groups" in results:
               print(make_table(results, "groups"))
           print('--------lm_eval Result End---------')

           # acc_score = results['results'].get(self.task, {}).get('acc,none')
           acc_norm_score = results['results'].get(self.task, {}).get('acc_norm,none')

           # self.assertGreaterEqual(acc_score, self.acc_score, "acc score does not match expected result")
           self.assertGreaterEqual(acc_norm_score, 0.3000, "acc_norm score does not match expected result")
