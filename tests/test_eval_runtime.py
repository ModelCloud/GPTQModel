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

from gptqmodel import BACKEND
from tests.eval import evaluate, format_eval_result_table, get_eval_task_metrics  # noqa: E402


class TestEvalRuntime(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"
        self.random_seed = 1234
        self.task = "arc_challenge"

        # self.acc_score = 0.3183
        # self.acc_norm_score = 0.3515


    def test_eval_direct(self):
       with tempfile.TemporaryDirectory() as tmp_dir:
           results = evaluate(
                model_or_id_or_path=self.MODEL_ID,
                apply_chat_template=True,
                output_path=tmp_dir,
                tasks=[self.task],
            )

           print('--------Evalution Eval Result---------')
           print(format_eval_result_table(results))
           print('--------Evalution Result End---------')

           metrics = get_eval_task_metrics(results, self.task)
           acc_norm_score = metrics.get('accuracy,loglikelihood_norm')

           # self.assertGreaterEqual(acc_score, self.acc_score, "acc score does not match expected result")
           self.assertGreaterEqual(acc_norm_score, 0.3400, "acc_norm score does not match expected result")

    def test_eval_path(self):
       with tempfile.TemporaryDirectory() as tmp_dir:
           results = evaluate(
                model_or_id_or_path=self.MODEL_ID,
                backend = BACKEND.EXLLAMA_V2, # for path loading, can override backend
                output_path=tmp_dir,
                tasks=[self.task],
                model_args={
                    "device": "cuda"
                }
            )

           print('--------Evalution Eval Result---------')
           print(format_eval_result_table(results))
           print('--------Evalution Result End---------')

           metrics = get_eval_task_metrics(results, self.task)
           acc_norm_score = metrics.get('accuracy,loglikelihood_norm')

           # self.assertGreaterEqual(acc_score, self.acc_score, "acc score does not match expected result")
           self.assertGreaterEqual(acc_norm_score, 0.3000, "acc_norm score does not match expected result")
