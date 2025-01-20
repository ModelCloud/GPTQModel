# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest
from typing import Union

from gptqmodel import GPTQModel
from gptqmodel.utils import EVAL
from parameterized import parameterized

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class TestEval(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1"

    @parameterized.expand(
        [
            (EVAL.LM_EVAL, EVAL.LM_EVAL.ARC_CHALLENGE, 'gptqmodel'),
            (EVAL.EVALPLUS, EVAL.EVALPLUS.HUMAN, 'gptqmodel'),
            (EVAL.LM_EVAL, EVAL.LM_EVAL.ARC_CHALLENGE, 'vllm'),
            (EVAL.EVALPLUS, EVAL.EVALPLUS.HUMAN, 'vllm'),
            (EVAL.LM_EVAL, EVAL.LM_EVAL.GPQA, 'vllm'),
        ]
    )
    def test_eval_gptqmodel(self, eval_backend: EVAL, task: Union[EVAL.LM_EVAL, EVAL.EVALPLUS], backend: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = f"{tmp_dir}/result.json"
            extra_model_args = ""
            if task == EVAL.LM_EVAL.GPQA:
                extra_model_args = "gpu_memory_utilization=0.7"

            results = GPTQModel.eval(self.MODEL_ID, framework=eval_backend, tasks=[task], batch=32,
                                     output_file=output_file, backend=backend, extra_model_args=extra_model_args)

            if eval_backend == EVAL.LM_EVAL:
                if task == EVAL.LM_EVAL.GPQA:
                    gpqa_main_n_shot = results['results'].get('gpqa_main_n_shot', {}).get('acc,none')
                    gpqa_main_zeroshot = results['results'].get('gpqa_main_zeroshot', {}).get('acc,none')

                    self.assertGreaterEqual(gpqa_main_n_shot, 0.21, "acc score does not match expected result")
                    self.assertGreaterEqual(gpqa_main_zeroshot, 0.25, "acc_norm score does not match expected result")
                else:
                    acc_score = results['results'].get(task.value, {}).get('acc,none')
                    acc_norm_score = results['results'].get(task.value, {}).get('acc_norm,none')

                    self.assertGreaterEqual(acc_score, 0.28, "acc score does not match expected result")
                    self.assertGreaterEqual(acc_norm_score, 0.32, "acc_norm score does not match expected result")
            elif eval_backend == EVAL.EVALPLUS:
                result = results.get(task.value)
                base_formatted, plus_formatted, _ = float(result.get("base tests")), float(
                    result.get("base + extra tests")), result.get("results_path")
                self.assertGreaterEqual(base_formatted, 0.27, "Base score does not match expected result")
                self.assertGreaterEqual(plus_formatted, 0.24, "Plus score does not match expected result")





