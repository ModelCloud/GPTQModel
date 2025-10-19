# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestDeci(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/DeciLM-7B-instruct" # "Deci/DeciLM-7B-instruct"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5239, "floor_pct": 0.8},
            "acc_norm": {"value": 0.5222, "floor_pct": 0.8},
        },
    }
    TRUST_REMOTE_CODE = True
    USE_VLLM = False
    EVAL_BATCH_SIZE = 6

    def test_deci(self):
        self.quant_lm_eval()
