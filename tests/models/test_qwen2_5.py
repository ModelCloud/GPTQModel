# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest
from gptqmodel.utils.eval import EVAL


class TestQwen2_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.2705, "floor_pct": 0.05},
            "acc_norm": {"value": 0.3063, "floor_pct": 0.05},
        },
    }
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = True
    #EVAL_BATCH_SIZE = 6

    def test_qwen2_5(self):
        self.quant_lm_eval()
