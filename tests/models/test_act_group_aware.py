# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestHybridActOrder(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": 0.3140, "floor_pct": 0.05},
            "acc_norm": {"value": 0.3439, "floor_pct": 0.05},
        },
    }
    GPTQA = False
    ACT_GROUP_AWARE = True

    def test_llama3_2(self):
        self.quant_lm_eval()
