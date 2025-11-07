# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestQwen2_5_Moe(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen1.5-MoE-A2.7B" # Qwen/Qwen1.5-MoE-A2.7B
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": 0.2739, "floor_pct": 0.2},
            "acc_norm": {"value": 0.3055, "floor_pct": 0.2},
        },
    }

    def test_qwen2_5(self):
        self.quant_lm_eval()
