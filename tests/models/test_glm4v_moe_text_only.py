# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from gptqmodel import BACKEND
from model_test import ModelTest


class TestGlm4VTextOnly(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/GLM-4.5V-text-only"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5887, "floor_pct": 0.04},
            "acc_norm": {"value": 0.6100, "floor_pct": 0.04},
        },
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.3871,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    LOAD_BACKEND = BACKEND.AUTO


    def test_glm4_moe_txt(self):
        # self.quantize_and_evaluate()

        self.evaluate_model(self.SAVE_PATH)
