# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest


class TestQwen3(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-4B" # Qwen/Qwen3-4B
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.6092, "floor_pct": 0.04},
            "acc_norm": {"value": 0.6143, "floor_pct": 0.04},
        },
        "mmlu_stem": {
            "acc": {"value": 0.8461, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"


    def test_qwen3(self):
        self.quant_lm_eval()
