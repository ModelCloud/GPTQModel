# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestZamba2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Zamba2-1.2B-Instruct-v2" # Zyphra/Zamba2-1.2B-Instruct-v2
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5026, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5171, "floor_pct": 0.04},
        },
        "gsm8k_platinum_cot": {
            "acc": {"value": 0.6362, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    # MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_zamba2(self):
        self.quantize_and_evaluate()
