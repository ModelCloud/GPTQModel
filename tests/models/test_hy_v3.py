# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestHyV3(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Hy-MT2-30B-A3B" # tencent/Hy-MT2-30B-A3B
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.5324, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5341, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_hy_v3(self):
        self.quantize_and_evaluate()
