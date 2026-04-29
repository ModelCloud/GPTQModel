# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/stablelm-base-alpha-3b"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.2363, "floor_pct": 0.2},
            "acc_norm": {"value": 0.2577, "floor_pct": 0.2},
        },
    }
    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "acc": {"value": 0.23720136518771331, "floor_pct": 0.2, "ceil_pct": 1.0},
            "acc_norm": {"value": 0.26023890784982934, "floor_pct": 0.2, "ceil_pct": 1.0},
        },
    }
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6

    def test_stablelm(self):
        self.quantize_and_evaluate()
