# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestNanbeige42(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Nanbeige4.2-3B" # Nanbeige/Nanbeige4.2-3B
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 8
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.31399317406143346, "floor_pct": 0.10},
            "acc_norm": {"value": 0.3447098976109215, "floor_pct": 0.10},
        },
    }
    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.3242320819112628, "floor_pct": 0.10, "ceil_pct": 1.0},
            "acc_norm": {
                "value": 0.3506825938566553,
                "floor_pct": 0.10,
                "ceil_pct": 1.0,
            },
        },
    }

    def test_nanbeige42(self):
        self.quantize_and_evaluate()
