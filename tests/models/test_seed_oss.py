# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestSeedOSS(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Seed-OSS-36B-Instruct/"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.2739, "floor_pct": 0.2},
            "acc_norm": {"value": 0.3055, "floor_pct": 0.2},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6

    def test_seed_oss(self):
        self.quantize_and_evaluate()
