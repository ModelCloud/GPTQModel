# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel import BACKEND


class TestApertus(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Apertus-8B-Instruct-2509/"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": {"A100": 0.5136, "RTX4090": 0.5136}, "floor_pct": 0.20},
            "acc_norm": {"value": {"A100": 0.5085, "RTX4090": 0.5059}, "floor_pct": 0.20},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6
    LOAD_BACKEND = BACKEND.TORCH

    def test_apertus(self):
        self.quantize_and_evaluate()
