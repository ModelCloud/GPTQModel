# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.5154 |
# | arc_challenge :: acc_norm,none |   0.535  |
# | mmlu_stem :: acc,none          |   0.6325 |
class TestGlm(ModelTest):
    GROUP_SIZE = 32
    # real: THUDM/glm-4-9b-chat-hf
    NATIVE_MODEL_ID = "/monster/data/model/glm-4-9b-chat-hf"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5154, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5350, "floor_pct": 0.04},
        },
        "mmlu_stem": {
            "acc": {"value": 0.5810, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_glm(self):
        self.quantize_and_evaluate()
