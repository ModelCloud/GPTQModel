# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestPhi_3(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Phi-3-mini-4k-instruct" # "microsoft/Phi-3-mini-4k-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.5401
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5674
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = 0.5477815699658704
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = 0.5742320819112628
    TRUST_REMOTE_CODE = True
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": NATIVE_ARC_CHALLENGE_ACC},
            "acc_norm": {"value": NATIVE_ARC_CHALLENGE_ACC_NORM},
        },
    }
    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": NATIVE_ARC_CHALLENGE_ACC_FAST, "ceil_pct": 1.0},
            "acc_norm": {"value": NATIVE_ARC_CHALLENGE_ACC_NORM_FAST, "ceil_pct": 1.0},
        },
    }

    def test_phi_3(self):
        self.quantize_and_evaluate()
