# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestYi(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Yi-Coder-1.5B-Chat" # "01-ai/Yi-Coder-1.5B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.2679
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2986
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = 0.24232081911262798
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = 0.2781569965870307
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 4
    APPLY_CHAT_TEMPLATE = True
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

    def test_yi(self):
        self.quantize_and_evaluate()
