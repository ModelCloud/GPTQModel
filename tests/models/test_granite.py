# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGranite(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/granite-3.0-2b-instruct" # "ibm-granite/granite-3.0-2b-instruct"
    TRUST_REMOTE_CODE = True
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.4505, "floor_pct": 0.2},
            "acc_norm": {"value": 0.4770, "floor_pct": 0.2},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_granite(self):
        self.quantize_and_evaluate()
