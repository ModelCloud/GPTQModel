# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGraniteSWA(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/granite-swash-2b"  # "ibm-granite/granite-swash-2b"
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.4411, "floor_pct": 0.04},
            "acc_norm": {"value": 0.4590, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_granite_swa(self):
        self.quantize_and_evaluate()
