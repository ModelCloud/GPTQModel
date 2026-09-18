# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math

from model_test import ModelTest


class TestZDTaichu5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/ZDTaichu5.0-9B" # TaichuAI/ZDTaichu5.0-9B
    TRUST_REMOTE_CODE = True
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": {"A100": 0.5427}, "floor_pct": 0.04},
            "acc_norm": {"value": {"A100": 0.5631}, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 8

    def test_zamba2(self):
        self.quantize_and_evaluate()
