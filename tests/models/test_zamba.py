# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestZamba(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Zamba-7B-v1" # Zyphra/Zamba-7B-v1
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": {"A100": 0.4599}, "floor_pct": 0.04},
            "acc_norm": {"value": {"A100": 0.4778}, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    USE_FLASH_ATTN = False

    def test_zamba(self):
        self.quantize_and_evaluate()
