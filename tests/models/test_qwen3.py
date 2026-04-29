# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest


class TestQwen3(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-4B" # Qwen/Qwen3-4B
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": {"A100": 0.5094}, "floor_pct": 0.04},
            "acc_norm": {"value": {"A100": 0.5145}, "floor_pct": 0.04},
        },
        "mmlu_stem": {
            "acc": {"value": {"A100": 0.7101}, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"


    def test_qwen3(self):
        self.quantize_and_evaluate()
