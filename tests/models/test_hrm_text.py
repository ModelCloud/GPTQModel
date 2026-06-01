# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest


class TestHRMText(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/HRM-Text-1B" # sapientinc/HRM-Text-1B
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": {"A100": 0.3234}, "floor_pct": 0.04},
            "acc_norm": {"value": {"A100": 0.3208}, "floor_pct": 0.04},
        },
        "mmlu_stem": {
            "acc": {"value": {"A100": 0.2743}, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"


    def test_hrm_text(self):
        self.quantize_and_evaluate()
