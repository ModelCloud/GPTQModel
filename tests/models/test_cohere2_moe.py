# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest

from gptqmodel.quantization.config import ExpertsRoutingOverride, Fallback, MoEConfig, VramStrategy
from gptqmodel.utils.backend import BACKEND


class TestCohere2Moe(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/North-Mini-Code-1.0" # CohereLabs/North-Mini-Code-1.0
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5623, "floor_pct": 0.15},
            "acc_norm": {"value": 0.5853, "floor_pct": 0.15},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    EVAL_BATCH_SIZE = 4
    USE_FLASH_ATTN = False

    def test_cohere2_moe(self):
        self.quantize_and_evaluate()
