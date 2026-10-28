# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

from gptqmodel import BACKEND
from model_test import ModelTest


class TestFlint(ModelTest):
    NATIVE_MODEL_ID = os.getenv("FLINT_MODEL_ID", "RecursiveSI/flint")
    TRUST_REMOTE_CODE = False
    # Torch keeps this model test usable on platforms without Marlin.
    QUANT_BACKEND = BACKEND.TORCH
    LOAD_BACKEND = BACKEND.TORCH
    _TINY_CHECKPOINT = NATIVE_MODEL_ID != "RecursiveSI/flint"
    _EVAL_TASKS = {
        "arc_challenge": {
            "acc": {"value": 0.32337883959044367, "floor_pct": 0.04},
            "acc_norm": {"value": 0.34982935153583616, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_SLOW = None if _TINY_CHECKPOINT else _EVAL_TASKS
    EVAL_TASKS_FAST = None if _TINY_CHECKPOINT else ModelTest.derive_fast_eval_tasks(_EVAL_TASKS)

    def test_flint(self):
        self.quantize_and_evaluate()
