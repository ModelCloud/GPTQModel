# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest


# Native BF16 baseline measured with Evalution 0.0.10 on the complete
# ARC-Challenge test split (1,172 samples), raw prompts, and eager attention.
class TestAgnes(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Agnes-3.0-Flash" # Agnes-AI/Agnes-3.0-Flash
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 1

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.5725, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5853, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_SINGLE_GPU = False

    def test_agnes(self):
        self.quantize_and_evaluate()


__all__ = ["TestAgnes"]
