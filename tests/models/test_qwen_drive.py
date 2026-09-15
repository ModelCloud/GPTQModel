# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest


class TestQwenDrive(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen-Drive-1.0-4B" # Qwen/Qwen-Drive-1.0-4B
    TRUST_REMOTE_CODE = False
    GROUP_SIZE = 32
    HESSIAN_CHUNK_SIZE = 256 * 1024 * 1024
    DATASET_CONCAT_SIZE = 2048
    EVAL_BATCH_SIZE = 32
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5315699658703071, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5349829351535836, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_qwen_drive(self):
        self.quantize_and_evaluate()


__all__ = ["TestQwenDrive"]
