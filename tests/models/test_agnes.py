# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest


# Native BF16 baseline measured with Evalution 0.0.10 on the complete
# ARC-Challenge test split (1,172 samples), raw prompts, and eager attention.
class TestAgnes(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Agnes-3.0-Flash"
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    GROUP_SIZE = 32
    HESSIAN_CHUNK_SIZE = 256 * 1024 * 1024
    DATASET_CONCAT_SIZE = 2048
    EVAL_BATCH_SIZE = 16

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.5716723549488054, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5861774744027304, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    # Four layers cover the three delta layers and first global layer in the
    # first repeated Agnes hybrid block during fast compatibility runs.
    MODEL_COMPAT_FAST_LAYER_COUNT = 4
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_agnes(self):
        self.quantize_and_evaluate()


__all__ = ["TestAgnes"]
