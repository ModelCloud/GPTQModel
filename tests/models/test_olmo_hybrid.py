# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel import BACKEND


class TestOlmoHybrid(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Olmo-Hybrid-7B"
    TRUST_REMOTE_CODE = False
    DATASET_SIZE_FAST = 128
    EVAL_BATCH_SIZE = 32
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {
                "value": 0.5307167235494881,
                "floor_pct": 0.04,
                "ceil_pct": 0.10,
            },
            "acc_norm": {
                "value": 0.5597269624573379,
                "floor_pct": 0.04,
                "ceil_pct": 0.10,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    LOAD_BACKEND = BACKEND.AUTO
    USE_FLASH_ATTN = False
    # The first full-attention block follows three DeltaNet blocks.
    MODEL_COMPAT_FAST_LAYER_COUNT = 4
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    # The hybrid DeltaNet reference forward is stateful; keep this 7B smoke
    # test on one device so calibration and evaluation share one allocator.
    DENSE_VRAM_STRATEGY_DEVICES = ["cuda:0"]

    def test_olmo_hybrid(self):
        self.quantize_and_evaluate()
