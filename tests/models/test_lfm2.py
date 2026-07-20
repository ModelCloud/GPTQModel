# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestLFM2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/LFM2.5-1.2B-Instruct" # "LiquidAI/LFM2.5-1.2B-Instruct"
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    GROUP_SIZE = 32
    DATASET_SIZE = 512
    EVAL_BATCH_SIZE = 4
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3780,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3968,
                "floor_pct": 0.04,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_lfm2(self):
        self.quantize_and_evaluate()
