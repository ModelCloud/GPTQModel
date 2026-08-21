# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel import BACKEND
from model_test import ModelTest


class TestOlmo3(ModelTest):
    NATIVE_MODEL_ID = "allenai/Olmo-3-1025-7B"
    TRUST_REMOTE_CODE = False
    DATASET_SIZE_FAST = 128
    EVAL_BATCH_SIZE = 32
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {
                "value": 0.48293515358361777,
                "floor_pct": 0.10,
            },
            "acc_norm": {
                "value": 0.5196245733788396,
                "floor_pct": 0.10,
            },
        },
    }
    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {
                "value": 0.46875,
                "floor_pct": 0.25,
            },
            "acc_norm": {
                "value": 0.59375,
                "floor_pct": 0.25,
            },
        },
    }
    LOAD_BACKEND = BACKEND.AUTO
    USE_FLASH_ATTN = False
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_olmo3(self):
        self.quantize_and_evaluate()
