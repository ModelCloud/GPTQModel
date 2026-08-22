# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel import BACKEND
from model_test import ModelTest


class TestSmolLM3(ModelTest):
    NATIVE_MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
    TRUST_REMOTE_CODE = False
    DATASET_SIZE_FAST = 128
    EVAL_BATCH_SIZE = 32
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc_norm": {
                "value": 0.538,
                "floor_pct": 0.10,
                "ceil_pct": 0.10,
            },
        },
    }
    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {
                "value": 0.40625,
                "floor_pct": 0.25,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.5,
                "floor_pct": 0.25,
                "ceil_pct": 1.0,
            },
        },
    }
    LOAD_BACKEND = BACKEND.AUTO
    USE_FLASH_ATTN = False
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_smollm3(self):
        self.quantize_and_evaluate()
