# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest

from gptqmodel import BACKEND


class TestHymba(ModelTest):
    NATIVE_MODEL_ID = "nvidia/Hymba-1.5B-Instruct"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": {"A100": 0.3737}, "floor_pct": 0.75},
            "acc_norm": {"value": {"A100": 0.3703}, "floor_pct": 0.75},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_MAX_LEN = 8192
    TRUST_REMOTE_CODE = False
    # Preserve the published checkpoint's existing evaluation recipe.
    EVAL_BATCH_SIZE = 1

    # Keep the existing no-act-order calibration recipe.
    DESC_ACT = False
    OFFLOAD_TO_DISK = False  # trained-checkpoint/offload integration is not qualified
    LOAD_BACKEND = BACKEND.AUTO
    USE_FLASH_ATTN = False  # package-owned SDPA/portable SSM reference implementation


    def test_hymba(self):
        self.quantize_and_evaluate()
