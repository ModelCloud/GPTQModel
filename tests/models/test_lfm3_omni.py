# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestLFM3Omni(ModelTest):
    # Only the text decoder is quantized; the audio encoder, depth decoder and
    # projectors are materialized in their checkpoint dtype.
    NATIVE_MODEL_ID = "/monster/data/models/LFM3-Omni-2B"  # LiquidAI/LFM3-Omni-2B
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    QUANT_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 4
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3976,
                "floor_pct": 0.05,
            },
            "acc_norm": {
                "value": 0.4113,
                "floor_pct": 0.05,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_lfm3_omni(self):
        self.quantize_and_evaluate()
