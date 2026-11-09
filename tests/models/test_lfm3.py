# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestLFM3(ModelTest):
    # LFM3 ships as a remote-code (`auto_map`) checkpoint on both the 4.x and 5.x
    # Transformers releases, so the definition always needs `trust_remote_code`.
    NATIVE_MODEL_ID = "/monster/data/models/LFM3-1.7B"  # LiquidAI/LFM3-1.7B
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    GROUP_SIZE = 32
    DATASET_SIZE = 512
    EVAL_BATCH_SIZE = 4
    # The tri-hybrid stack starts with `conv` layers; keep the fast-mode prefix so
    # the conv, full-attention and sliding-attention modules are all covered.
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3268,
                "floor_pct": 0.05,
            },
            "acc_norm": {
                "value": 0.3396,
                "floor_pct": 0.05,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_lfm3(self):
        self.quantize_and_evaluate()
