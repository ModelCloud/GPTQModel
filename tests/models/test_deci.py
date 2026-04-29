# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestDeci(ModelTest):
    """Compat coverage for Deci remote code through quantize, save, reload, and eval."""

    NATIVE_MODEL_ID = "/monster/data/model/DeciLM-7B-instruct" # "Deci/DeciLM-7B-instruct"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5239, "floor_pct": 0.8},
            "acc_norm": {"value": 0.5222, "floor_pct": 0.8},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    TRUST_REMOTE_CODE = True
    USE_VLLM = False
    USE_FLASH_ATTN = False  # Deci remote code rejects flash_attention_2 during model init.
    EVAL_BATCH_SIZE = 6

    def test_deci(self):
        self.quantize_and_evaluate()
