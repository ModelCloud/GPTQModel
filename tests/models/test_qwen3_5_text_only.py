# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest


# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.6092 |
# | arc_challenge :: acc_norm,none |   0.6143 |
# | mmlu_stem :: acc,none          |   0.8461 |
class TestQwen3_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3.5-9B-text-only" # principled-intelligence/Qwen3.5-9B-text-only
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.6092, "floor_pct": 0.04},
            "acc_norm": {"value": 0.6143, "floor_pct": 0.04},
        },
        ######################################################################
        # ⚠️  IMPORTANT NOTE (GSM8K_PLATINUM_COT)
        #
        # For SOME models (e.g. qwen3_moe, Qwen3.5-9B):
        #
        #   - apply_chat_template MUST be set to False
        #   - Otherwise, GSM8K_PLATINUM_COT scores will be SEVERELY degraded
        #
        # Empirical comparison (Qwen3.5-9B as an example):
        #   - apply_chat_template = False  → score = 0.8991
        #   - apply_chat_template = True   → score = 0.0438 (broken)
        #
        # Conclusion:
        #   GSM8K_PLATINUM_COT should NOT use chat templates for certain models,
        #   as lead to unreliable evaluation results.
        #
        ######################################################################
        "gsm8k_platinum_cot": {
            "chat_template": False,
            "acc,num": {
                "value": 0.8991,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)


    def test_qwen3_5(self):
        self.quantize_and_evaluate()
