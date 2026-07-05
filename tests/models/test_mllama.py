# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest


class TestMLlama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-11B-Vision-Instruct" # "meta-llama/Llama-3.2-11B-Vision-Instruct"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.5051, "floor_pct": 0.36},
            "acc_norm": {"value": 0.5026, "floor_pct": 0.36},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False

    def test_mllama(self):
        self.quantize_and_evaluate()
        model = self.model
        tokenizer = model.tokenizer
        generate_str = self.generate_stable_with_limit(
            model,
            tokenizer,
            "The capital of France is is",
        )

        print(f"generate_str: {generate_str}")

        self.assertIn("paris", generate_str.lower())
