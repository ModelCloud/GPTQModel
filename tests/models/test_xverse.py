# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestXVerse(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/XVERSE-7B-Chat" # "xverse/XVERSE-7B-Chat"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.4198, "floor_pct": 0.2},
            "acc_norm": {"value": 0.4044, "floor_pct": 0.2},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False
    USE_FLASH_ATTN = False

    def test_xverse(self):
        try:
            self.load_tokenizer(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE)
        except Exception as exc:
            if "add_prefix_space does not match declared prepend_scheme" in str(exc):
                self.skipTest(f"Tokenizer assets are incompatible with the installed tokenizers runtime: {exc}")
            raise

        self.quantize_and_evaluate()
