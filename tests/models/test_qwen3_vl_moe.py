# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestQwen3VLMoe(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-VL-30B-A3B-Instruct" # Qwen/Qwen3-VL-30B-A3B-Instruct
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_BATCH_SIZE = 8

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.4505, "floor_pct": 0.04},
            "acc_norm": {"value": 0.4291, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_qwen3_vl_moe(self):
        self.quantize_and_evaluate()
