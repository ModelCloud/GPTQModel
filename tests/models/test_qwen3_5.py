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
    GROUP_SIZE = 32
    HESSIAN_CHUNK_SIZE = 256 * 1024 * 1024
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3.5-27B"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.6092, "floor_pct": 0.04},
            "acc_norm": {"value": 0.6143, "floor_pct": 0.04},
        },
        "mmlu_stem": {
            "acc": {"value": 0.8461, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)


    def test_qwen3_5(self):
        self.quantize_and_evaluate()
