# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGlm4Moe(ModelTest):
    # FORMAT = FORMAT.GEMM
    # METHOD = METHOD.AWQ

    NATIVE_MODEL_ID = "/monster/data/model/GLM-4.6/"
    DELETE_QUANTIZED_MODEL = False
    DATASET_SIZE = 512
    GROUP_SIZE = 32
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5026, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5171, "floor_pct": 0.04},
        },
        "mmlu_stem": {
            "acc": {"value": 0.6362, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    def test_glm4moe(self):
        self.quantize_and_evaluate()
