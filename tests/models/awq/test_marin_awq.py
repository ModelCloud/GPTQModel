# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.quantization.config import FORMAT, METHOD


class TestMarin(ModelTest):
    DATASET_SIZE = 1024
    GROUP_SIZE = 32
    METHOD = METHOD.AWQ
    FORMAT = FORMAT.GEMM

    NATIVE_MODEL_ID = "/monster/data/model/marin-32b-base"
    # VRAM_STRATEGY = VramStrategy.BALANCED
    # Marin inherits Qwen3's backbone with QK-Norm attention.
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5299, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5546, "floor_pct": 0.04},
        },
        "gsm8k_platinum_cot": {
            "chat_template": False,
            "acc,num": {
                "value": 0.6716294458229942,
                "floor_pct": 0.04,
            },
        },
        "mmlu_stem": {
            "acc": {"value": 0.6676, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_marin(self):
        self.quantize_and_evaluate()
