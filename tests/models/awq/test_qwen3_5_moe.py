# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_test import ModelTest

from gptqmodel.quantization.config import FORMAT, METHOD, Fallback, VramStrategy


class TestQwen3_5Moe(ModelTest):
    FALLBACK = Fallback()
    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ

    NATIVE_MODEL_ID = "/monster/data/model/Qwen3.5-35B-A3B"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5887, "floor_pct": 0.04},
            "acc_norm": {"value": 0.6100, "floor_pct": 0.04},
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.8106,
                "floor_pct": 0.04,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    DENSE_VRAM_STRATEGY = VramStrategy.EXCLUSIVE
    MOE_VRAM_STRATEGY = VramStrategy.BALANCED

    def test_qwen3_5_moe(self):
        self.quantize_and_evaluate()
