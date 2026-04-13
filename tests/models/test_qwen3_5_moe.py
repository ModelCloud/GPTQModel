# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest

from gptqmodel.quantization.config import ExpertsRoutingOverride, Fallback, MoEConfig, VramStrategy


class TestQwen3_5Moe(ModelTest):
    FALLBACK = Fallback()
    # FORMAT = FORMAT.GEMM
    # METHOD = METHOD.AWQ

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
    # Keep the dense serial path on the first visible GPU and spread experts across the rest.
    DENSE_VRAM_STRATEGY_DEVICES = ["cuda:0"]
    MOE_VRAM_STRATEGY = VramStrategy.BALANCED
    MOE_VRAM_STRATEGY_DEVICES = ["cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5"]
    # Route every calibration token through every expert so MoE quant sees full coverage.
    MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok="all"))

    def test_qwen3_5_moe(self):
        self.quantize_and_evaluate()
