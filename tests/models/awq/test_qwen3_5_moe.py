# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest

from gptqmodel.quantization.config import FailSafe, VramStrategy
from gptqmodel.utils.eval import EVAL
from gptqmodel.quantization.config import FORMAT, METHOD


class TestQwen3_5Moe(ModelTest):
    FAILSAFE = FailSafe()
    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ

    NATIVE_MODEL_ID = "/monster/data/model/Qwen3.5-35B-A3B"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5887, "floor_pct": 0.04},
            "acc_norm": {"value": 0.6100, "floor_pct": 0.04},
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.8106,
                "floor_pct": 0.04,
            },
        },
    }

    VRAM_STRATEGY = VramStrategy.BALANCED
    OFFLOAD_TO_DISK = False  # FIXME Currently, after defuser transforms the model, OFFLOAD_TO_DISK must be False for quantization.

    def test_qwen3_5_moe(self):
        self.quant_lm_eval()
