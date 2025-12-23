# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.quantization.config import VramStrategy
from gptqmodel.utils.eval import EVAL


# | Metric                         |   MARLIN |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.6271 |
# | arc_challenge :: acc_norm,none |   0.6613 |
# | mmlu_stem :: acc,none          |   0.8403 |
class TestQwen3Next(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-Next-80B-A3B-Instruct"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.6271, "floor_pct": 0.04},
            "acc_norm": {"value": 0.6613, "floor_pct": 0.04},
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "acc": {"value": 0.8403, "floor_pct": 0.04},
        },
    }

    VRAM_STRATEGY = VramStrategy.BALANCED
    # DATASET_SIZE = 2048
    # TRUST_REMOTE_CODE = True
    # APPLY_CHAT_TEMPLATE = True
    # EVAL_BATCH_SIZE = 4
    # V2 = False
    # DEBUG = True
    # ACT_GROUP_AWARE = True
    # DESC_ACT = False
    # DATASET_SIZE = 1024
    # DATASET_SORT = "desc"
    # QUANT_BATCH_SIZE = 4
    # CALIB_NOISE_MODE = "unseen"
    # CALIB_NOISE_PERCENT = 0.025
    # USE_FLASH_ATTN = True

    def test_mimo(self):
        self.quant_lm_eval()
