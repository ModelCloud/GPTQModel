# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest

from gptqmodel.quantization.config import FailSafe, VramStrategy
from gptqmodel.utils.eval import EVAL


# | Metric                         |   MARLIN |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.5094 |
# | arc_challenge :: acc_norm,none |   0.5486 |
# Qwen3-30B-A3B-MainBranch-FailSafe_Enable
#
# |    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |arc_challenge|      1|none  |     0|acc     |↑  |0.5307|±  |0.0146|
# |             |       |none  |     0|acc_norm|↑  |0.5674|±  |0.0145|
class TestQwen3Moe(ModelTest):
    FAILSAFE = FailSafe()
    # FORMAT = FORMAT.GEMM
    # METHOD = METHOD.AWQ

    #DATASET_SIZE = 1
    # DEVICE = torch.device("cpu")
    # HESSIAN_CHUNK_SIZE = 256 * 1024 * 1024
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-30B-A3B"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5137, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5307, "floor_pct": 0.04},
        },
        ######################################################################
        # ⚠️  IMPORTANT NOTE (GSM8K_PLATINUM_COT)
        #
        # For SOME models (e.g. qwen3_moe):
        #
        #   - apply_chat_template MUST be set to False
        #   - Otherwise, GSM8K_PLATINUM_COT scores will be SEVERELY degraded
        #
        # Empirical comparison (qwen3_moe as an example):
        #   - apply_chat_template = False  → score = 0.9380
        #   - apply_chat_template = True   → score = 0.2498 (broken)
        #
        # Conclusion:
        #   GSM8K_PLATINUM_COT should NOT use chat templates for certain models,
        #   as lead to unreliable evaluation results.
        #
        ######################################################################
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": False,
            "exact_match,flexible-extract": {
                "value": 0.9380,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.7805,
                "floor_pct": 0.04,
            },
        },
    }

    VRAM_STRATEGY = VramStrategy.BALANCED
    # TRUST_REMOTE_CODE = False
    # APPLY_CHAT_TEMPLATE = True
    # EVAL_BATCH_SIZE = 6
    # V2 = False
    # DEBUG = True
    # ACT_GROUP_AWARE = True
    # DESC_ACT = False
    # DATASET_SIZE = 512
    # DATASET_SORT = "desc"
    # QUANT_BATCH_SIZE = 4
    # CALIB_NOISE_MODE = "unseen"
    # CALIB_NOISE_PERCENT = 0.025

    def test_mimo(self):
        self.quant_lm_eval()
