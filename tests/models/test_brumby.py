# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


# | Metric                    |   MARLIN |
# |---------------------------|----------|
# | arc_challenge :: acc      |   0.8900 |
# | gsm8k_cot :: exact        |   0.8800 |
# | gsm8k_platinum_cot :: exact_match,flexible-extract |   0.8700 |
# | mmlu :: acc               |   0.7100 |
class TestBrumby(ModelTest):
    GROUP_SIZE = 32
    NATIVE_MODEL_ID = "/monster/data/model/Brumby-14B-Base"
    # EVAL_BATCH_SIZE = 32
    TRUST_REMOTE_CODE = True
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.87,
                "floor_pct": 6.05,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {
                "value": 0.89,
                "floor_pct": 4.05,
            },
        },
        EVAL.LM_EVAL.MMLU: {
            "acc": {
                "value": 0.71,
                "floor_pct": 4.05,
            },
        },
    }

    def test_brumby(self):
        self.quant_lm_eval()
