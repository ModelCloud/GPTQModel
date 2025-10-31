# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.eval import EVAL


# | Metric                         |   MARLIN |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.3106 |
# | arc_challenge :: acc_norm,none |   0.3532 |
# | mmlu_stem :: acc,none          |   0.3527 |
# | gsm8k_plat :: exact,flexible   |   0.2192 |
class TestLlama3_2_awq(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048 # new
    EVAL_TASKS = {
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.2192,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3106,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3532,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.3527,
                "floor_pct": 0.04,
            },
        },
    }
    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ

    def test_llama3_2_awq(self):
        self.quant_lm_eval()
