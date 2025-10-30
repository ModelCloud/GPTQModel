# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.eval import EVAL


# a100:0
# desc_act = False, act_group_aware = False 0.2500/0.2841
# desc_act = False, act_group_aware = True 0.3063/0.3456
# desc_act = True, 0.3089/0.3328
class TestLlama3_2_awq(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3200,
                "floor_pct": 0.04,
                "ceil_pct": 0.10,
            },
            "acc_norm": {
                "value": 0.3362,
                "floor_pct": 0.04,
                "ceil_pct": 0.10,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.3657,
                "floor_pct": 0.04,
                "ceil_pct": 0.10,
            },
        },
    }
    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ

    def test_llama3_2_awq(self):
        self.quant_lm_eval()
