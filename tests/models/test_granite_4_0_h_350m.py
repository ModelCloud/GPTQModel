# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.utils.eval import EVAL


# a100:0, TORCH kernel
# desc_act = False, act_group_aware = True
# | Metric                         |   MARLIN |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.3046 |
# | arc_challenge :: acc_norm,none |   0.3157 |
# | mmlu_stem :: acc,none          |   0.2915 |
class Test_Granite_4_0_H_350M(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/granite-4.0-h-350m" # "ibm-granite/granite-4.0-h-350m"
    GROUP_SIZE = 32
    EVAL_BATCH_SIZE = 16
    LOAD_BACKEND = BACKEND.TORCH
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3046,
                "floor_pct": 0.04,
                "ceil_pct": 0.10,
            },
            "acc_norm": {
                "value": 0.3157,
                "floor_pct": 0.04,
                "ceil_pct": 0.10,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.2915,
                "floor_pct": 0.1,
                "ceil_pct": 0.20,
            },
        },
    }

    def test_granite(self):
        self.quant_lm_eval()
