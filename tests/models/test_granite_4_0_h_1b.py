# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from gptqmodel import BACKEND
from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


# a100:7, MARLIN kernel
# desc_act = False, act_group_aware = False 0.3200/0.3447
# desc_act = False, act_group_aware = True 0.3181/0.3481
# desc_act = True, REGRESSION 0.3191/0.3601
# a100:6+7: MARLIN kernel
# desc_act = False, act_group_aware = True 0.3217/0.3643
# | Metric                         |   MARLIN |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.3174 |
# | arc_challenge :: acc_norm,none |   0.3601 |
# | mmlu_stem :: acc,none          |   0.3186 |
class Test_Granite_4_0_H_1B(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/granite-4.0-h-1b" # "ibm-granite/granite-4.0-h-1b"
    GROUP_SIZE = 32
    EVAL_BATCH_SIZE = 1
    LOAD_BACKEND = BACKEND.TORCH
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3968,
                "floor_pct": 0.04,
                "ceil_pct": 0.10,
            },
            "acc_norm": {
                "value": 0.4138,
                "floor_pct": 0.04,
                "ceil_pct": 0.10,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.4015,
                "floor_pct": 0.1,
                "ceil_pct": 0.20,
            },
        },
    }

    def test_granite(self):
        self.quant_lm_eval()
