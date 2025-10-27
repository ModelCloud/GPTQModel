# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestGranite(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/granite-3.0-2b-instruct" # "ibm-granite/granite-3.0-2b-instruct"
    TRUST_REMOTE_CODE = True
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": 0.4505, "floor_pct": 0.2},
            "acc_norm": {"value": 0.4770, "floor_pct": 0.2},
        },
    }

    def test_granite(self):
        self.quant_lm_eval()
