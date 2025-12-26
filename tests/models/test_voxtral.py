# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestVoxtral(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/shared/model/Voxtral-Mini-3B-2507/"
    NATIVE_ARC_CHALLENGE_ACC = 0.3392
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3360
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": False,
            "acc": {"value": NATIVE_ARC_CHALLENGE_ACC},
            "acc_norm": {"value": NATIVE_ARC_CHALLENGE_ACC_NORM},
        },
    }

    def test_voxtral(self):
        self.quant_lm_eval()
