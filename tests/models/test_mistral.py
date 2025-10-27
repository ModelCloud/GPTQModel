# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestMistral(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Mistral-7B-Instruct-v0.2" # "mistralai/Mistral-7B-Instruct-v0.2"
    NATIVE_ARC_CHALLENGE_ACC = 0.5427
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5597
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": NATIVE_ARC_CHALLENGE_ACC},
            "acc_norm": {"value": NATIVE_ARC_CHALLENGE_ACC_NORM},
        },
    }

    def test_mistral(self):
        self.quant_lm_eval()
