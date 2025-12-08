# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestMistral3(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ministral-3-3B-Instruct-2512-BF16" # "mistralai/Ministral-3-3B-Instruct-2512-BF16"
    NATIVE_ARC_CHALLENGE_ACC = 0.4974
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5256
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": False,
            "acc": {"value": NATIVE_ARC_CHALLENGE_ACC},
            "acc_norm": {"value": NATIVE_ARC_CHALLENGE_ACC_NORM},
        },
    }

    def test_mistral3(self):
        self.quant_lm_eval()
