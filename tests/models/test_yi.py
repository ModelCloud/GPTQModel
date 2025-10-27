# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestYi(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Yi-Coder-1.5B-Chat" # "01-ai/Yi-Coder-1.5B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.2679
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2986
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 4
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": NATIVE_ARC_CHALLENGE_ACC},
            "acc_norm": {"value": NATIVE_ARC_CHALLENGE_ACC_NORM},
        },
    }

    def test_yi(self):
        self.quant_lm_eval()
