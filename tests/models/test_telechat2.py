# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestTeleChat_2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TeleChat2-7B/"  # "Tele-AI/TeleChat2-7B"
    NATIVE_ARC_CHALLENGE_ACC = 0.3677
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3831
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False
    USE_FLASH_ATTN = False
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": NATIVE_ARC_CHALLENGE_ACC},
            "acc_norm": {"value": NATIVE_ARC_CHALLENGE_ACC_NORM},
        },
    }


    def test_telechat2(self):
        self.quant_lm_eval()
