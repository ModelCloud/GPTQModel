# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestInternlm2_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/internlm2_5-1_8b-chat" # "internlm/internlm2_5-1_8b-chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.3217
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3575
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False


    def test_internlm2_5(self):
        # transformers<=4.44.2 run normal
        self.quant_lm_eval()



