# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


# The official THUDM/chatglm3-6b's tokenization_chatglm.py has compatibility issues with transformers.
# It will throw a TypeError: ChatGLMTokenizer._pad() got an unexpected keyword argument 'padding_side'
# Adding a temporary padding_side parameter to the _pad method in tokenization_chatglm.py can prevent errors.
class TestChatGlm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/chatglm3-6b"  # "THUDM/chatglm3-6b"
    NATIVE_ARC_CHALLENGE_ACC = 0.3319
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3729
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False

    def test_chatglm(self):
        self.quant_lm_eval()
