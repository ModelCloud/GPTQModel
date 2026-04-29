# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from accelerate import init_empty_weights
from model_test import ModelTest
from transformers import AutoConfig, AutoModelForCausalLM

from gptqmodel.utils.hf import prepare_remote_code_compat


# The official THUDM/chatglm3-6b's tokenization_chatglm.py has compatibility issues with transformers.
# It will throw a TypeError: ChatGLMTokenizer._pad() got an unexpected keyword argument 'padding_side'
# Adding a temporary padding_side parameter to the _pad method in tokenization_chatglm.py can prevent errors.
class TestChatGlm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/chatglm3-6b"  # "THUDM/chatglm3-6b"
    NATIVE_ARC_CHALLENGE_ACC = 0.3319
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3729
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = NATIVE_ARC_CHALLENGE_ACC_SLOW
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False

    def test_chatglm_from_config_compat(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=True)
        prepare_remote_code_compat(config)

        with init_empty_weights(include_buffers=True):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        assert model.max_sequence_length == config.seq_length
        assert isinstance(model.all_tied_weights_keys, dict)

    def test_chatglm(self):
        self.quantize_and_evaluate()
