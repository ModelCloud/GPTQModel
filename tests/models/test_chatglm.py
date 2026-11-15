# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from accelerate import init_empty_weights
from model_test import ModelTest
from transformers import AutoConfig, AutoModelForCausalLM

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.chatglm import ChatGLMQModel
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
        self._assert_generate(self.model)

    def test_chatglm_native_generate(self):
        """Issue #4093 item 11: native `generate()` must accept the config aliases."""

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=None,
            trust_remote_code=True,
            device="cuda:0",
        )
        self._assert_generate(model)

    def test_chatglm_generate_after_config_compat(self):
        """Tiny from-config model: `max_length` alias must not leak into generation."""

        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=True)
        config.hidden_size = 32
        config.ffn_hidden_size = 64
        config.num_layers = 1
        config.num_attention_heads = 2
        config.multi_query_group_num = 1
        config.kv_channels = 16
        config.padded_vocab_size = 64
        config.seq_length = 32
        config.torch_dtype = "float32"
        prepare_remote_code_compat(config)

        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        ChatGLMQModel._patch_generation_compat(model)

        assert model.config.max_length == 32
        assert "max_length" not in model.config._get_generation_parameters()
        self._assert_generate(model, device="cpu")

    @staticmethod
    def _assert_generate(model, device=None):
        if model is None:
            raise AssertionError("ChatGLM model is unavailable for generate()")
        if device is None:
            device = getattr(model, "device", None) or "cuda:0"
        ids = torch.tensor([[1, 2, 3, 4]], device=device)
        out = model.generate(input_ids=ids, max_new_tokens=2, do_sample=False)
        assert out.shape[-1] >= ids.shape[-1] + 2
        if hasattr(model, "model"):
            direct = model.model.generate(ids, max_new_tokens=2, do_sample=False)
            assert direct.shape[-1] >= ids.shape[-1] + 2
