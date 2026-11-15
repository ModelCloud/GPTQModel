# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from model_test import ModelTest

from gptqmodel import GPTQModel


class TestInstella(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Instella-3B-Instruct/"
    NATIVE_ARC_CHALLENGE_ACC = 0.4377
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4804
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = NATIVE_ARC_CHALLENGE_ACC_SLOW
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW
    TRUST_REMOTE_CODE = True

    def test_instella(self):
        self.quantize_and_evaluate()

    def test_instella_native_rotary_inv_freq(self):
        """Issue #4093 item 10: a native load must not keep uninitialized RoPE buffers."""

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=None,
            trust_remote_code=True,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        config = model.model.config
        dim = config.hidden_size // config.num_attention_heads
        expected = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        )
        for layer in model.model.model.layers:
            inv_freq = layer.self_attn.rotary_emb.inv_freq
            assert torch.allclose(inv_freq.double().cpu(), expected, rtol=1e-6, atol=0.0)
