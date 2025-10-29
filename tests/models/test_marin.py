# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from model_test import ModelTest

from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.quantization.config import VRAMStrategy


class TestMarin(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/marin-32b-base"
    VRAM_STRATEGY = VRAMStrategy.BALANCED
    # Marin inherits Qwen3's backbone with QK-Norm attention.

    def test_marin_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=True)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        decoder_layer = shell.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "k_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_proj"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "o_proj"))
        self.assertIn("q_norm:!", Qwen3QModel.module_tree[3]["self_attn"])
        self.assertIn("k_norm:!", Qwen3QModel.module_tree[3]["self_attn"])

    def test_marin(self):
        self.quant_lm_eval()
