# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from model_test import ModelTest

from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.quantization.config import VRAMStrategy, METHOD, FORMAT
from gptqmodel.utils.eval import EVAL


class TestMarin(ModelTest):
    DATASET_SIZE = 1024
    GROUP_SIZE = 32
    METHOD = METHOD.AWQ
    FORMAT = FORMAT.GEMM

    NATIVE_MODEL_ID = "/monster/data/model/marin-32b-base"
    # VRAM_STRATEGY = VRAMStrategy.BALANCED
    # Marin inherits Qwen3's backbone with QK-Norm attention.
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5828, "floor_pct": 0.04},
            "acc_norm": {"value": 0.6007, "floor_pct": 0.04},
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "acc": {"value": 0.6673, "floor_pct": 0.04},
        },
    }

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
