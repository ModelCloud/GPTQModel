# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest

from accelerate import init_empty_weights
from model_test import ModelTest
from transformers import AutoConfig, AutoModelForCausalLM

from gptqmodel.models.definitions.dots1 import Dots1QModel
from gptqmodel.quantization import METHOD
from gptqmodel.quantization.config import QuantizeConfig


class TestDots1Inst(ModelTest):
    NATIVE_MODEL_ID = "rednote-hilab/dots.llm1.inst"
    TRUST_REMOTE_CODE = True

    def test_module_tree_alignment(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=True)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        dense_layer = shell.model.layers[0]
        self.assertTrue(hasattr(dense_layer.mlp, "gate_proj"))
        self.assertFalse(hasattr(dense_layer.mlp, "experts"))

        moe_layer = shell.model.layers[config.first_k_dense_replace]
        self.assertTrue(hasattr(moe_layer.mlp, "experts"))
        self.assertEqual(len(moe_layer.mlp.experts), config.n_routed_experts)
        self.assertTrue(hasattr(moe_layer.mlp, "shared_experts"))

        self.assertIn("q_norm:!", Dots1QModel.module_tree[3]["self_attn"])
        self.assertIn("k_norm:!", Dots1QModel.module_tree[3]["self_attn"])
        self.assertIn("gate_proj:0", Dots1QModel.module_tree[3]["mlp"][""])
        self.assertIn("#", Dots1QModel.module_tree[3]["mlp"]["experts"])
        self.assertIn("shared_experts", Dots1QModel.module_tree[3]["mlp"])

        qcfg = QuantizeConfig(bits=4, group_size=128, quant_method=METHOD.GPTQ)
        layer_modules = Dots1QModel.simple_layer_modules(config, qcfg)
        flattened = [name for block in layer_modules for name in block]
        self.assertNotIn("self_attn.q_norm", flattened)
        self.assertTrue(any(name.startswith("mlp.shared_experts.") for name in flattened))
        self.assertTrue(any(name.startswith("mlp.experts.") for name in flattened))

    @unittest.skip("dots.llm1.inst is too large to quantize during automated tests.")
    def test_dots1_inst(self):
        self.quant_lm_eval()
