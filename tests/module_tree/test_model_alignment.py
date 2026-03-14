# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from pathlib import Path

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTextToWaveform

from gptqmodel.models.definitions.dots1 import Dots1QModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.models.definitions.qwen3_next import Qwen3NextGPTQ
from gptqmodel.models.definitions.qwen3_omni_moe import Qwen3OmniMoeGPTQ


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from model_test import ModelTest  # noqa: E402
from torch import nn


class TestDots1Struct(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/dots.llm1.inst"
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
        if isinstance(moe_layer.mlp.experts, nn.ModuleList):
            expert_num = len(moe_layer.mlp.experts)
        else:
            expert_num = moe_layer.mlp.experts.num_experts
        self.assertEqual(expert_num, config.n_routed_experts)
        self.assertTrue(hasattr(moe_layer.mlp, "shared_experts"))

        self.assertIn("q_norm:!", Dots1QModel.module_tree[3]["self_attn"])
        self.assertIn("k_norm:!", Dots1QModel.module_tree[3]["self_attn"])
        mlp_key = next(
            key for key in Dots1QModel.module_tree[3].keys()
            if Dots1QModel._parse_module_flags(key)[0] == "mlp"
        )
        self.assertIn("gate_proj:0", Dots1QModel.module_tree[3][mlp_key][""])
        self.assertIn("#", Dots1QModel.module_tree[3][mlp_key]["experts"])
        self.assertIn("shared_experts", Dots1QModel.module_tree[3][mlp_key])


class TestMarinModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/marin-32b-base"

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


class TestMarinAwqModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/marin-32b-base"

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


class TestQwen3MoeModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-30B-A3B"

    def test_qwen3_moe_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

        decoder_layer = shell.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "k_norm"))
        self.assertIn("q_norm:!", Qwen3MoeQModel.module_tree[-1]["self_attn"])
        self.assertIn("k_norm:!", Qwen3MoeQModel.module_tree[-1]["self_attn"])


class TestQwen3OmniModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-Omni-30B-A3B-Instruct"

    def test_qwen3_omni_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForTextToWaveform.from_config(config, trust_remote_code=False)

        decoder_layer = shell.thinker.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "k_norm"))
        self.assertIn("q_norm:!", Qwen3OmniMoeGPTQ.module_tree[-1]["self_attn"])
        self.assertIn("k_norm:!", Qwen3OmniMoeGPTQ.module_tree[-1]["self_attn"])


class TestQwen3NextModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-Next-80B-A3B-Instruct"

    def test_qwen3_next_full_attention_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

        decoder_layer = shell.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "k_norm"))
        self.assertIn("q_norm:!", Qwen3NextGPTQ.module_tree[-1]["self_attn"])
        self.assertIn("k_norm:!", Qwen3NextGPTQ.module_tree[-1]["self_attn"])
