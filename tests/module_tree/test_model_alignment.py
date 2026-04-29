# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from pathlib import Path

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForTextToWaveform

from gptqmodel.models.definitions.dots1 import Dots1QModel
from gptqmodel.models.definitions.qwen2 import Qwen2QModel
from gptqmodel.models.definitions.qwen2_5_omni import Qwen2_5_OmniGPTQ
from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel
from gptqmodel.models.definitions.qwen2_moe import Qwen2MoeQModel
from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5_MoeQModel
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


class TestQwen2ModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct"

    def test_qwen2_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

        decoder_layer = shell.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_proj"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "k_proj"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "v_proj"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "o_proj"))
        self.assertFalse(hasattr(decoder_layer.self_attn, "q_norm"))
        self.assertFalse(hasattr(decoder_layer.self_attn, "k_norm"))
        self.assertIn("q_proj:0", Qwen2QModel.module_tree[3]["self_attn"])
        self.assertIn("o_proj:1", Qwen2QModel.module_tree[3]["self_attn"])


class TestQwen2MoeModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen1.5-MoE-A2.7B"

    def test_qwen2_moe_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

        decoder_layer = shell.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_proj"))
        self.assertFalse(hasattr(decoder_layer.self_attn, "q_norm"))
        self.assertTrue(hasattr(decoder_layer.mlp, "gate"))
        self.assertTrue(hasattr(decoder_layer.mlp, "shared_expert"))
        self.assertTrue(hasattr(decoder_layer.mlp, "shared_expert_gate"))
        self.assertTrue(hasattr(decoder_layer.mlp, "experts"))
        self.assertIn("shared_expert_gate", Qwen2MoeQModel.module_tree[-1]["mlp:moe:?"])
        self.assertIn("shared_expert:0", Qwen2MoeQModel.module_tree[-1]["mlp:moe:?"])
        self.assertIn("experts:0", Qwen2MoeQModel.module_tree[-1]["mlp:moe:?"])


class TestQwen2VLModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2-VL-2B-Instruct"

    def test_qwen2_vl_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)

        decoder_layer = shell.model.language_model.layers[0]
        self.assertTrue(hasattr(shell.model, "language_model"))
        self.assertTrue(hasattr(shell.model, "visual"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_proj"))
        self.assertIn("q_proj:0", Qwen2VLQModel.module_tree[-1]["self_attn"])
        self.assertEqual(Qwen2VLQModel.module_tree[:4], ["model", "language_model", "layers", "#"])


class TestQwen2_5_VLModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-VL-3B-Instruct"

    def test_qwen2_5_vl_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)

        decoder_layer = shell.model.language_model.layers[0]
        self.assertTrue(hasattr(shell.model, "language_model"))
        self.assertTrue(hasattr(shell.model, "visual"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_proj"))
        self.assertIn("q_proj:0", Qwen2_5_VLQModel.module_tree[-1]["self_attn"])
        self.assertEqual(Qwen2_5_VLQModel.module_tree[:4], ["model", "language_model", "layers", "#"])


class TestQwen2_5_OmniModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-Omni-3B"

    def test_qwen2_5_omni_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForTextToWaveform.from_config(config, trust_remote_code=False)

        decoder_layer = shell.thinker.model.layers[0]
        self.assertTrue(hasattr(shell, "thinker"))
        self.assertTrue(hasattr(shell.thinker, "visual"))
        self.assertTrue(hasattr(shell.thinker, "audio_tower"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_proj"))
        self.assertIn("q_proj:0", Qwen2_5_OmniGPTQ.module_tree[-1]["self_attn"])
        self.assertEqual(Qwen2_5_OmniGPTQ.module_tree[:4], ["thinker", "model", "layers", "#"])


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


class TestQwen3_5ModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3.5-27B"

    def test_qwen3_5_linear_attention_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        config = config.text_config
        config.num_hidden_layers = 1
        config.layer_types = ["linear_attention"]
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

        decoder_layer = shell.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.linear_attn, "conv1d"))
        self.assertTrue(hasattr(decoder_layer.linear_attn, "in_proj_b"))
        self.assertTrue(hasattr(decoder_layer.linear_attn, "in_proj_a"))
        self.assertIn("conv1d:!", Qwen3_5QModel.module_tree[-1]["linear_attn"])
        self.assertIn("in_proj_b:!:1", Qwen3_5QModel.module_tree[-1]["linear_attn"])
        self.assertIn("in_proj_a:!:1", Qwen3_5QModel.module_tree[-1]["linear_attn"])


class TestQwen3_5MoeModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3.5-35B-A3B"

    def test_qwen3_5_moe_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        config.text_config.num_hidden_layers = 2
        config.text_config.layer_types = ["linear_attention", "full_attention"]
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)

        linear_layer = shell.model.language_model.layers[0]
        full_layer = shell.model.language_model.layers[1]

        self.assertTrue(hasattr(linear_layer.linear_attn, "conv1d"))
        self.assertTrue(hasattr(linear_layer.linear_attn, "in_proj_b"))
        self.assertTrue(hasattr(linear_layer.linear_attn, "in_proj_a"))
        self.assertTrue(hasattr(linear_layer.mlp, "shared_expert"))
        self.assertTrue(hasattr(linear_layer.mlp, "shared_expert_gate"))
        self.assertTrue(hasattr(full_layer.self_attn, "q_norm"))
        self.assertTrue(hasattr(full_layer.self_attn, "k_norm"))
        self.assertIn("q_norm:!", Qwen3_5_MoeQModel.module_tree[-1]["self_attn"])
        self.assertIn("k_norm:!", Qwen3_5_MoeQModel.module_tree[-1]["self_attn"])
        self.assertIn("conv1d:!", Qwen3_5_MoeQModel.module_tree[-1]["linear_attn"])
        self.assertIn("in_proj_b:!:1", Qwen3_5_MoeQModel.module_tree[-1]["linear_attn"])
        self.assertIn("in_proj_a:!:1", Qwen3_5_MoeQModel.module_tree[-1]["linear_attn"])
        self.assertIn("shared_expert_gate", Qwen3_5_MoeQModel.module_tree[-1]["mlp:moe:?"])
        self.assertIn("shared_expert:0", Qwen3_5_MoeQModel.module_tree[-1]["mlp:moe:?"])


class TestQwen3OmniModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-Omni-30B-A3B-Instruct"

    def test_qwen3_omni_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForTextToWaveform.from_config(config, trust_remote_code=False)

        decoder_layer = shell.thinker.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "k_norm"))
        self.assertEqual(Qwen3OmniMoeGPTQ.module_tree[-1]["mlp:moe"]["gate"], ("gate:!",))
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

    def test_qwen3_next_linear_attention_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=False)
        config.num_hidden_layers = 1
        config.layer_types = ["linear_attention"]
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

        decoder_layer = shell.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.linear_attn, "norm"))
        self.assertTrue(hasattr(decoder_layer.linear_attn, "conv1d"))
        self.assertTrue(hasattr(decoder_layer.linear_attn, "in_proj_ba"))
        self.assertTrue(hasattr(decoder_layer.mlp, "shared_expert_gate"))
        self.assertIn("norm:!", Qwen3NextGPTQ.module_tree[-1]["linear_attn"])
        self.assertIn("conv1d:!", Qwen3NextGPTQ.module_tree[-1]["linear_attn"])
        self.assertIn("in_proj_qkvz:0", Qwen3NextGPTQ.module_tree[-1]["linear_attn"])
        self.assertIn("in_proj_ba:!:0", Qwen3NextGPTQ.module_tree[-1]["linear_attn"])

        blocks = Qwen3NextGPTQ.build_layer_modules(Qwen3NextGPTQ.module_tree)
        linear_in_block = next(block for block in blocks if "linear_attn.in_proj_qkvz" in block)
        linear_out_block = next(block for block in blocks if "linear_attn.out_proj" in block)
        shared_expert_block = next(block for block in blocks if "mlp.shared_expert.gate_proj" in block)

        self.assertNotIn("linear_attn.out_proj", linear_in_block)
        self.assertNotIn("linear_attn.in_proj_qkvz", linear_out_block)
        self.assertIn("mlp.shared_expert.up_proj", shared_expert_block)
        self.assertNotIn("mlp.shared_expert.down_proj", shared_expert_block)
