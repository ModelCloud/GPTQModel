# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


from accelerate import init_empty_weights
from model_test import ModelTest
from transformers import AutoConfig, AutoModelForCausalLM

from gptqmodel.models.definitions.dots1 import Dots1QModel
from gptqmodel.quantization import METHOD
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.eval import EVAL


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

# | Metric                         |   MARLIN |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.3046 |
# | arc_challenge :: acc_norm,none |   0.3345 |
# | mmlu_stem :: acc,none          |   0.3768 |
# | gsm8k_plat :: exact,flexible   |   0.1944 |
class TestDotsOne(ModelTest):
    # DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/monster/data/model/dots.llm1.inst"
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.1944,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.3768, # 0.3099 4096, 0.3270 2048
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3046,  # 0.3294 4096, 0.3242 2048
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3345,  # 0.3558 4096, 0.3635 2048
                "floor_pct": 0.04,
            },
        },
    }

    # llama 3.2 Instruct requires chat = true to have normal ARC scores
    # mmlu requires chat = false
    # APPLY_CHAT_TEMPLATE = True
    # QUANT_BATCH_SIZE = 4

    # EORA = Lora(
    #     # for quant, path is save path. for load, it is loading path
    #     path="./eora_test",
    #     rank=128,
    # )
    # b1 = 0.315, b4 = 0.3106, b8 = 0.3148, b32 = 0.3148, b16 = 0.3234

    def test_dots_one(self):
        self.quant_lm_eval()
