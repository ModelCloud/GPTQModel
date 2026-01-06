# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from models.model_test import ModelTest
from torch import nn

from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    ExpertsRoutingOverride,
    MoEConfig,
    QuantizeConfig,
    VramStrategy,
)
from gptqmodel.utils.model import get_module_by_name


EXPERTS = ['model.layers.0.mlp.experts.10', 'model.layers.0.mlp.experts.15', 'model.layers.0.mlp.experts.20',
           'model.layers.0.mlp.experts.39', 'model.layers.0.mlp.experts.51', 'model.layers.0.mlp.experts.55',
           'model.layers.0.mlp.experts.90', 'model.layers.0.mlp.experts.91']

PROJ_SUFFIXES = ["gate_proj", "up_proj", "down_proj"]

def assert_results(model, target_class, moe_config: MoEConfig):
    for base_path in EXPERTS:
        for suffix in PROJ_SUFFIXES:
            full_path = f"model.{base_path}.{suffix}"

            module = get_module_by_name(model, full_path)

            print("fff", full_path, module)

            assert isinstance(module, target_class), (
                f"{full_path} exists but is {target_class} (got {type(module)})"
            )
    print(model.quantize_config)
    qcfg: QuantizeConfig = model.quantize_config
    print("qcfg.meta_get", qcfg.meta_get("moe"))
    if moe_config is None:
        assert qcfg.meta_get("moe") is None
    elif isinstance(moe_config.routing, ExpertsRoutingOverride):
        moe = qcfg.meta_get("moe")
        assert moe["routing"]["class"] == ExpertsRoutingOverride.__name__
        assert moe["routing"]["num_experts_per_tok"] == moe_config.routing.num_experts_per_tok
    elif isinstance(moe_config.routing, ExpertsRoutingBypass):
        assert qcfg.meta_get("moe")["routing"]["class"] == ExpertsRoutingBypass.__name__


class TestMoEConfig(ModelTest):
    FAILSAFE = None
    DATASET_SIZE = 1  # Intentionally set to 1 to explicitly trigger and observe MoE routing
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-30B-A3B-layers-1"
    VRAM_STRATEGY = VramStrategy.BALANCED
    # MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride())
    SAVE_PATH = "Qwen3-30B-A3B-layers-1-gptq"

    def test_none_moe_config(self):
        self.MOE_CONFIG = None
        model, _ = self.quantModel(self.NATIVE_MODEL_ID, batch_size=self.QUANT_BATCH_SIZE,
                                   trust_remote_code=self.TRUST_REMOTE_CODE, dtype=self.TORCH_DTYPE, need_eval=False)

        assert_results(model, nn.Linear, self.MOE_CONFIG)

    def test_moe_routing_override(self):
        self.MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride())
        model, _ = self.quantModel(self.NATIVE_MODEL_ID, batch_size=self.QUANT_BATCH_SIZE,
                                   trust_remote_code=self.TRUST_REMOTE_CODE, dtype=self.TORCH_DTYPE, need_eval=False)

        assert_results(model, MarlinQuantLinear, self.MOE_CONFIG)

    def test_moe_routing_bypass(self):
        self.MOE_CONFIG = MoEConfig(routing=ExpertsRoutingBypass())
        model, _ = self.quantModel(self.NATIVE_MODEL_ID, batch_size=self.QUANT_BATCH_SIZE,
                                   trust_remote_code=self.TRUST_REMOTE_CODE, dtype=self.TORCH_DTYPE, need_eval=False)

        assert_results(model, MarlinQuantLinear, self.MOE_CONFIG)


