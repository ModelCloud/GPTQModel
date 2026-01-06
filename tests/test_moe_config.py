# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import tempfile

import torch
from torch import nn

from models.model_test import ModelTest

from gptqmodel import GPTQModel
from gptqmodel.models.writer import QUANT_LOG_NSAMPLES
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    ExpertsRoutingOverride,
    MoEConfig,
    QuantizeConfig,
    VramStrategy,
)
from gptqmodel.utils.calibration import prepare_calibration_dataset
from gptqmodel.utils.model import get_module_by_name


# Explicitly selected experts for validation.
# These are used to verify that MoE structure is preserved after quantization.
EXPERTS = [
    'model.layers.0.mlp.experts.10',
    'model.layers.0.mlp.experts.15',
    'model.layers.0.mlp.experts.20',
    'model.layers.0.mlp.experts.39',
    'model.layers.0.mlp.experts.51',
    'model.layers.0.mlp.experts.55',
    'model.layers.0.mlp.experts.90',
    'model.layers.0.mlp.experts.91',
]

# Standard MoE MLP projections per expert
PROJ_SUFFIXES = ["gate_proj", "up_proj", "down_proj"]


def assert_results(model, target_class, moe_config: MoEConfig):
    """
    Validate that:
    1) All selected MoE expert submodules (gate/up/down) exist and have expected types
    2) MoE-related metadata is correctly recorded in quantize_config
    """

    # Structural validation: expert modules must remain present
    for base_path in EXPERTS:
        for suffix in PROJ_SUFFIXES:
            full_path = f"model.{base_path}.{suffix}"
            module = get_module_by_name(model, full_path)
            assert isinstance(module, target_class), (
                f"{full_path} exists but is {target_class} (got {type(module)})"
            )

    # Metadata validation
    qcfg: QuantizeConfig = model.quantize_config

    if moe_config is None:
        # No MoE config → no MoE metadata should be emitted
        assert qcfg.meta_get("moe") is None

    elif isinstance(moe_config.routing, ExpertsRoutingOverride):
        # Override routing must be reflected exactly in metadata
        moe = qcfg.meta_get("moe")
        assert moe["routing"]["class"] == ExpertsRoutingOverride.__name__
        assert moe["routing"]["num_experts_per_tok"] == moe_config.routing.num_experts_per_tok

    elif isinstance(moe_config.routing, ExpertsRoutingBypass):
        # Bypass routing disables expert selection while preserving structure
        assert qcfg.meta_get("moe")["routing"]["class"] == ExpertsRoutingBypass.__name__


class TestMoEConfig(ModelTest):
    """
    Tests MoE quantization behavior under different routing configurations.
    Focus:
    - quant_log coverage
    - calibration sample accounting
    - structural integrity of MoE experts after quantization
    """

    FAILSAFE = None

    # Intentionally minimal to force observable MoE routing behavior
    DATASET_SIZE = 1

    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-30B-A3B-layers-1"
    VRAM_STRATEGY = VramStrategy.BALANCED
    SAVE_PATH = "Qwen3-30B-A3B-layers-1-gptq"

    calibration_dataset = None
    calibration_dataset_token_size = None

    @classmethod
    def setUpClass(cls):
        tokenizer = cls.load_tokenizer(cls.NATIVE_MODEL_ID)
        cls.calibration_dataset = cls.load_dataset(tokenizer, cls.DATASET_SIZE)

    def quantize_and_assert(self):
        # Apply GPTQ quantization with optional MoE routing configuration
        quant_config = QuantizeConfig(bits=4, group_size=128, moe=self.MOE_CONFIG)
        model = GPTQModel.load(self.NATIVE_MODEL_ID, quant_config)

        # Compute total calibration token size
        calibration_dataset_token_size = 0
        for data in prepare_calibration_dataset(
            qmodel=model,
            calibration_dataset=self.calibration_dataset,
        ):
            for input_id in data["input_ids"]:
                calibration_dataset_token_size += len(input_id)

        # Perform quantization; quant_log will record per-module calibration stats
        model.quantize(self.calibration_dataset, batch_size=8)

        # Compute expected total number of quantizable modules
        # One layer contains:
        # - 4 attention projections (q, k, v, o)
        # - num_experts × 3 MLP projections (gate, up, down)
        num_attention_modules = 4
        num_experts = model.get_num_experts(model.model.config)
        num_mlp_modules_per_expert = 3

        total_quant_module_size = (
            num_attention_modules +
            num_experts * num_mlp_modules_per_expert
        )

        # Collect per-module calibration sample counts
        quant_samples = []
        for entry in model.quant_log:
            quant_samples.append(int(entry.get(QUANT_LOG_NSAMPLES)))

        avg_quant_samples = sum(quant_samples) / len(quant_samples)

        if self.MOE_CONFIG is None:
            # Without MoE routing, only a subset of modules should be quantized
            self.assertLess(len(quant_samples), total_quant_module_size)
            self.assertLess(avg_quant_samples, calibration_dataset_token_size)
        else:
            # With MoE routing enabled, all expert modules should be calibrated
            self.assertEqual(len(quant_samples), total_quant_module_size)
            self.assertEqual(avg_quant_samples, calibration_dataset_token_size)

        # Reload the saved model and validate post-quantization structure
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            del model
            torch.cuda.empty_cache()

            quantized_model = GPTQModel.load(tmp_dir, device_map="auto")
            target_cls = MarlinQuantLinear if self.MOE_CONFIG else nn.Linear
            assert_results(quantized_model, target_cls, self.MOE_CONFIG)

    def test_none_moe_config(self):
        # Baseline: no MoE routing enabled
        self.MOE_CONFIG = None
        self.quantize_and_assert()

    def test_moe_routing_override(self):
        # Explicitly override expert routing during quantization
        self.MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride())
        self.quantize_and_assert()

    def test_moe_routing_bypass(self):
        # Bypass routing logic while keeping MoE structure intact
        self.MOE_CONFIG = MoEConfig(routing=ExpertsRoutingBypass())
        self.quantize_and_assert()
