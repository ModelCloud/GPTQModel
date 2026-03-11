# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchQuantLinear
from gptqmodel.quantization.protocol import (
    Rule,
    Stage,
    compile_protocol,
    compile_protocol_yaml_text,
    compile_plan_to_quantize_config,
)
from gptqmodel.utils.eval import EVAL


def _python_protocol():
    return {
        "version": 2,
        "stages": [
            Stage(
                name="weight_only",
                rules=[
                    Rule(
                        match="*",
                        weight={
                            "quantize": {
                                "method": "gguf",
                                "bits": "q4_k_m",
                            },
                            "export": {
                                "format": "gguf",
                                "variant": "q_k_m",
                                "impl": "gguf_torch",
                            },
                        },
                    ),
                ],
            ),
        ],
    }


def _yaml_protocol() -> str:
    return """
version: 2
stages:
  - name: weight_only
    rules:
      - match: "*"
        weight:
          quantize:
            method: gguf
            bits: q4_k_m
          export:
            format: gguf
            variant: q_k_m
            impl: gguf_torch
"""


class _BaseLlama3_2GGUFProtocol(ModelTest):
    pytestmark = pytest.mark.skipif(
        (not __import__("torch").cuda.is_available()) or __import__("torch").cuda.device_count() <= 3,
        reason="CUDA devices 2 and 3 are required for protocol GGUF integration tests",
    )

    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.3871,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.3955,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3106,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3532,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
    }
    LOAD_BACKEND = BACKEND.GGUF_TORCH
    KERNEL_INFERENCE = {GGUFTorchQuantLinear}

    def _compiled_protocol_plan(self):
        raise NotImplementedError

    def _build_quantize_config(self):
        return compile_plan_to_quantize_config(self._compiled_protocol_plan())


class TestLlama3_2_GGUFProtocolPython(_BaseLlama3_2GGUFProtocol):
    PIN_CUDA_DEVICE = 2

    def _compiled_protocol_plan(self):
        return compile_protocol(_python_protocol())

    def test_llama3_2_gguf_protocol_python(self):
        self.quant_lm_eval()


class TestLlama3_2_GGUFProtocolYAML(_BaseLlama3_2GGUFProtocol):
    PIN_CUDA_DEVICE = 3

    def _compiled_protocol_plan(self):
        return compile_protocol_yaml_text(_yaml_protocol())

    def test_llama3_2_gguf_protocol_yaml(self):
        self.quant_lm_eval()
