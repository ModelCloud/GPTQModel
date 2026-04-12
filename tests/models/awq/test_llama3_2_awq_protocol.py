# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest
import torch


TESTS_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TESTS_MODELS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_MODELS_ROOT)

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.quantization import FORMAT, METHOD, AWQConfig
from gptqmodel.quantization.protocol import (
    Rule,
    Stage,
    compile_plan_to_quantize_config,
    compile_protocol,
    compile_protocol_yaml_text,
)
from gptqmodel.utils.machete import _validate_machete_device_support


LAYER0_ONLY_NEGATIVE_MATCH = r".*layers\.(?:[1-9]|[12][0-9]|3[0-2])\..*"


def _python_protocol():
    return {
        "version": 2,
        "stages": [
            Stage(
                name="ptq",
                rules=[
                    Rule(
                        match=["*", f"-:{LAYER0_ONLY_NEGATIVE_MATCH}"],
                        weight={
                            "quantize": {
                                "method": "awq",
                                "bits": 4,
                                "group_size": 128,
                                "sym": True,
                                "desc_act": False,
                            },
                            "export": {
                                "format": "awq",
                                "variant": "gemm",
                                "impl": "marlin_awq",
                            },
                        },
                    ),
                ],
            ),
        ],
    }


def _yaml_protocol() -> str:
    return r"""
version: 2
stages:
  - name: ptq
    rules:
      - match:
          - "*"
          - '-:.*layers\.(?:[1-9]|[12][0-9]|3[0-2])\..*'
        weight:
          quantize:
            method: awq
            bits: 4
            group_size: 128
            sym: true
            desc_act: false
          export:
            format: awq
            variant: gemm
            impl: marlin_awq
"""


class _BaseLlama3_2AWQProtocol(ModelTest):
    pytestmark = pytest.mark.skipif(
        (
            (not __import__("torch").cuda.is_available())
            or not _validate_machete_device_support()
        ),
        reason="CUDA plus NVIDIA Hopper-or-newer GPUs are required for AWQ protocol dynamic-match integration tests",
    )

    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.4690,
                "floor_pct": 0.05,
                "ceil_pct": 0.05,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.3999,
                "floor_pct": 0.03,
                "ceil_pct": 0.03,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3221,
                "floor_pct": 0.05,
                "ceil_pct": 0.05,
            },
            "acc_norm": {
                "value": 0.3528,
                "floor_pct": 0.03,
                "ceil_pct": 0.03,
            },
        },
    }
    TORCH_DTYPE = torch.float16
    QUANT_BACKEND = BACKEND.MACHETE
    LOAD_BACKEND = BACKEND.MACHETE
    KERNEL_INFERENCE = {AwqMacheteLinear}

    def _compiled_protocol_plan(self):
        raise NotImplementedError

    def _build_quantize_config(self):
        return compile_plan_to_quantize_config(self._compiled_protocol_plan())

    def _assert_layer0_only_dynamic(self, cfg):
        assert isinstance(cfg, AWQConfig)
        assert cfg.quant_method == METHOD.AWQ
        assert cfg.format == FORMAT.GEMM
        assert cfg.dynamic == {f"-:{LAYER0_ONLY_NEGATIVE_MATCH}": {}}

    def _assert_only_first_layer_quantized(self, model):
        layer0_quantized = []
        later_layer_quantized = []

        for name, module in model.named_modules():
            if not isinstance(module, BaseQuantLinear):
                continue
            if ".layers.0." in name:
                layer0_quantized.append(name)
            elif ".layers." in name:
                later_layer_quantized.append(name)

        assert layer0_quantized, "Expected at least one quantized module in layer 0."
        assert not later_layer_quantized, (
            "Expected quantization only in layer 0, "
            f"but found later-layer modules: {later_layer_quantized[:8]}"
        )

    def _run_layer0_only_protocol_eval(self):
        cfg = self._build_quantize_config()
        self._assert_layer0_only_dynamic(cfg)

        self.model, _, _ = self.quantModel(
            self.NATIVE_MODEL_ID,
            batch_size=self.QUANT_BATCH_SIZE,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
            call_perform_post_quant_validation=False,
        )
        self.check_kernel(self.model, self.KERNEL_INFERENCE)
        self._assert_only_first_layer_quantized(self.model)

        eval_records = getattr(self, "_post_quant_eval_records", {})
        target_backend = self._current_load_backend()
        if eval_records and target_backend in eval_records:
            task_results = eval_records[target_backend]
        else:
            task_results = self.evaluate_model(
                model=self.SAVE_PATH if self.SAVE_PATH else self.model,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                delete_quantized_model=self.DELETE_QUANTIZED_MODEL,
            )
        self.check_results(task_results)
        self._cleanup_quantized_model(self.model, enabled=self.DELETE_QUANTIZED_MODEL)


class TestLlama3_2_AWQProtocolPython(_BaseLlama3_2AWQProtocol):
    def _compiled_protocol_plan(self):
        return compile_protocol(_python_protocol())

    def test_llama3_2_awq_protocol_python(self):
        self._run_layer0_only_protocol_eval()


class TestLlama3_2_AWQProtocolYAML(_BaseLlama3_2AWQProtocol):
    def _compiled_protocol_plan(self):
        return compile_protocol_yaml_text(_yaml_protocol())

    def test_llama3_2_awq_protocol_yaml(self):
        self._run_layer0_only_protocol_eval()
