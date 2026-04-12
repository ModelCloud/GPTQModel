# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.quantization import FORMAT, METHOD, GPTQConfig
from gptqmodel.quantization.protocol import (
    Rule,
    Stage,
    compile_plan_to_quantize_config,
    compile_protocol,
    compile_protocol_yaml_text,
)


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
                                "method": "gptq",
                                "bits": 4,
                                "group_size": 128,
                                "sym": True,
                                "desc_act": False,
                            },
                            "export": {
                                "format": "gptq",
                                "variant": "gptq",
                                "impl": "marlin",
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
            method: gptq
            bits: 4
            group_size: 128
            sym: true
            desc_act: false
          export:
            format: gptq
            variant: gptq
            impl: marlin
"""


class _BaseLlama3_2GPTQProtocol(ModelTest):
    pytestmark = pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="CUDA is required for protocol GPTQ integration tests",
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
    LOAD_BACKEND = BACKEND.MARLIN
    KERNEL_INFERENCE = {MarlinLinear}

    def _compiled_protocol_plan(self):
        raise NotImplementedError

    def _build_quantize_config(self):
        return compile_plan_to_quantize_config(self._compiled_protocol_plan())

    def _assert_layer0_only_dynamic(self, cfg):
        assert isinstance(cfg, GPTQConfig)
        assert cfg.quant_method == METHOD.GPTQ
        assert cfg.format == FORMAT.GPTQ
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
        )
        self.check_kernel(self.model, self.KERNEL_INFERENCE)
        self._assert_only_first_layer_quantized(self.model)

        eval_records = getattr(self, "_post_quant_eval_records", {})
        target_backend = self._current_load_backend()
        if eval_records and len(eval_records) == 1 and target_backend in eval_records:
            task_results = eval_records[target_backend]
        else:
            task_results = self.evaluate_model(
                model=self.SAVE_PATH if self.SAVE_PATH else self.model,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                delete_quantized_model=self.DELETE_QUANTIZED_MODEL,
            )
        self.check_results(task_results)
        self._cleanup_quantized_model(self.model, enabled=self.DELETE_QUANTIZED_MODEL)


class TestLlama3_2_GPTQProtocolPython(_BaseLlama3_2GPTQProtocol):
    def _compiled_protocol_plan(self):
        return compile_protocol(_python_protocol())

    def test_llama3_2_gptq_protocol_python(self):
        self._run_layer0_only_protocol_eval()


class TestLlama3_2_GPTQProtocolYAML(_BaseLlama3_2GPTQProtocol):
    def _compiled_protocol_plan(self):
        return compile_protocol_yaml_text(_yaml_protocol())

    def test_llama3_2_gptq_protocol_yaml(self):
        self._run_layer0_only_protocol_eval()
