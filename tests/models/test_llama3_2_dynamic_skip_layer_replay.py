# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import torch
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear


LAYER0_AND_LAYER2_ONLY_NEGATIVE_MATCH = r"^model\.layers\.(?!(?:0|2)\.)\d+\."


class TestLlama3_2DynamicSkipLayerReplay(ModelTest):
    """Exercise dynamic full-layer skips across a quantized -> skipped -> quantized chain."""

    pytestmark = pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is required for Llama-3.2 GPTQ Marlin integration tests",
    )

    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    # Keep this regression on a single deterministic CUDA device in PCI bus order.
    PIN_CUDA_DEVICE = 0
    LOAD_BACKEND = BACKEND.MARLIN
    KERNEL_INFERENCE = {MarlinQuantLinear}
    DYNAMIC = {
        f"-:{LAYER0_AND_LAYER2_ONLY_NEGATIVE_MATCH}": {},
    }
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.3987,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3234,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3643,
                "floor_pct": 0.04,
            },
        },
    }
    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "paged|flash_attention_2",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 24,
                "max_new_tokens": 96,
                "stream": True,
                "max_rows": 128,
            },
            "acc,num": {
                "value": 0.390625,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3166,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.3430,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }

    def _assert_dynamic_config_targets_only_layers_0_and_2(self, cfg) -> None:
        assert cfg.dynamic == self.DYNAMIC
        assert cfg.dynamic_get("model.layers.0.self_attn.q_proj") is not False
        assert cfg.dynamic_get("model.layers.1.self_attn.q_proj") is False
        assert cfg.dynamic_get("model.layers.2.self_attn.q_proj") is not False
        assert cfg.dynamic_get("model.layers.3.self_attn.q_proj") is False

    def _assert_only_layers_0_and_2_quantized(self, model) -> None:
        quantized_layer_names = {0: [], 1: [], 2: []}
        unexpected_quantized = []

        for name, module in model.named_modules():
            if not isinstance(module, BaseQuantLinear):
                continue

            layer_match = re.search(r"\.layers\.(\d+)\.", name)
            if layer_match is None:
                continue

            layer_idx = int(layer_match.group(1))
            if layer_idx in quantized_layer_names:
                quantized_layer_names[layer_idx].append(name)
            else:
                unexpected_quantized.append(name)

        assert quantized_layer_names[0], "Expected quantized modules in layer 0."
        assert not quantized_layer_names[1], (
            "Layer 1 should be fully skipped by QuantizeConfig.dynamic, "
            f"but found quantized modules: {quantized_layer_names[1][:8]}"
        )
        assert quantized_layer_names[2], "Expected quantized modules in layer 2."
        assert not unexpected_quantized, (
            "Only layers 0 and 2 should be quantized, "
            f"but found additional quantized modules: {unexpected_quantized[:8]}"
        )

    def _run_dynamic_skip_replay_eval(self) -> None:
        cfg = self._build_quantize_config()
        self._assert_dynamic_config_targets_only_layers_0_and_2(cfg)

        self.model, _, _ = self.quantModel(
            self.NATIVE_MODEL_ID,
            batch_size=self.QUANT_BATCH_SIZE,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
        )
        self.check_kernel(self.model, self.KERNEL_INFERENCE)
        self._assert_only_layers_0_and_2_quantized(self.model)

        eval_records = getattr(self, "_post_quant_eval_records", {})
        target_backend = self._current_load_backend()
        if eval_records and len(eval_records) == 1 and target_backend in eval_records:
            task_results = eval_records[target_backend]
        else:
            task_results = self.lm_eval(
                model=self.SAVE_PATH if self.SAVE_PATH else self.model,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                delete_quantized_model=self.DELETE_QUANTIZED_MODEL,
            )

        self.check_results(task_results)
        self._cleanup_quantized_model(self.model, enabled=self.DELETE_QUANTIZED_MODEL)

    def test_llama3_2_dynamic_skip_layer_replay(self):
        self._run_dynamic_skip_replay_eval()
