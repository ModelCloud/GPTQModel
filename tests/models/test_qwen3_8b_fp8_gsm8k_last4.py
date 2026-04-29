# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import torch


TESTS_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TESTS_MODELS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_MODELS_ROOT)

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.quantization import AutoModuleDecoderConfig


LAST_FOUR_ONLY_NEGATIVE_MATCH = r"^model\.layers\.(?:[0-2]?\d|3[0-1])\."
# Qwen3-8B-FP8 exposes 36 decoder layers; skip 0-31 so only the last four are quantized.


def _gsm8k_expected_acc() -> float:
    return float(os.environ.get("GPTQMODEL_QWEN3_8B_FP8_LAST4_GSM8K_ACC", "0.0"))


class TestQwen3_8B_FP8_Gsm8kLast4(ModelTest):
    # Keep one stable saved checkpoint so eval-only repro runs can reuse the exact post-quant model.
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_QWEN3_8B_FP8_LAST4_SAVE_PATH",
        "/tmp/qwen3_8b_fp8_last4_gptq_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/mnt/SFS-6CFyUykx/models/Qwen3-8B-FP8"
    LOAD_BACKEND = BACKEND.TORCH
    QUANT_BACKEND = BACKEND.TORCH
    PIN_CUDA_DEVICE = 0
    TORCH_DTYPE = "bfloat16"
    USE_FLASH_ATTN = False
    QUANT_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 32
    DATASET_SIZE = 32
    DATASET_CONCAT_SIZE = 2048
    OFFLOAD_TO_DISK = True
    DYNAMIC = {
        f"-:{LAST_FOUR_ONLY_NEGATIVE_MATCH}": {},
    }
    CALIBRATION_DATASET = [
        "Solve the arithmetic word problem carefully and provide the final numeric answer.",
        "Reason step by step about a math problem, then end with a short final answer.",
        "Explain the difference between calibration data and evaluation data in quantization workflows.",
        "Summarize the tradeoffs between FP8 checkpoints and dense bfloat16 weights for inference.",
        "Describe why a decoder-only language model may need left padding during batched generation.",
        "Write a concise explanation of how GPTQ quantization uses calibration activations.",
        "Explain how Qwen-style chat templates can affect benchmark accuracy when applied incorrectly.",
        "Give a short note on why some FP8 checkpoints store scale and others store inverse scale.",
    ] * 4
    EVAL_TASKS_SLOW = {
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
                "batch_size": 32,
                "max_new_tokens": 256,
                "stream": True,
            },
            "acc,num": {
                "value": _gsm8k_expected_acc(),
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }
    EVAL_TASKS_FAST = EVAL_TASKS_SLOW

    @classmethod
    def load_dataset(cls, tokenizer=None, rows: int = 0):
        del tokenizer
        if rows > 0:
            return cls.CALIBRATION_DATASET[:rows]
        return list(cls.CALIBRATION_DATASET)

    def _build_quantize_config(self):
        cfg = super()._build_quantize_config()
        # The source checkpoint is FP8, so attach the auto-decoder preprocessor before GPTQ quantization.
        cfg.preprocessors = [
            AutoModuleDecoderConfig(target_dtype=torch.bfloat16)
        ]
        cfg.wait_for_submodule_finalizers = True
        return cfg

    def _model_test_mode(self) -> str:
        # This regression intentionally validates a fixed last-4-layer quantization recipe,
        # so opt out of the harness's default fast-mode layer trimming.
        return self.MODEL_TEST_MODE_SLOW

    def test_qwen3_8b_fp8_last4_gsm8k_platinum(self):
        if _gsm8k_expected_acc() <= 0.0:
            self.skipTest(
                "Set GPTQMODEL_QWEN3_8B_FP8_LAST4_GSM8K_ACC to a recorded gsm8k_platinum_cot acc,num baseline."
            )
        self.quant_lm_eval()
