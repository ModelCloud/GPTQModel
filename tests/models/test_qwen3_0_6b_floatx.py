# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Small native-floatx W4A16 end-to-end regressions.

The official Qwen3 0.6B FP8 and NVFP4 checkpoints make this suite practical on
one GPU.  Fast mode quantizes the final two decoder layers, saves the W4A16
artifact, reloads it with TorchLinear, then evaluates GSM8K Platinum through
Evalution.  The score check compares the W4A16 artifact with a dense BF16
decode of the same native source across 64 GSM8K Platinum rows.
"""

from __future__ import annotations

import os
import sys

from gptqmodel import BACKEND


TESTS_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TESTS_MODELS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_MODELS_ROOT)

from model_test import ModelTest


CALIBRATION_DATASET = [
    "Solve the arithmetic word problem carefully and provide the final numeric answer.",
    "Reason step by step about a math problem, then end with a short final answer.",
    "Explain why GPTQ retains FP32 Hessian accumulation during calibration.",
    "Describe how a native FP8 source differs from a decoded BF16 quantization source.",
    "Explain what a mutex protects in concurrent software.",
    "Give the result of adding twenty-seven and fifteen.",
    "Summarize why model weights can be quantized one module at a time.",
    "Explain the difference between a list and a dictionary.",
] * 2


class _Qwen3_0_6BFloatxBase:
    DELETE_QUANTIZED_MODEL = False
    LOAD_BACKEND = BACKEND.TORCH
    QUANT_BACKEND = BACKEND.TORCH
    PIN_CUDA_DEVICE = 0
    TORCH_DTYPE = "bfloat16"
    USE_FLASH_ATTN = False
    QUANT_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = "auto"
    DATASET_SIZE_FAST = 16
    DATASET_CONCAT_SIZE_FAST = 512
    OFFLOAD_TO_DISK = True
    MODEL_COMPAT_FAST_LAYER_COUNT = 2
    MODEL_COMPAT_FAST_LAYER_POSITION = "last"

    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "eager",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 8,
                "max_rows": 64,
                "max_new_tokens": 256,
                "stream": True,
            },
        },
    }

    @classmethod
    def load_dataset(cls, tokenizer=None, rows: int = 0):
        del tokenizer
        return CALIBRATION_DATASET[:rows] if rows > 0 else list(CALIBRATION_DATASET)

    def test_native_floatx_w4a16_gsm8k_platinum(self):
        self.quantize_and_evaluate()

    def check_results(self, task_results):
        """Validate the smoke metric without dividing by a zero score."""

        metric = task_results.get("gsm8k_platinum_cot", {}).get("acc,num")
        self.assertIsNotNone(metric, "GSM8K Platinum did not return acc,num")
        self.assertGreaterEqual(metric, 0.0)
        self.assertLessEqual(metric, 1.0)

        expected_value = self.DENSE_GSM8K_PLATINUM_ACC
        # This fixed 64-row evaluation moves in 1/64 increments.  Permit the
        # three-answer FP8 variance observed in the matched native-source
        # run, but fail a larger W4A16 regression.
        regression_floor = expected_value - (3 / 64)
        self.assertGreaterEqual(
            metric,
            regression_floor,
            f"GSM8K Platinum W4A16 score {metric} fell below the dense-source "
            f"regression floor {regression_floor} (baseline={expected_value})",
        )


class TestQwen3_0_6BFP8(_Qwen3_0_6BFloatxBase, ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-0.6B-FP8"
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_QWEN3_0_6B_FP8_SAVE_PATH",
        "/tmp/qwen3_0_6b_fp8_w4a16_g128",
    )
    # Dense BF16 decode of this exact source, Evalution 0.0.17, first 64 rows,
    # suite batch_size=8, greedy generation, eager attention, CUDA:0. Matched
    # E2E result: 12/64.
    DENSE_GSM8K_PLATINUM_ACC = 0.1875


class TestQwen3_0_6BNVFP4(_Qwen3_0_6BFloatxBase, ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-0.6B-NVFP4"
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_QWEN3_0_6B_NVFP4_SAVE_PATH",
        "/tmp/qwen3_0_6b_nvfp4_w4a16_g128",
    )
    # Same matched dense-source reference configuration as the FP8 case: 16/64.
    DENSE_GSM8K_PLATINUM_ACC = 0.25
