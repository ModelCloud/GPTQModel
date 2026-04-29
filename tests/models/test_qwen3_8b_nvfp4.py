# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import torch
from datasets import load_dataset as hf_load_dataset


TESTS_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TESTS_MODELS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_MODELS_ROOT)

from model_test import ModelTest

from gptqmodel.quantization import AutoModuleDecoderConfig


class TestQwen3_8BNVFP4(ModelTest):
    """End-to-end NVFP4 regression for local Qwen3-8B, quantizing the last four layers."""

    SAVE_PATH = os.environ.get(
        "GPTQMODEL_QWEN3_8B_NVFP4_SAVE_PATH",
        "/tmp/qwen3_8b_nvfp4_last4_gptq_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/mnt/SFS-6CFyUykx/models/Qwen3-8B-NVFP4"
    PIN_CUDA_DEVICE = 0
    TORCH_DTYPE = "bfloat16"
    USE_FLASH_ATTN = False
    QUANT_BATCH_SIZE = 1
    DATASET_CONCAT_SIZE = 2048
    OFFLOAD_TO_DISK = True
    MODEL_COMPAT_FAST_LAYER_COUNT = 4
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
                "batch_size": 16,
                "max_new_tokens": 256,
                "stream": True,
            },
            "acc,num": {
                "value": 0.25,
                "floor_pct": 0.25,
                "ceil_pct": 1.0,
            },
        },
    }

    def _build_quantize_config(self):
        cfg = super()._build_quantize_config()
        cfg.preprocessors = [
            AutoModuleDecoderConfig(
                target_dtype=torch.bfloat16,
            )
        ]
        cfg.wait_for_submodule_finalizers = True
        return cfg

    @classmethod
    def load_dataset(cls, tokenizer=None, rows: int = 0):
        dataset = hf_load_dataset("neuralmagic/calibration", name="LLM", split="train")
        if rows > 0:
            return dataset.select(range(min(rows, len(dataset))))
        return dataset

    def test_qwen3_8b_nvfp4(self):
        self.quant_lm_eval()
