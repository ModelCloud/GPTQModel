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
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not hasattr(torch, "float8_e4m3fn")),
    reason="CUDA with FP8 dtypes is required for AWQ W4A8 Evalution coverage.",
)
class TestLlama3_2_AWQ_W4A8(ModelTest):
    """Quantize and Evalution-test the dedicated AWQ W4A8 lifecycle on Llama 3.2 1B Instruct."""

    SAVE_PATH = os.environ.get(
        "GPTQMODEL_LLAMA3_2_AWQ_W4A8_SAVE_PATH",
        "/tmp/llama3_2_awq_w4a8_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    TORCH_DTYPE = torch.float16
    EVAL_BATCH_SIZE = 64
    DATASET_SIZE = 64
    DATASET_CONCAT_SIZE = 512

    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ
    BITS = 4
    GROUP_SIZE = 128
    SYM = True
    # Use AUTO explicitly so the smoke validates the dedicated AWQ W4A8 default
    # routing instead of a manually forced backend.
    QUANT_BACKEND = BACKEND.AUTO
    LOAD_BACKEND = BACKEND.AUTO
    KERNEL_QUANT = {AwqTorchLinear}
    KERNEL_INFERENCE = {AwqTorchLinear}
    INPUT_ACTIVATIONS = {
        "dtype": "float8_e4m3fn",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True,
    }
    # GPU 0 on this host reports as the Ampere datacenter board `PG506-230`,
    # and ModelTest resolves that path through the shared A100-class baseline.
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
                "value": {
                    "A100": 0.3316790736145575,
                    "RTX4090": 0.3242349048800662,
                },
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
                "batch_size": 32,
                "max_new_tokens": 256,
                "stream": True,
                "max_rows": 128,
            },
            "acc,num": {
                "value": {
                    "A100": 0.515625,
                    "RTX4090": 0.5,
                },
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }

    def _assert_static_activation_scales(self, model):
        """Ensure the reloaded AWQ W4A8 modules kept their calibrated static FP8 input scales."""

        qmodules = [module for _, module in model.named_modules() if isinstance(module, AwqTorchLinear)]
        self.assertTrue(qmodules, "Expected at least one AWQ linear module in the quantized model.")

        first = qmodules[0]
        self.assertIsNotNone(first.input_activations)
        self.assertFalse(first.input_activations.dynamic)
        self.assertTrue(hasattr(first, "input_scale_inv"))
        self.assertGreater(float(first.input_scale_inv.item()), 0.0)

    def test_llama3_2_awq_w4a8(self):
        self.quantize_and_evaluate()
        self._assert_static_activation_scales(self.model)
