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
    reason="CUDA with FP8 dtypes is required for AWQ W4A8 smoke coverage.",
)
class TestLlama3_2_AWQ_W4A8(ModelTest):
    """Smoke-test the dedicated user-facing AWQ W4A8 lifecycle on Llama 3.2 1B Instruct."""

    SAVE_PATH = os.environ.get(
        "GPTQMODEL_LLAMA3_2_AWQ_W4A8_SAVE_PATH",
        "/tmp/llama3_2_awq_w4a8_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    TORCH_DTYPE = torch.float16
    DATASET_SIZE = 64
    DATASET_CONCAT_SIZE = 512
    GENERATE_EVAL_SIZE_MIN = 32
    GENERATE_EVAL_SIZE_MAX = 32

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
    ACTIVATION = {
        "method": "fp8",
        "format": "f8_e4m3",
    }

    def test_llama3_2_awq_w4a8_smoke(self):
        model, tokenizer, _ = self.quantModel(
            self.NATIVE_MODEL_ID,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
            need_eval=False,
            call_perform_post_quant_validation=False,
        )
        try:
            self.check_kernel(model, self.KERNEL_INFERENCE)
            self.assertInference(model, tokenizer)
        finally:
            self._cleanup_quantized_model(model, enabled=self.DELETE_QUANTIZED_MODEL)
