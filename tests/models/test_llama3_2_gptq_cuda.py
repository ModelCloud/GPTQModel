"""Opt-in Llama 3.2 1B quality check for the native CUDA GPTQ block update.

The full quantize-and-evaluate check is selected with ``-m gptq_cuda_e2e``.
The saved-checkpoint-only check is selected with ``-m gptq_cuda_eval_only``;
it reads the checkpoint written by the first run (``SAVE_PATH``).
"""

import os
from unittest.mock import patch

import pytest
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.quantization import gptq_cuda


class TestLlama3_2GPTQCUDA(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    SAVE_PATH = os.environ.get("GPTQMODEL_LLAMA3_2_CUDA_SAVE_PATH", "/tmp/llama3_2_gptq_cuda")
    DELETE_QUANTIZED_MODEL = False
    LOAD_BACKEND = BACKEND.TORCH
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "sdpa",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 8,
                "max_new_tokens": 256,
                "stream": True,
            },
            "acc,num": {"value": 0.4690, "floor_pct": 0.04, "ceil_pct": 1.0},
        },
    }

    @pytest.mark.gptq_cuda_e2e
    def test_gsm8k_platinum_cuda(self):
        if not gptq_cuda.block_update_available():
            self.skipTest(gptq_cuda._EXTENSION.last_error_message())

        original = gptq_cuda.gptq_block_update
        block_calls = 0

        def counted(*args, **kwargs):
            nonlocal block_calls
            block_calls += 1
            return original(*args, **kwargs)

        with patch.object(gptq_cuda, "gptq_block_update", counted):
            self.quantize_and_evaluate()
        self.assertGreater(block_calls, 0, "The native CUDA GPTQ block update was not used")

    @pytest.mark.gptq_cuda_eval_only
    def test_gsm8k_platinum_cuda_eval_only(self):
        self.get_eval_tasks()
        results = self.evaluate_model(model=self.SAVE_PATH, delete_quantized_model=False)
        self.check_results(results)
