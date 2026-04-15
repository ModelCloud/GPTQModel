# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.torch import torch_empty_cache


# Reuse the saved fused W4A8 artifact so prompt and Evalution regressions
# exercise the activation-aware loader path instead of re-quantizing.
_DEFAULT_CHECKPOINT_PATH = "/monster/data/model/Llama-3.2-1B-Instruct-AWQ-W4A8-FP8-Dynamic"
# Keep one exact prompt stable across loader and kernel changes.
_SOLAR_MASS_PROMPT = (
    "What is the mass of the Sun? If you don't know the exact value, "
    "at least describe how one would create that equation and Sun's mass composition."
)
# Assert on stable content fragments instead of the entire decode so harmless
# wording drift does not break the regression.
_EXPECTED_SOLAR_MASS_SNIPPETS = (
    "1.989 x 10^30 kilograms",
    "hydrogen and helium",
)
# The third assertion allows modest wording drift while still ensuring the
# model attempts the requested equation/composition explanation.
_EXPECTED_SOLAR_MASS_EXPLANATION_SNIPPETS = (
    "to create an equation",
    "to estimate the mass of the sun",
    "sun's composition",
    "weighted average",
)


class TestLlama3_2_AWQ_W4A8_FP8_Dynamic(ModelTest):
    """Regression harness for the saved Llama 3.2 AWQ W4A8 FP8 dynamic checkpoint on RTX 4090."""

    pytestmark = pytest.mark.skipif(
        (not torch.cuda.is_available()) or (not hasattr(torch, "float8_e4m3fn")),
        reason="CUDA plus float8_e4m3fn support are required for AWQ W4A8 FP8 dynamic regressions.",
    )

    # Reuse the existing saved checkpoint so loader and Evalution tests can be rerun cheaply.
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_LLAMA3_2_AWQ_W4A8_FP8_DYNAMIC_SAVE_PATH",
        _DEFAULT_CHECKPOINT_PATH,
    )
    # Point the harness at the same artifact because this test validates reload, not quantization.
    NATIVE_MODEL_ID = SAVE_PATH
    # Keep the saved artifact on disk for repeated local kernel and Evalution runs.
    DELETE_QUANTIZED_MODEL = False
    TORCH_DTYPE = torch.float16

    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ
    BITS = 4
    GROUP_SIZE = 128
    SYM = True
    # Evalution currently needs the explicit activation-aware AWQ Marlin backend
    # to avoid an upstream auto-backend mismatch before GPTQModel sees the checkpoint.
    LOAD_BACKEND = BACKEND.AWQ_MARLIN
    KERNEL_INFERENCE = {AwqMarlinLinear}
    APPLY_CHAT_TEMPLATE = True
    EVAL_BATCH_SIZE = 32

    # This regression only has a recorded score on RTX 4090, where the fused
    # Ada FP8 activation path is available in the current environment.
    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "float16",
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
                    "RTX4090": 0.3125,
                },
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }
    # Keep slow-mode bounded to the same saved-artifact regression because the
    # fused FP8 coverage is tied to one hardware profile in this lab.
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST

    def _require_saved_4090_checkpoint(self) -> None:
        """Skip unless the saved checkpoint is present and the visible GPU is an RTX 4090."""

        checkpoint_path = Path(self.NATIVE_MODEL_ID)
        if not checkpoint_path.exists():
            self.skipTest(f"Saved AWQ W4A8 FP8 dynamic checkpoint is missing: {checkpoint_path}")

        profile = self._detect_gpu_profile()
        if profile != "RTX4090":
            self.skipTest(
                "This regression requires the visible cuda:0 device to resolve as RTX4090. "
                "Run it with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6` on this host."
            )

    def _load_saved_checkpoint(self, *, backend: BACKEND) -> object:
        """Load the saved checkpoint through the requested backend for prompt and kernel regressions."""

        self._require_saved_4090_checkpoint()
        return self.loadQuantModel(
            self.NATIVE_MODEL_ID,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            backend=backend,
            device_map={"": "cuda:0"},
        )

    def _assert_dynamic_fp8_marlin_metadata(self, model) -> None:
        """Verify the loaded checkpoint stayed on the fused AWQ Marlin FP8 input path."""

        self.check_kernel(model, self.KERNEL_INFERENCE)

        qmodules = [module for _, module in model.named_modules() if isinstance(module, AwqMarlinLinear)]
        self.assertTrue(qmodules, "Expected at least one AwqMarlinLinear module in the loaded checkpoint.")

        first = qmodules[0]
        self.assertIsNotNone(first.input_activations)
        self.assertTrue(first.input_activations.dynamic)
        self.assertEqual(first.input_activations.format, "float8_e4m3fn")
        self.assertEqual(getattr(first, "marlin_input_dtype", None), torch.float8_e4m3fn)

    def test_llama3_2_awq_w4a8_fp8_dynamic_solar_mass_prompt(self) -> None:
        """Run one exact chat-style prompt through the saved checkpoint and assert stable answer fragments."""

        model = None
        try:
            # Use AUTO here so this test also covers the loader's activation-aware
            # backend resolution, not only the explicit Marlin code path.
            model = self._load_saved_checkpoint(backend=BACKEND.AUTO)
            self._assert_dynamic_fp8_marlin_metadata(model)

            tokenizer = model.tokenizer
            inputs, decode_start_idx = self._prepare_generic_inference_inputs(tokenizer, _SOLAR_MASS_PROMPT)
            self.assertIsNotNone(inputs, "Expected chat-template-backed tokenizer inputs for the regression prompt.")
            self.assertIsNotNone(decode_start_idx, "Expected a decode start offset for the regression prompt.")

            response = self.generate_stable_with_limit(
                model,
                tokenizer,
                _SOLAR_MASS_PROMPT,
                inputs=inputs,
                decode_start_idx=decode_start_idx,
                max_new_tokens=192,
            )
            normalized = response.lower()
            for snippet in _EXPECTED_SOLAR_MASS_SNIPPETS:
                self.assertIn(snippet.lower(), normalized)
            self.assertTrue(
                any(snippet in normalized for snippet in _EXPECTED_SOLAR_MASS_EXPLANATION_SNIPPETS),
                f"Expected one explanation fragment from {_EXPECTED_SOLAR_MASS_EXPLANATION_SNIPPETS}, got: {response}",
            )
        finally:
            del model
            torch_empty_cache()

    def test_llama3_2_awq_w4a8_fp8_dynamic_gsm8k_platinum(self) -> None:
        """Record a bounded gsm8k_platinum_cot Evalution baseline for the saved 4090 fused FP8 checkpoint."""

        self._require_saved_4090_checkpoint()
        self.quantize_and_evaluate()
        self._assert_dynamic_fp8_marlin_metadata(self.model)
