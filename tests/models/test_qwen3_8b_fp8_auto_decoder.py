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
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.models import auto
from gptqmodel.quantization import AutoModuleDecoderConfig
from gptqmodel.quantization.dtype import get_device_dtype_support
from gptqmodel.utils.torch import torch_empty_cache


FIRST_LAYER_ONLY_NEGATIVE_MATCH = r"^model\.layers\.(?!0\.)\d+\."
# Restrict the regression to layer 0 so one visible GPU can validate the mode switch quickly.


class TestQwen3_8BFp8AutoDecoder(ModelTest):
    """Verify FP8 auto-decoder mode selection on one visible GPU for local Qwen3-8B-FP8."""

    NATIVE_MODEL_ID = "/mnt/SFS-6CFyUykx/models/Qwen3-8B-FP8"
    LOAD_BACKEND = BACKEND.TORCH
    QUANT_BACKEND = BACKEND.TORCH
    TORCH_DTYPE = "bfloat16"
    USE_FLASH_ATTN = False
    QUANT_BATCH_SIZE = 1
    DATASET_SIZE = 4
    DATASET_CONCAT_SIZE = 2048
    OFFLOAD_TO_DISK = True
    DYNAMIC = {
        f"-:{FIRST_LAYER_ONLY_NEGATIVE_MATCH}": {},
    }
    CALIBRATION_DATASET = [
        "Explain how rotary position embeddings are applied in decoder-only transformers.",
        "Summarize the tradeoffs between dense weights and FP8 checkpoint storage.",
        "Describe how auto-decoder preprocessing can switch between native and decoded forward paths.",
        "Write a short note on why calibration examples matter for GPTQ quantization.",
    ]

    def _build_quantize_config(self):
        cfg = super()._build_quantize_config()
        cfg.preprocessors = [
            AutoModuleDecoderConfig(
                target_dtype=torch.bfloat16,
            )
        ]
        cfg.wait_for_submodule_finalizers = True
        return cfg

    def _expected_forward_mode(self) -> str:
        support = get_device_dtype_support(torch.device("cuda"), validate=False)
        return "native" if torch.float8_e4m3fn in support.advertised_linear_dtypes else "decode"

    def test_qwen3_8b_fp8_auto_decoder_selects_forward_role_by_gpu_capability(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the Qwen3-8B FP8 auto-decoder test.")

        model = None
        dataset = None
        try:
            quantize_config = self._build_quantize_config()
            quantize_config.device = torch.device("cuda")

            model_definition = auto.check_and_get_model_definition(
                self.NATIVE_MODEL_ID,
                self.TRUST_REMOTE_CODE,
            )
            model = model_definition.from_pretrained(
                pretrained_model_id_or_path=self.NATIVE_MODEL_ID,
                quantize_config=quantize_config,
                backend=self.LOAD_BACKEND,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                attn_implementation="eager",
            )
            model.layer_callback = self._build_layer_stop_callback(0)

            dataset = self.CALIBRATION_DATASET[: self.DATASET_SIZE]

            try:
                model.quantize(
                    dataset,
                    calibration_concat_size=self.DATASET_CONCAT_SIZE,
                    calibration_concat_separator=self.DATASET_CONCAT_SEPARATOR,
                    calibration_sort=self.DATASET_SORT,
                    backend=self.QUANT_BACKEND,
                    batch_size=self.QUANT_BATCH_SIZE,
                )
            except StopMainLoop:
                # The layer callback intentionally stops after layer 0 once the mode decision is observed.
                pass

            events = [
                entry
                for entry in getattr(model, "auto_module_decoder_events", [])
                if entry["module"].startswith("model.layers.0.")
            ]
            self.assertTrue(events, "Expected layer-0 auto-decoder events for Qwen3-8B-FP8.")
            self.assertTrue(all(entry["source_dtype"] == "float8_e4m3fn" for entry in events))
            self.assertTrue(all(entry["target_dtype"] == "bfloat16" for entry in events))

            expected_mode = self._expected_forward_mode()
            if expected_mode == "native":
                self.assertTrue(
                    any(entry["forward_mode"] == "native" for entry in events),
                    f"Expected at least one native FP8 forward event, got {events[:8]}",
                )
            else:
                self.assertTrue(
                    all(entry["forward_mode"] == "decode" for entry in events),
                    f"Expected decode-only events, got {events[:8]}",
                )
        finally:
            del dataset
            del model
            torch_empty_cache()

    def test_qwen3_8b_fp8_auto_decoder_uses_native_on_weight_scale_checkpoint(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the Qwen3-8B FP8 auto-decoder test.")

        support = get_device_dtype_support(torch.device("cuda"), validate=False)
        if torch.float8_e4m3fn not in support.advertised_linear_dtypes:
            self.skipTest("This regression requires a GPU that advertises native FP8 linear support.")

        model = None
        dataset = None
        try:
            quantize_config = self._build_quantize_config()
            quantize_config.device = torch.device("cuda")

            model_definition = auto.check_and_get_model_definition(
                self.NATIVE_MODEL_ID,
                self.TRUST_REMOTE_CODE,
            )
            model = model_definition.from_pretrained(
                pretrained_model_id_or_path=self.NATIVE_MODEL_ID,
                quantize_config=quantize_config,
                backend=self.LOAD_BACKEND,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                attn_implementation="eager",
            )
            model.layer_callback = self._build_layer_stop_callback(0)

            dataset = self.CALIBRATION_DATASET[: self.DATASET_SIZE]

            try:
                model.quantize(
                    dataset,
                    calibration_concat_size=self.DATASET_CONCAT_SIZE,
                    calibration_concat_separator=self.DATASET_CONCAT_SEPARATOR,
                    calibration_sort=self.DATASET_SORT,
                    backend=self.QUANT_BACKEND,
                    batch_size=self.QUANT_BATCH_SIZE,
                )
            except StopMainLoop:
                # The layer callback intentionally stops after layer 0 once the mode decision is observed.
                pass

            events = [
                entry
                for entry in getattr(model, "auto_module_decoder_events", [])
                if entry["module"].startswith("model.layers.0.")
            ]
            self.assertTrue(events, "Expected layer-0 auto-decoder events for Qwen3-8B-FP8.")
            self.assertTrue(all(entry["source_dtype"] == "float8_e4m3fn" for entry in events))
            self.assertTrue(all(entry["target_dtype"] == "bfloat16" for entry in events))
            self.assertTrue(
                all(entry["forward_mode"] == "native" for entry in events),
                f"Expected native events on this weight_scale-backed checkpoint, got {events[:8]}",
            )
        finally:
            del dataset
            del model
            torch_empty_cache()
