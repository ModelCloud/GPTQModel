# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.models import auto
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import AutoModuleDecoderConfig
from gptqmodel.quantization.dtype import get_device_dtype_support
from gptqmodel.utils.torch import torch_empty_cache


FIRST_LAYER_ONLY_NEGATIVE_MATCH = r"^model\.layers\.(?!0\.)\d+\."
# Keep the regression on layer 0 so the 70B checkpoint validates quickly on one GPU.


class TestLlama3_3FP4AutoDecoder(ModelTest):
    """Verify GPTQ can quantize one real FP4 Llama layer through the auto-decoder path."""

    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.3-70B-Instruct-FP4"
    LOAD_BACKEND = BACKEND.TORCH
    QUANT_BACKEND = BACKEND.TORCH
    TORCH_DTYPE = "bfloat16"
    USE_FLASH_ATTN = False
    QUANT_BATCH_SIZE = 1
    DATASET_SIZE = 4
    DATASET_CONCAT_SIZE = 1024
    OFFLOAD_TO_DISK = True
    DYNAMIC = {
        f"-:{FIRST_LAYER_ONLY_NEGATIVE_MATCH}": {},
    }

    def _build_quantize_config(self):
        """Attach the auto-decoder preprocessor to the standard GPTQ config."""

        cfg = super()._build_quantize_config()
        cfg.preprocessors = [
            AutoModuleDecoderConfig(
                target_dtype=torch.bfloat16,
            )
        ]
        cfg.wait_for_submodule_finalizers = True
        return cfg

    def _expected_forward_mode(self) -> str:
        """Mirror runtime device capability checks so the assertion stays future-proof."""

        support = get_device_dtype_support(torch.device("cuda"), validate=False)
        if hasattr(torch, "float4_e2m1fn_x2") and torch.float4_e2m1fn_x2 in support.advertised_linear_dtypes:
            return "native"
        return "decode"

    def _assert_only_first_layer_quantized(self, model) -> None:
        """Ensure the debug-short-circuited quantization run touched only layer 0."""

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

    def test_llama3_3_fp4_auto_decoder_quantizes_first_layer(self) -> None:
        """Run one real 70B FP4 quantization layer and verify the auto-decoder path."""

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

            dataset = self.load_dataset(model.tokenizer, rows=self.DATASET_SIZE)

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
                # The layer callback intentionally stops after layer 0 once the FP4 decode path is observed.
                pass

            self._assert_only_first_layer_quantized(model)

            events = [
                entry
                for entry in getattr(model, "auto_module_decoder_events", [])
                if entry["module"].startswith("model.layers.0.")
            ]
            assert events, "Expected layer-0 auto-decoder events for Llama-3.3-70B-Instruct-FP4."
            assert all(entry["target_dtype"] == "bfloat16" for entry in events)

            expected_mode = self._expected_forward_mode()
            if expected_mode == "native":
                assert any(entry["forward_mode"] == "native" for entry in events), (
                    f"Expected at least one native FP4 forward event, got {events[:8]}"
                )
            else:
                assert all(entry["forward_mode"] == "decode" for entry in events), (
                    f"Expected decode-only events, got {events[:8]}"
                )
        finally:
            del dataset
            del model
            torch_empty_cache()
