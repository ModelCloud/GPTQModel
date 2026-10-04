# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from model_test import ModelTest
from safetensors import safe_open
from w4a_gb10_memory import require_w4a_test_headroom

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.w4a_activation import pack_activation
from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear


class TestLlama3_2_W4AFP8(ModelTest):
    NATIVE_MODEL_ID = os.environ.get(
        "GPTQMODEL_LLAMA3_2_MODEL",
        "/monster/data/model/Llama-3.2-1B-Instruct",
    )
    SAVE_PATH = os.environ.get("GPTQMODEL_W4AFP8_SAVE_PATH")
    DELETE_QUANTIZED_MODEL = False
    # This validates the W4A lifecycle, so retain every decoder layer even
    # when the shared model-test harness runs in fast mode.
    MODEL_COMPAT_FAST_LAYER_COUNT = "all"
    ACTIVATION = "w4afp8"
    DATASET_SIZE_FAST = int(os.environ.get("GPTQMODEL_W4A_CALIBRATION_ROWS", "32"))
    DATASET_CONCAT_SIZE_FAST = int(os.environ.get("GPTQMODEL_W4A_CALIBRATION_CONCAT_SIZE", "2048"))
    EVAL_BATCH_SIZE = 8
    OFFLOAD_TO_DISK = False
    ACT_GROUP_AWARE = False
    QUANT_BACKEND = BACKEND.GPTQ_W4AFP8
    LOAD_BACKEND = BACKEND.GPTQ_W4AFP8
    KERNEL_QUANT = {W4AFP8Linear}
    KERNEL_INFERENCE = {W4AFP8Linear}
    def test_llama3_2_w4afp8(self):
        require_w4a_test_headroom(require_scope=True)
        save_only = os.environ.get("GPTQMODEL_W4A_TEST_PHASE") == "quant-save"
        no_replay = os.environ.get("GPTQMODEL_W4A_TEST_SKIP_REPLAY") == "1"
        with ExitStack() as stack:
            if no_replay:
                from gptqmodel.looper import gptq_processor
                from gptqmodel.nn_modules.qlinear import w4a_llama_replay

                stack.enter_context(patch.object(w4a_llama_replay, "install_w4a_llama_replay", lambda *_: None))
                stack.enter_context(patch.object(gptq_processor, "enable_w4afp8_replay", lambda *_: None))
            stack.enter_context(self.model_compat_test_context())
            model, _, _ = self.quantModel(
                self.NATIVE_MODEL_ID,
                batch_size=self.QUANT_BATCH_SIZE,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                need_eval=False,
                call_perform_post_quant_validation=False,
                reload_after_save=not save_only,
            )
        if save_only:
            path = Path(self.SAVE_PATH)
            self.assertTrue((path / "model.safetensors").is_file())
            config = json.loads((path / "quantize_config.json").read_text())
            self.assertEqual(config["activation"]["version"], 3)
            with safe_open(path / "model.safetensors", framework="pt") as saved:
                self.assertEqual(sum(key.endswith(".qweight") for key in saved.keys()), 112)
            return
        self.check_kernel(model, self.KERNEL_INFERENCE)
        self.assertEqual(sum(isinstance(module, W4AFP8Linear)
                             for module in model.model.modules()), 112)
        layer = next(module for module in model.model.modules() if isinstance(module, W4AFP8Linear))
        x = torch.randn(1, layer.in_features, device=layer.qweight.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            output = layer(pack_activation(x, self.ACTIVATION))
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(output.shape, (1, layer.out_features))
        self.assertTrue(bool(torch.isfinite(output).all()))
