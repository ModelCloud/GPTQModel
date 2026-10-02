# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Quantize a plain GPTQ W4A16 control with matched calibration settings."""

import os
from pathlib import Path

from model_test import ModelTest
from safetensors import safe_open
from w4a_gb10_memory import require_w4a_test_headroom

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear


class TestLlama3_2_W4A16Reference(ModelTest):
    NATIVE_MODEL_ID = os.environ.get(
        "GPTQMODEL_LLAMA3_2_MODEL", "/monster/data/model/Llama-3.2-1B-Instruct",
    )
    SAVE_PATH = os.environ.get("GPTQMODEL_W4A16_SAVE_PATH")
    DELETE_QUANTIZED_MODEL = False
    # Controls must quantize the same complete decoder as the W4A lanes.
    MODEL_COMPAT_FAST_LAYER_COUNT = "all"
    DATASET_SIZE_FAST = int(os.environ.get("GPTQMODEL_W4A_CALIBRATION_ROWS", "256"))
    DATASET_CONCAT_SIZE_FAST = int(os.environ.get("GPTQMODEL_W4A_CALIBRATION_CONCAT_SIZE", "2048"))
    OFFLOAD_TO_DISK = False
    ACT_GROUP_AWARE = False
    QUANT_BACKEND = BACKEND.GPTQ_TRITON
    LOAD_BACKEND = BACKEND.GPTQ_TRITON
    KERNEL_QUANT = {TritonV2Linear}

    def test_quantize_save(self):
        require_w4a_test_headroom(require_scope=True)
        with self.model_compat_test_context():
            self.quantModel(
                self.NATIVE_MODEL_ID,
                batch_size=1,
                trust_remote_code=False,
                dtype="auto",
                need_eval=False,
                call_perform_post_quant_validation=False,
                reload_after_save=False,
            )
        self.assertTrue((Path(self.SAVE_PATH) / "model.safetensors").is_file())
        with safe_open(Path(self.SAVE_PATH) / "model.safetensors", framework="pt") as saved:
            self.assertEqual(sum(key.endswith(".qweight") for key in saved.keys()), 112)
