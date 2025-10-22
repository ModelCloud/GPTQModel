# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import unittest

import torch

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.machete import MacheteQuantLinear
from gptqmodel.utils.machete import (
    _validate_machete_device_support,
    machete_import_exception,
)
from gptqmodel.utils.model import find_modules


class TestMacheteKernelOutput(unittest.TestCase):
    model_path = "sliuau/Llama-3.2-3B_4bits_128group_size"
    target = "model.layers.6.self_attn.v_proj"

    @classmethod
    def setUpClass(cls):
        if machete_import_exception is not None:
            raise unittest.SkipTest(
                f"Machete kernel unavailable: {machete_import_exception}"
            )
        if not _validate_machete_device_support():
            raise unittest.SkipTest("Machete requires NVIDIA Hopper or newer (SM90+)")
        preferred_index = int(os.getenv("GPTQMODEL_TEST_CUDA_INDEX", "8"))
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible:
            entries = [v.strip() for v in visible.split(",") if v.strip()]
            # If a single device is exposed (e.g., CUDA_VISIBLE_DEVICES=8), map to index 0.
            if len(entries) == 1:
                preferred_index = 0
        if torch.cuda.device_count() == 0:
            raise unittest.SkipTest("CUDA device not available for machete tests.")
        if preferred_index >= torch.cuda.device_count():
            preferred_index = torch.cuda.device_count() - 1
        cls.device_index = preferred_index
        cls.device = torch.device(f"cuda:{cls.device_index}")
        torch.cuda.set_device(cls.device)

    def _load_target_module(self, dtype: torch.dtype):
        model = GPTQModel.load(
            self.model_path,
            backend=BACKEND.MACHETE,
            dtype=dtype,
        )
        modules = find_modules(model.model, layers=[MacheteQuantLinear])
        module = modules.get(self.target)
        self.assertIsNotNone(module, f"Target module `{self.target}` not found")
        return model, module

    def _run_forward(self, module: MacheteQuantLinear, dtype: torch.dtype):
        x = torch.randn((1, module.in_features), device=self.device, dtype=dtype)
        with torch.inference_mode():
            return module(x)

    def test_machete_forward_float16(self):
        model, module = self._load_target_module(torch.float16)
        out = self._run_forward(module, torch.float16)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaNs for float16")
        del module
        del model
        torch.cuda.empty_cache()

    def test_machete_forward_bfloat16(self):
        model, module = self._load_target_module(torch.bfloat16)
        out = self._run_forward(module, torch.bfloat16)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaNs for bfloat16")
        del module
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
