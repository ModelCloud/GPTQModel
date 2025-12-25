# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT, FORMAT_FIELD_CHECKPOINT, QuantizeConfig  # noqa: E402
from gptqmodel.quantization.config import VramStrategy  # noqa: E402


class TestSerialization(unittest.TestCase):
    MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def test_marlin_local_serialization(self):
        model = GPTQModel.load(self.MODEL_ID, device="cuda:0", backend=BACKEND.MARLIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "model.safetensors")))

            model = GPTQModel.load(tmpdir, device="cuda:0", backend=BACKEND.MARLIN)

    def test_gptq_v1_to_v2_runtime_convert(self):
        model = GPTQModel.load(self.MODEL_ID, device="cuda:0", backend=BACKEND.EXLLAMA_V2)
        self.assertEqual(model.quantize_config.runtime_format, FORMAT.GPTQ_V2)

    def test_gptq_v1_serialization(self):
        model = GPTQModel.load(self.MODEL_ID, device="cuda:0")
        model.quantize_config.format = FORMAT.GPTQ

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            with open(os.path.join(tmpdir, "quantize_config.json"), "r") as f:
                quantize_config = json.load(f)

            self.assertEqual(quantize_config[FORMAT_FIELD_CHECKPOINT], "gptq")

    def test_quantize_config_meta_only_fields_serialization(self):
        cfg = QuantizeConfig(
            gptaq=True,
            gptaq_alpha=0.75,
            gptaq_memory_device="cpu",
            offload_to_disk=True,
            offload_to_disk_path="./offload-test",
            pack_impl="gpu",
            mse=0.125,
            mock_quantization=True,
            hessian_chunk_size=256,
            hessian_chunk_bytes=4096,
            hessian_use_bfloat16_staging=True,
            vram_strategy=VramStrategy.BALANCED,
        )

        payload = cfg.to_dict()
        meta = payload.get("meta")
        self.assertIsInstance(meta, dict)

        meta_only_fields = [
            "failsafe",
            "gptaq",
            "gptaq_alpha",
            "gptaq_memory_device",
            "offload_to_disk",
            "offload_to_disk_path",
            "pack_impl",
            "mse",
            "mock_quantization",
            "act_group_aware",
            "hessian_chunk_size",
            "hessian_chunk_bytes",
            "hessian_use_bfloat16_staging",
            "vram_strategy",
        ]
        for field in meta_only_fields:
            self.assertNotIn(field, payload)
            self.assertIn(field, meta)

        self.assertEqual(meta["gptaq"], cfg.gptaq)
        self.assertEqual(meta["gptaq_alpha"], cfg.gptaq_alpha)
        self.assertEqual(meta["gptaq_memory_device"], cfg.gptaq_memory_device)
        self.assertEqual(meta["offload_to_disk"], cfg.offload_to_disk)
        self.assertEqual(meta["offload_to_disk_path"], cfg.offload_to_disk_path)
        self.assertEqual(meta["pack_impl"], cfg.pack_impl)
        self.assertEqual(meta["mse"], cfg.mse)
        self.assertEqual(meta["mock_quantization"], cfg.mock_quantization)
        self.assertEqual(meta["act_group_aware"], cfg.act_group_aware)
        self.assertEqual(meta["hessian_chunk_size"], cfg.hessian_chunk_size)
        self.assertEqual(meta["hessian_chunk_bytes"], cfg.hessian_chunk_bytes)
        self.assertEqual(meta["hessian_use_bfloat16_staging"], cfg.hessian_use_bfloat16_staging)
        self.assertEqual(meta["vram_strategy"], cfg.vram_strategy.value)
