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

import torch  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT, FORMAT_FIELD_CHECKPOINT, QuantizeConfig  # noqa: E402
from gptqmodel.quantization.config import GPTAQConfig, VramStrategy  # noqa: E402


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
            gptaq=GPTAQConfig(alpha=0.75, device="cpu"),
            offload_to_disk=True,
            offload_to_disk_path="./offload-test",
            pack_impl="gpu",
            mock_quantization=True,
            process={
                "gptq": {
                    "mse": 0.125,
                    "hessian": {
                        "chunk_size": 256,
                        "chunk_bytes": 4096,
                        "staging_dtype": "bfloat16",
                    },
                }
            },
            vram_strategy=VramStrategy.BALANCED,
        )

        payload = cfg.to_dict()
        meta = payload.get("meta")
        self.assertIsInstance(meta, dict)

        meta_only_fields = [
            "failsafe",
            "gptaq",
            "offload_to_disk",
            "offload_to_disk_path",
            "pack_impl",
            "mock_quantization",
            "vram_strategy",
        ]
        for field in meta_only_fields:
            self.assertNotIn(field, payload)
            self.assertIn(field, meta)

        self.assertEqual(meta["gptaq"]["alpha"], cfg.gptaq.alpha)
        self.assertEqual(meta["gptaq"]["device"], cfg.gptaq.device)
        self.assertEqual(meta["offload_to_disk"], cfg.offload_to_disk)
        self.assertEqual(meta["offload_to_disk_path"], cfg.offload_to_disk_path)
        self.assertEqual(meta["pack_impl"], cfg.pack_impl)
        self.assertEqual(meta["mock_quantization"], cfg.mock_quantization)
        self.assertEqual(meta["vram_strategy"], cfg.vram_strategy.value)

        process_payload = payload.get("process")
        self.assertIsInstance(process_payload, dict)
        self.assertIs(process_payload["gptq"]["act_group_aware"], True)
        self.assertEqual(process_payload["gptq"]["mse"], 0.125)
        self.assertEqual(process_payload["gptq"]["hessian"]["chunk_size"], 256)
        self.assertEqual(process_payload["gptq"]["hessian"]["chunk_bytes"], 4096)
        self.assertEqual(process_payload["gptq"]["hessian"]["staging_dtype"], "bfloat16")

    def test_gptaq_config_none_serialization(self):
        cfg = QuantizeConfig()

        payload = cfg.to_dict()
        meta = payload.get("meta")
        self.assertIsInstance(meta, dict)
        self.assertIn("gptaq", meta)
        self.assertIsNone(meta["gptaq"])
