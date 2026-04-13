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
from gptqmodel.quantization import (  # noqa: E402
    FORMAT,
    FORMAT_FIELD_CHECKPOINT,
    FORMAT_FIELD_CODE,
    METHOD_FIELD_CODE,
    QuantizeConfig,
)
from gptqmodel.quantization.config import (  # noqa: E402  # noqa: E402
    METHOD,
    GGUFConfig,
    GPTAQConfig,
    HessianConfig,
    VramStrategy,
)


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

            self.assertEqual(quantize_config[METHOD_FIELD_CODE], "gptq")
            self.assertEqual(quantize_config["quant_method"], "gptq")
            self.assertEqual(quantize_config[FORMAT_FIELD_CODE], "gptq")
            self.assertEqual(quantize_config[FORMAT_FIELD_CHECKPOINT], "gptq")

    def test_legacy_checkpoint_format_load_normalizes_to_format(self):
        cfg = QuantizeConfig.from_quant_config(
            {
                "bits": 4,
                "checkpoint_format": "gguf",
            }
        )

        self.assertIsInstance(cfg, GGUFConfig)
        self.assertEqual(cfg.format, "q_0")
        self.assertEqual(cfg.method, METHOD.GGUF)
        self.assertEqual(cfg.quant_method, METHOD.GGUF)

    def test_quantize_config_meta_only_fields_serialization(self):
        cfg = QuantizeConfig(
            gptaq=GPTAQConfig(alpha=0.75, device="cpu"),
            offload_to_disk=True,
            offload_to_disk_path="./offload-test",
            pack_impl="gpu",
            mse=0.125,
            mock_quantization=True,
            hessian=HessianConfig(
                chunk_size=256,
                chunk_bytes=4096,
                staging_dtype=torch.bfloat16,
            ),
            dense_vram_strategy=VramStrategy.BALANCED,
            dense_vram_strategy_devices=["cuda:0", "cuda:1"],
            moe_vram_strategy=VramStrategy.BALANCED,
            moe_vram_strategy_devices=["cuda:2", "cuda:3"],
        )

        payload = cfg.to_dict()
        meta = payload.get("meta")
        self.assertIsInstance(meta, dict)

        meta_only_fields = [
            "fallback",
            "gptaq",
            "offload_to_disk",
            "offload_to_disk_path",
            "pack_impl",
            "mse",
            "mock_quantization",
            "act_group_aware",
            "hessian",
            "dense_vram_strategy",
            "dense_vram_strategy_devices",
            "moe_vram_strategy",
            "moe_vram_strategy_devices",
        ]
        for field in meta_only_fields:
            self.assertNotIn(field, payload)
            self.assertIn(field, meta)

        self.assertEqual(meta["gptaq"]["alpha"], cfg.gptaq.alpha)
        self.assertEqual(meta["gptaq"]["device"], cfg.gptaq.device)
        self.assertEqual(meta["offload_to_disk"], cfg.offload_to_disk)
        self.assertEqual(meta["offload_to_disk_path"], cfg.offload_to_disk_path)
        self.assertEqual(meta["pack_impl"], cfg.pack_impl)
        self.assertEqual(meta["mse"], cfg.mse)
        self.assertEqual(meta["mock_quantization"], cfg.mock_quantization)
        self.assertEqual(meta["act_group_aware"], cfg.act_group_aware)
        self.assertEqual(meta["hessian"]["chunk_size"], cfg.hessian.chunk_size)
        self.assertEqual(meta["hessian"]["chunk_bytes"], cfg.hessian.chunk_bytes)
        self.assertEqual(meta["hessian"]["staging_dtype"], "bfloat16")
        self.assertEqual(meta["dense_vram_strategy"], cfg.dense_vram_strategy.value)
        self.assertEqual(meta["dense_vram_strategy_devices"], cfg.dense_vram_strategy_devices)
        self.assertEqual(meta["moe_vram_strategy"], cfg.moe_vram_strategy.value)
        self.assertEqual(meta["moe_vram_strategy_devices"], cfg.moe_vram_strategy_devices)

    def test_gptaq_config_none_serialization(self):
        cfg = QuantizeConfig()

        payload = cfg.to_dict()
        meta = payload.get("meta")
        self.assertIsInstance(meta, dict)
        self.assertIn("gptaq", meta)
        self.assertIsNone(meta["gptaq"])
