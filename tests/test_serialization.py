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
    AWQConfig,
    GGUFConfig,
    GPTAQConfig,
    HessianConfig,
    VramStrategy,
)
from gptqmodel.quantization.adjacent_model import AdjacentModelConfig  # noqa: E402


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
            "scale_search",
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
        self.assertEqual(meta["scale_search"], cfg.scale_search.value)
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

    def test_awq_scale_search_chunked_activations_roundtrip(self):
        qkv_pattern = r".*\.self_attn\.(q_proj|k_proj|v_proj)$"
        cfg = AWQConfig(
            scale_search_chunked_activations=False,
            scale_search_refine_steps=7,
            dynamic={qkv_pattern: {"scale_search_refine_steps": 4}},
        )

        payload = cfg.to_dict()
        meta = payload.get("meta")
        self.assertIsInstance(meta, dict)
        self.assertIn("scale_search_chunked_activations", meta)
        self.assertFalse(meta["scale_search_chunked_activations"])
        self.assertEqual(meta["scale_search_refine_steps"], 7)
        self.assertEqual(payload["dynamic"][qkv_pattern]["scale_search_refine_steps"], 4)

        loaded = QuantizeConfig.from_quant_config(payload)
        self.assertIsInstance(loaded, AWQConfig)
        self.assertFalse(loaded.scale_search_chunked_activations)
        self.assertEqual(loaded.scale_search_refine_steps, 7)
        self.assertEqual(loaded.dynamic[qkv_pattern]["scale_search_refine_steps"], 4)

    def test_adjacent_model_saved_config_roundtrip(self):
        adjacent = AdjacentModelConfig(
            coordinate_starts=("linear", "nearest"),
            max_coordinate_flips=19,
            executor="auto",
            cpu_workers=8,
            activation_chunk_size=257,
            native_refinements_per_module=2,
            selection_tolerance=4e-7,
        )
        adjacent.record({"status": "must-not-serialize"})
        expected = adjacent.to_dict()
        cases = {
            "gptq": QuantizeConfig(
                bits=4,
                group_size=128,
                desc_act=False,
                adjacent_model=adjacent,
                offload_to_disk=False,
            ),
            "awq": AWQConfig(
                bits=4,
                group_size=128,
                adjacent_model=adjacent,
                offload_to_disk=False,
            ),
        }

        for method, cfg in cases.items():
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmpdir:
                cfg.save_pretrained(tmpdir)
                with open(os.path.join(tmpdir, "quantize_config.json"), encoding="utf-8") as config_file:
                    payload = json.load(config_file)

                self.assertNotIn("adjacent_model", payload)
                self.assertEqual(payload["meta"]["adjacent_model"], expected)
                loaded = QuantizeConfig.from_pretrained(tmpdir)
                self.assertIsInstance(loaded.adjacent_model, AdjacentModelConfig)
                self.assertEqual(loaded.adjacent_model.to_dict(), expected)
                self.assertEqual(loaded.adjacent_model.snapshot(), [])
                loaded.adjacent_model = None
                self.assertNotIn("adjacent_model", loaded.to_dict().get("meta", {}))

    def test_awq_scale_search_refine_steps_validation(self):
        for invalid in (-1, 1, 1.5, True):
            with self.subTest(scope="global", invalid=invalid):
                with self.assertRaisesRegex(ValueError, "scale_search_refine_steps"):
                    AWQConfig(scale_search_refine_steps=invalid)
            with self.subTest(scope="dynamic", invalid=invalid):
                with self.assertRaisesRegex(ValueError, "dynamic `scale_search_refine_steps`"):
                    AWQConfig(dynamic={r".*\.q_proj$": {"scale_search_refine_steps": invalid}})

        self.assertEqual(AWQConfig().scale_search_refine_steps, 0)
        self.assertEqual(AWQConfig(scale_search_refine_steps=0).scale_search_refine_steps, 0)
        self.assertEqual(AWQConfig(scale_search_refine_steps=2).scale_search_refine_steps, 2)
        self.assertEqual(
            AWQConfig(dynamic={r".*\.q_proj$": {"scale_search_refine_steps": 2}})
            .dynamic[r".*\.q_proj$"]["scale_search_refine_steps"],
            2,
        )
