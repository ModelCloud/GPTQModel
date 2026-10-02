# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import json
from pathlib import Path

import torch
from safetensors import safe_open

from model_test import ModelTest
from w4a_gb10_memory import require_w4a_test_headroom

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.w4a_activation import pack_activation
from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear
from tests.models.w4a_calibration_data import load_calibration_artifact
from tests.models.w4a_nvfp4_norm_qat import _calibration_ids


class TestLlama3_2_W4ANVFP4(ModelTest):
    NATIVE_MODEL_ID = os.environ.get(
        "GPTQMODEL_LLAMA3_2_MODEL",
        "/monster/data/model/Llama-3.2-1B-Instruct",
    )
    SAVE_PATH = os.environ.get("GPTQMODEL_W4A_NVFP4_SAVE_PATH")
    DELETE_QUANTIZED_MODEL = False
    # This validates an end-to-end activation stream rather than the generic
    # two-layer model-compatibility shortcut.
    MODEL_COMPAT_FAST_LAYER_COUNT = "all"
    ACTIVATION_VERSION = int(os.environ.get("GPTQMODEL_W4A_ACTIVATION_VERSION", "3"))
    ACTIVATION_RECIPE = os.environ.get("GPTQMODEL_W4A_NVFP4_RECIPE", "least_squares")
    ACTIVATION = {
        "version": ACTIVATION_VERSION, "mode": "w4a_nvfp4", "recipe": ACTIVATION_RECIPE
    }
    ROTATION = os.environ.get("GPTQMODEL_W4A_ROTATION") or None
    # Acceptance covers plain GPTQ; an inherited environment must not silently
    # select a different quantization algorithm for this checkpoint.
    GPTAQ = None
    FOEM = None
    DATASET_SIZE_FAST = int(os.environ.get("GPTQMODEL_W4A_CALIBRATION_ROWS", "32"))
    DATASET_CONCAT_SIZE_FAST = int(os.environ.get("GPTQMODEL_W4A_CALIBRATION_CONCAT_SIZE", "2048"))
    EVAL_BATCH_SIZE = 8
    OFFLOAD_TO_DISK = False
    ACT_GROUP_AWARE = False
    QUANT_BACKEND = BACKEND.GPTQ_W4A_NVFP4
    LOAD_BACKEND = BACKEND.GPTQ_W4A_NVFP4
    KERNEL_QUANT = {W4ANVFP4Linear}
    KERNEL_INFERENCE = {W4ANVFP4Linear}

    @classmethod
    def calibration_artifact(cls):
        value = os.environ.get("GPTQMODEL_W4A_CALIBRATION_ARTIFACT")
        if not value:
            raise ValueError("The 1B A4 test requires GPTQMODEL_W4A_CALIBRATION_ARTIFACT with audited fit data")
        return Path(value)

    @classmethod
    def calibration_token_limit(cls):
        value = int(os.environ.get("GPTQMODEL_W4A_CALIBRATION_MAX_TOKENS", "2048"))
        if value <= 0:
            raise ValueError("GPTQMODEL_W4A_CALIBRATION_MAX_TOKENS must be positive")
        return value

    @classmethod
    def load_dataset(cls, tokenizer=None, rows: int = 0):
        artifact = cls.calibration_artifact()
        partitions, manifest = load_calibration_artifact(artifact)
        token_limit = cls.calibration_token_limit()
        ids = _calibration_ids(tokenizer, artifact, rows, token_limit, partition="fit")
        cls._w4a_fit_provenance = {
            "artifact": str(artifact.resolve()), "manifest_sha256": manifest["manifest_sha256"],
            "corpus": manifest["corpus"], "partition": "fit", "rows": len(ids),
            "tokens_before_concatenation": sum(sample.numel() for sample in ids),
            "max_tokens_per_article": token_limit,
            "article_ids": [record["article_id"] for record in partitions["fit"][:rows]],
        }
        return [{"input_ids": sample.tolist(), "attention_mask": [1] * len(sample)} for sample in ids]

    def test_llama3_2_w4a_nvfp4(self):
        require_w4a_test_headroom(require_scope=True)
        self.calibration_token_limit()
        load_calibration_artifact(self.calibration_artifact())
        self.assertIsNone(self.GPTAQ)
        self.assertIsNone(self.FOEM)
        save_only = os.environ.get("GPTQMODEL_W4A_TEST_PHASE") == "quant-save"
        with self.model_compat_test_context():
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
            self.assertEqual(config["activation"]["version"], self.ACTIVATION_VERSION)
            self.assertEqual(config["quant_method"], "gptq")
            self.assertIsNone(config.get("meta", {}).get("gptaq"))
            self.assertIsNone(config.get("meta", {}).get("foem"))
            with safe_open(path / "model.safetensors", framework="pt") as saved:
                self.assertEqual(sum(key.endswith(".qweight") for key in saved.keys()), 112)
            with (path / "w4a_calibration_manifest.json").open("x") as handle:
                json.dump(self._w4a_fit_provenance, handle, indent=2)
                handle.write("\n")
            return
        self.check_kernel(model, self.KERNEL_INFERENCE)
        self.assertEqual(sum(isinstance(module, W4ANVFP4Linear)
                             for module in model.model.modules()), 112)
        layer = next(module for module in model.model.modules() if isinstance(module, W4ANVFP4Linear))
        x = torch.randn(1, layer.in_features, device=layer.qweight.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            recipe = model.quantize_config.activation_recipe
            output = layer(pack_activation(
                x, "w4a_nvfp4", recipe=recipe,
                global_scale=(layer.activation_global_scale
                              if recipe in {"nvidia_headroom", "least_squares_headroom"} else None),
            ))
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(output.shape, (1, layer.out_features))
        self.assertTrue(bool(torch.isfinite(output).all()))
