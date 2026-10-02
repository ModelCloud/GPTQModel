# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Experimental 1B W4A NVFP4 calibration that enables GPTAQ.

GPTAQ adds the native-versus-rounded input cross term to the GPTQ objective, so
the saved INT4 codes compensate for the E2M1 activation grid instead of being
solved against unrounded activations. Storage is unchanged: ordinary group-128
GPTQ INT4 `qweight` plus `qzeros`/`scales`/`g_idx`.

This module is an accuracy experiment for W4A4. The plain-GPTQ acceptance test
`test_llama3_2_w4a_nvfp4.py` is intentionally left untouched and still requires
`meta.gptaq is None`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from safetensors import safe_open

from gptqmodel.quantization.config import GPTAQConfig

from tests.models.test_llama3_2_w4a_nvfp4 import TestLlama3_2_W4ANVFP4
from tests.models.w4a_calibration_data import load_calibration_artifact
from w4a_gb10_memory import require_w4a_test_headroom


def gptaq_alpha() -> float:
    value = float(os.environ.get("GPTQMODEL_W4A_GPTAQ_ALPHA", "0.25"))
    if not 0.0 < value <= 1.0:
        raise ValueError("GPTQMODEL_W4A_GPTAQ_ALPHA must be in (0, 1].")
    return value


class TestLlama3_2_W4ANVFP4GPTAQ(TestLlama3_2_W4ANVFP4):
    """GPTAQ-calibrated variant; produces a checkpoint for downstream scoring."""

    GPTAQ = GPTAQConfig(alpha=gptaq_alpha())

    def test_llama3_2_w4a_nvfp4(self):
        require_w4a_test_headroom(require_scope=True)
        self.calibration_token_limit()
        load_calibration_artifact(self.calibration_artifact())
        self.assertIsNotNone(self.GPTAQ)
        if os.environ.get("GPTQMODEL_W4A_TEST_PHASE") != "quant-save":
            raise ValueError(
                "The GPTAQ experiment is checkpoint-only; set GPTQMODEL_W4A_TEST_PHASE=quant-save."
            )
        with self.model_compat_test_context():
            self.quantModel(
                self.NATIVE_MODEL_ID,
                batch_size=self.QUANT_BATCH_SIZE,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                need_eval=False,
                call_perform_post_quant_validation=False,
                reload_after_save=False,
            )
        path = Path(self.SAVE_PATH)
        self.assertTrue((path / "model.safetensors").is_file())
        config = json.loads((path / "quantize_config.json").read_text())
        self.assertEqual(config["quant_method"], "gptq")
        self.assertEqual(config["bits"], 4)
        self.assertEqual(config["activation"]["version"], self.ACTIVATION_VERSION)
        meta = config.get("meta", {})
        self.assertIsNotNone(meta.get("gptaq"), "Saved metadata must record the GPTAQ objective.")
        self.assertEqual(meta["gptaq"]["alpha"], self.GPTAQ.alpha)
        self.assertIsNone(meta.get("foem"))
        with safe_open(path / "model.safetensors", framework="pt") as saved:
            self.assertEqual(sum(key.endswith(".qweight") for key in saved.keys()), 112)
        with (path / "w4a_calibration_manifest.json").open("x") as handle:
            json.dump(
                {
                    **self._w4a_fit_provenance,
                    "algorithm": "gptaq",
                    "gptaq_alpha": self.GPTAQ.alpha,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
