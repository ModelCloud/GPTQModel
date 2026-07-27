# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression test that replays Laguna-S-2.1-GPTQ-FIXED's post-quant dynamic config.

The fixed model ships with 2,270 exact-literal dynamic patterns and ~36,432
quantized modules.  A naive resolver would call pcre.match ~82 million times
(2,270 patterns x 36,432 modules).  This test guards against that regression by
verifying the exact-literal fast path resolves every module in O(1) without any
PCRE matching.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pcre
import pytest

from gptqmodel.quantization.config import QuantizeConfig


def _laguna_model_path() -> Path:
    """Resolve the Laguna-S-2.1-GPTQ-FIXED model directory for this environment."""
    return Path(
        os.environ.get(
            "LAGUNA_S21_GPTQ_FIXED",
            "/monster/data/model/Laguna-S-2.1-GPTQ-FIXED",
        )
    )


def _load_laguna_quantize_config() -> dict:
    path = _laguna_model_path() / "quantize_config.json"
    if not path.exists():
        pytest.skip(f"Laguna quantize config not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_laguna_model_config() -> dict:
    path = _laguna_model_path() / "config.json"
    if not path.exists():
        pytest.skip(f"Laguna model config not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _generate_quantized_module_names(config: dict) -> list:
    """Reproduce the 36,432 quantized module names for Laguna-S-2.1-GPTQ-FIXED."""
    num_layers = config["num_hidden_layers"]
    num_experts = config["num_experts"]
    mlp_only_layers = set(config.get("mlp_only_layers", []))

    names = []
    attention_projs = ["q_proj", "k_proj", "v_proj", "g_proj"]
    mlp_projs = ["gate_proj", "up_proj", "down_proj"]

    for layer in range(num_layers):
        for proj in attention_projs:
            names.append(f"model.layers.{layer}.self_attn.{proj}")

        if layer in mlp_only_layers:
            for proj in mlp_projs:
                names.append(f"model.layers.{layer}.mlp.{proj}")
        else:
            for proj in mlp_projs:
                names.append(f"model.layers.{layer}.mlp.shared_experts.{proj}")
            for expert in range(num_experts):
                for proj in mlp_projs:
                    names.append(f"model.layers.{layer}.mlp.experts.{expert}.{proj}")

    return names


def test_laguna_dynamic_config_no_pcre_regression():
    """Replay Laguna's full module list and assert zero pcre.match calls."""
    qcfg_dict = _load_laguna_quantize_config()
    model_config = _load_laguna_model_config()

    qcfg = QuantizeConfig(**qcfg_dict)
    module_names = _generate_quantized_module_names(model_config)
    expected_count = 36432
    assert len(module_names) == expected_count, (
        f"Expected {expected_count} quantized modules, got {len(module_names)}"
    )

    with patch.object(pcre.Pattern, "match") as mock_match:
        start = time.perf_counter()
        for name in module_names:
            qcfg.dynamic_get(name, "bits", qcfg.bits)
        elapsed = time.perf_counter() - start

        assert mock_match.call_count == 0, (
            f"pcre.Pattern.match was called {mock_match.call_count} times for a "
            f"fully exact dynamic config; expected zero calls."
        )

    assert elapsed < 2.0, (
        f"Resolving {expected_count} module dynamic overrides took {elapsed:.2f}s; "
        f"expected the exact-literal fast path to finish in under 2s."
    )
