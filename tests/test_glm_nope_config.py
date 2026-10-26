# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gptqmodel.utils.hf import normalize_hf_config_compat


@pytest.mark.parametrize("entry", ["root", "text", "both"])
@pytest.mark.parametrize("explicit_nope_flag", [False, True])
def test_glm_nope_roundtrip(tmp_path: Path, entry: str, explicit_nope_flag: bool) -> None:
    module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    text_config = {"qk_rope_head_dim": 0}
    if explicit_nope_flag:
        text_config["mla_use_nope"] = True
    config = module.Glm5NextConfig(text_config=text_config)
    if entry in {"root", "both"}:
        normalize_hf_config_compat(config)
    if entry in {"text", "both"}:
        normalize_hf_config_compat(config.text_config)
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    restored = module.Glm5NextConfig.from_pretrained(tmp_path)
    assert saved.get("rope_parameters") is None
    assert saved["text_config"].get("rope_parameters") is None
    assert getattr(restored, "rope_parameters", None) is None
    assert getattr(restored.text_config, "rope_parameters", None) is None


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text"])
def test_glm_nope_without_flag_keeps_rotary_config_absent(model_type: str) -> None:
    text = SimpleNamespace(model_type="glm5_next_text", qk_rope_head_dim=0)
    config = SimpleNamespace(model_type=model_type, text_config=text) if model_type == "glm5_next" else text

    normalize_hf_config_compat(config)

    assert getattr(config, "rope_parameters", None) is None


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text"])
@pytest.mark.parametrize(
    "rope_field, value",
    [
        ("rope_parameters", {"rope_type": "default", "rope_theta": 12345.0}),
        ("rope_parameters", {"rope_theta": 12345.0}),
        ("rope_scaling", {"type": "linear", "factor": 2.0}),
        ("rope_theta", 12345.0),
    ],
)
def test_explicit_glm_rotary_config_is_preserved(
    model_type: str,
    rope_field: str,
    value: object,
) -> None:
    text = SimpleNamespace(
        model_type="glm5_next_text",
        mla_use_nope=True,
        qk_rope_head_dim=0,
    )
    config = SimpleNamespace(model_type=model_type, text_config=text) if model_type == "glm5_next" else text
    setattr(config, rope_field, value)
    normalize_hf_config_compat(config)
    if isinstance(value, dict):
        for key, item in value.items():
            assert config.rope_parameters[key] == item
    else:
        assert config.rope_parameters["rope_theta"] == value


@pytest.mark.parametrize(
    "model_type, nope, dim",
    [
        ("other", True, 0),
        ("glm5_next_text", False, 0),
        ("glm5_next_text", True, 64),
    ],
)
def test_other_configs_keep_rotary_defaults(
    model_type: str,
    nope: bool,
    dim: int,
) -> None:
    config = SimpleNamespace(
        model_type=model_type,
        mla_use_nope=nope,
        qk_rope_head_dim=dim,
    )
    normalize_hf_config_compat(config)
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 10000.0}
