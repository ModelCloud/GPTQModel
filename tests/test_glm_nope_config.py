# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("explicit_nope_flag", [False, True])
def test_glm_nope_roundtrip(tmp_path: Path, explicit_nope_flag: bool) -> None:
    module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    text_config = {"qk_rope_head_dim": 0}
    if explicit_nope_flag:
        text_config["mla_use_nope"] = True
    config = module.Glm5NextConfig(text_config=text_config)
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    restored = module.Glm5NextConfig.from_pretrained(tmp_path)
    assert saved.get("rope_parameters") is None
    assert saved["text_config"].get("rope_parameters") is None
    assert getattr(restored, "rope_parameters", None) is None
    assert getattr(restored.text_config, "rope_parameters", None) is None


