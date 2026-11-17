# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4453: a remote ``auto_map`` class name is not a layout signal.

Remote and native classes for the same architecture are routinely spelled
differently (``PhiMoEForCausalLM`` vs ``PhimoeForCausalLM``, ``MPTForCausalLM``
vs ``MptForCausalLM``, ``MiniMaxM2ForCausalLM`` vs ``MiniMaxForCausalLM``).
Treating any spelling difference as a layout override made an explicit
``trust_remote_code=True`` run stale remote code for natively supported
checkpoints. Ling's ``q_lora_rank`` MLA marker still identifies a distinct
layout, now handled by the definition's local port regardless of trust or
security bypass. Fixtures are local ``config.json`` files only; no remote
module is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gptqmodel.utils import hf as hf_utils
from gptqmodel.utils.hf import resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


pytestmark = [pytest.mark.cpu]


def _write_config(directory: Path, payload: dict) -> str:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(directory)


@pytest.fixture(autouse=True)
def _clear_layout_caches():
    def clear():
        hf_utils._checkpoint_config_payload.cache_clear()
        hf_utils._checkpoint_remote_layout_marker.cache_clear()
        hf_utils._TRUST_REMOTE_CODE_OVERRIDE_WARNED.clear()
        hf_utils._TRUST_REMOTE_CODE_REMOTE_LAYOUT_LOGGED.clear()

    clear()
    yield
    clear()


# Synthetic configs mirroring the local checkpoints that flipped to remote code
# at 489aed6: same native `model_type`, remote class spelled differently.
_NATIVE_LAYOUT_REMOTE_CLASSES = [
    pytest.param(
        "phimoe",
        "modeling_slimmoe.PhiMoEForCausalLM",
        "PhimoeForCausalLM",
        id="phi-moe-case-only",
    ),
    pytest.param(
        "mpt",
        "modeling_mpt.MPTForCausalLM",
        "MptForCausalLM",
        id="mpt-case-only",
    ),
    pytest.param(
        "minimax",
        "modeling_minimax_m2.MiniMaxM2ForCausalLM",
        "MiniMaxForCausalLM",
        id="minimax-m2-vs-native-name",
    ),
    pytest.param(
        "bailing_hybrid",
        "modeling_bailing_moe_v2_5.BailingMoeV2_5ForCausalLM",
        "BailingHybridForCausalLM",
        id="bailing-hybrid-dense-q",
    ),
]


@pytest.mark.parametrize(("model_type", "remote_ref", "native_name"), _NATIVE_LAYOUT_REMOTE_CLASSES)
def test_remote_class_name_difference_keeps_native_layout(tmp_path, model_type, remote_ref, native_name):
    checkpoint = _write_config(
        tmp_path / model_type,
        {
            "model_type": model_type,
            "architectures": [remote_ref.rsplit(".", 1)[-1]],
            "auto_map": {
                "AutoConfig": f"configuration_{model_type}.RemoteConfig",
                "AutoModelForCausalLM": remote_ref,
            },
        },
    )

    native_supported, _, native_cls_name = hf_utils._detect_native_transformers_causallm_support(checkpoint)
    assert native_supported
    assert native_cls_name == native_name
    assert native_cls_name != remote_ref.rsplit(".", 1)[-1]

    assert hf_utils._checkpoint_remote_layout_marker(checkpoint) == (None, remote_ref.rsplit(".", 1)[-1])
    assert resolve_trust_remote_code(checkpoint, trust_remote_code=True) is False


def test_ling_mla_marker_uses_package_code_with_or_without_bypass(tmp_path, monkeypatch):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    checkpoint = _write_config(
        tmp_path / "ling_flash",
        {
            "model_type": "bailing_hybrid",
            "architectures": ["BailingMoeV2_5ForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_bailing_moe_v2_5.BailingMoeV2_5Config",
                "AutoModelForCausalLM": "modeling_bailing_moe_v2_5.BailingMoeV2_5ForCausalLM",
            },
            "q_lora_rank": 1536,
        },
    )

    assert hf_utils._checkpoint_remote_layout_marker(checkpoint) == ("q_lora_rank", "BailingMoeV2_5ForCausalLM")
    assert resolve_trust_remote_code(checkpoint, trust_remote_code=True) is False
    monkeypatch.setenv(BYPASS_ENV, "1")
    assert resolve_trust_remote_code(checkpoint, trust_remote_code=True) is False
    assert resolve_trust_remote_code(checkpoint, trust_remote_code=False) is False


@pytest.mark.parametrize("q_lora_rank", [None, 0])
def test_ling_unset_mla_marker_is_native_layout(tmp_path, q_lora_rank):
    checkpoint = _write_config(
        tmp_path / "ling_dense_q",
        {
            "model_type": "bailing_hybrid",
            "auto_map": {
                "AutoModelForCausalLM": "modeling_bailing_moe_v2_5.BailingMoeV2_5ForCausalLM",
            },
            "q_lora_rank": q_lora_rank,
        },
    )

    assert resolve_trust_remote_code(checkpoint, trust_remote_code=True) is False


def test_layout_marker_without_remote_model_code_selects_local_port(tmp_path, monkeypatch):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    checkpoint = _write_config(
        tmp_path / "ling_no_auto_map",
        {"model_type": "bailing_hybrid", "q_lora_rank": 1536},
    )

    assert hf_utils._checkpoint_remote_layout_marker(checkpoint) == (None, None)
    assert resolve_trust_remote_code(checkpoint, trust_remote_code=True) is False


def test_layout_marker_is_scoped_to_its_model_type(tmp_path):
    # `q_lora_rank` is native layout for DeepSeek-V3; only `bailing_hybrid`
    # declares it as a remote-only marker.
    checkpoint = _write_config(
        tmp_path / "deepseek_v3",
        {
            "model_type": "deepseek_v3",
            "auto_map": {
                "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
            },
            "q_lora_rank": 1536,
        },
    )

    assert resolve_trust_remote_code(checkpoint, trust_remote_code=True) is False
