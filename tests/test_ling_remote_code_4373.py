# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4373: `bailing_hybrid` remote-code layout must not be force-nativized.

Ling-2.6-flash declares ``model_type=bailing_hybrid`` but ships its own
``auto_map`` model class and the remote MLA attention layout. The installed
transformers ``BailingHybridForCausalLM`` is a different layout, so forcing
``trust_remote_code=False`` from ``model_type`` alone crashes in
``_init_weights``. These fixtures use local ``config.json`` files only; no
checkpoint or remote module is downloaded or imported.
"""

from __future__ import annotations

import json
from pathlib import Path

from gptqmodel.utils import hf as hf_utils
from gptqmodel.utils.hf import resolve_trust_remote_code


def _write_config(directory: Path, payload: dict) -> Path:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _clear_layout_caches() -> None:
    for name in (
        "_detect_native_transformers_causallm_support",
        "_checkpoint_config_payload",
        "_checkpoint_remote_layout_overrides_native",
    ):
        cached = getattr(hf_utils, name, None)
        if cached is not None and hasattr(cached, "cache_clear"):
            cached.cache_clear()
    hf_utils._TRUST_REMOTE_CODE_OVERRIDE_WARNED.clear()
    getattr(hf_utils, "_TRUST_REMOTE_CODE_REMOTE_LAYOUT_LOGGED", set()).clear()


def test_native_model_type_without_auto_map_stays_native(tmp_path):
    _clear_layout_caches()
    checkpoint = _write_config(
        tmp_path / "llama",
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 8,
        },
    )

    assert resolve_trust_remote_code(str(checkpoint), trust_remote_code=True) is False


def test_native_model_type_with_native_auto_map_stays_native(tmp_path):
    _clear_layout_caches()
    checkpoint = _write_config(
        tmp_path / "llama_auto_map",
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "auto_map": {
                "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM",
            },
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 8,
        },
    )

    assert resolve_trust_remote_code(str(checkpoint), trust_remote_code=True) is False


def test_auto_map_config_only_does_not_select_remote_model_code(tmp_path):
    _clear_layout_caches()
    checkpoint = _write_config(
        tmp_path / "llama_config_only",
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_llama.LlamaConfig",
            },
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 8,
        },
    )

    assert resolve_trust_remote_code(str(checkpoint), trust_remote_code=True) is False


def test_bailing_hybrid_remote_code_layout_respects_explicit_trust(tmp_path):
    _clear_layout_caches()
    checkpoint = _write_config(
        tmp_path / "ling_flash",
        {
            "model_type": "bailing_hybrid",
            "architectures": ["BailingMoeV2_5ForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_bailing_moe_v2_5.BailingMoeV2_5Config",
                "AutoModelForCausalLM": "modeling_bailing_moe_v2_5.BailingMoeV2_5ForCausalLM",
            },
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 8,
        },
    )

    assert resolve_trust_remote_code(str(checkpoint), trust_remote_code=True) is True


def test_bailing_hybrid_remote_code_layout_stays_off_without_trust(tmp_path):
    _clear_layout_caches()
    checkpoint = _write_config(
        tmp_path / "ling_flash_untrusted",
        {
            "model_type": "bailing_hybrid",
            "architectures": ["BailingMoeV2_5ForCausalLM"],
            "auto_map": {
                "AutoModelForCausalLM": "modeling_bailing_moe_v2_5.BailingMoeV2_5ForCausalLM",
            },
            "hidden_size": 8,
            # No `q_lora_rank`: the remote class name is the layout signal here.
            "vocab_size": 8,
        },
    )

    assert resolve_trust_remote_code(str(checkpoint), trust_remote_code=False) is False


def test_bailing_hybrid_mla_marker_with_same_named_remote_class(tmp_path):
    """A same-named remote class still declares the MLA layout via config."""

    _clear_layout_caches()
    checkpoint = _write_config(
        tmp_path / "ling_flash_same_name",
        {
            "model_type": "bailing_hybrid",
            "architectures": ["BailingHybridForCausalLM"],
            "auto_map": {
                "AutoModelForCausalLM": "modeling_bailing_hybrid.BailingHybridForCausalLM",
            },
            "hidden_size": 8,
            # The native config has no `q_lora_rank`; its presence means the
            # checkpoint owns the remote MLA branch.
            "q_lora_rank": 8,
            "vocab_size": 8,
        },
    )

    assert resolve_trust_remote_code(str(checkpoint), trust_remote_code=True) is True
