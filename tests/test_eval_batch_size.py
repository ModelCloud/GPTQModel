# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import types

import pytest

from tests.eval import evaluate, resolve_eval_batch_size


pytestmark = [pytest.mark.cpu]

_CONFIG = {
    "vocab_size": 1000,
    "hidden_size": 100,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 25,
}

_QWEN2_5_0_5B_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
}


def _resolve(
        tmp_path,
        total_memory_bytes,
        *,
        free_memory_bytes=None,
        dtype="bfloat16",
        max_new_tokens=0,
):
    return resolve_eval_batch_size(
        "auto",
        model_or_id_or_path=str(tmp_path),
        model_args={"dtype": dtype},
        suite_kwargs={"max_new_tokens": max_new_tokens},
        device="cpu",
        total_memory_bytes=total_memory_bytes,
        free_memory_bytes=free_memory_bytes,
    )


def test_explicit_batch_sizes_are_unchanged():
    assert resolve_eval_batch_size(64) == 64
    assert resolve_eval_batch_size("8") == "8"
    assert resolve_eval_batch_size(1) == 1


def test_auto_without_config_returns_known_safe_size(tmp_path):
    batch = resolve_eval_batch_size(
        "AUTO",
        model_or_id_or_path=str(tmp_path),
        total_memory_bytes=1024**3,
        free_memory_bytes=1024**3,
    )
    assert batch == 8


def test_auto_uses_free_memory_and_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    assert _resolve(tmp_path, 64 * 1024**2) == 3
    assert _resolve(tmp_path, 256 * 1024**2) == 13
    assert _resolve(tmp_path, 1024**3) == 16


def test_auto_never_exceeds_the_conservative_cap(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    assert _resolve(tmp_path, 1024**4) == 16


def test_auto_caps_at_free_memory(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    assert _resolve(tmp_path, 1024**3, free_memory_bytes=64 * 1024**2) == 3


def test_qwen2_5_0_5b_on_32gib_picks_known_safe_batch(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_QWEN2_5_0_5B_CONFIG))
    assert _resolve(tmp_path, 32 * 1024**3, max_new_tokens=256) == 8


def test_float32_is_more_conservative_than_bfloat16(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    bf16 = _resolve(tmp_path, 256 * 1024**2, dtype="bfloat16")
    fp32 = _resolve(tmp_path, 256 * 1024**2, dtype="float32")
    assert fp32 < bf16


def test_live_model_config_is_used():
    model = types.SimpleNamespace(config=_CONFIG)
    batch = resolve_eval_batch_size(
        "auto",
        model_or_id_or_path=model,
        suite_kwargs={"max_new_tokens": 0},
        device="cpu",
        total_memory_bytes=256 * 1024**2,
        free_memory_bytes=256 * 1024**2,
    )
    assert batch == 13


def test_missing_memory_falls_back_to_known_safe_size(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    assert resolve_eval_batch_size(
        "auto",
        model_or_id_or_path=str(tmp_path),
        device="cpu",
    ) == 8


def test_evaluate_resolves_auto_before_evalution(monkeypatch, tmp_path):
    captured = {}

    def fake_run_evalution(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("tests.eval.run_evalution", fake_run_evalution)
    evaluate(
        model_or_id_or_path=str(tmp_path),
        tasks=["arc_challenge"],
        batch_size="auto",
        model_args={"device": "cpu"},
        suite_kwargs={"max_new_tokens": 0},
    )
    assert captured["batch_size"] == 8


def test_evaluate_preserves_explicit_batch_size(monkeypatch, tmp_path):
    captured = {}

    def fake_run_evalution(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("tests.eval.run_evalution", fake_run_evalution)
    evaluate(
        model_or_id_or_path=str(tmp_path),
        tasks=["arc_challenge"],
        batch_size=6,
        model_args={"device": "cpu"},
    )
    assert captured["batch_size"] == 6
