# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for checkpoint-owned ``generation_config.do_sample``.

GPT-QModel used to force ``do_sample=True`` on every loaded model, so
``generate()`` sampled by default and quantized exports wrote
``"do_sample": true`` even when the source checkpoint declared greedy decoding.
These tests are CPU-only and never download a model.
"""

import json

import torch
from transformers import GenerationConfig
from transformers.generation import GenerationMode

from gptqmodel.models.writer import _save_model_configs_without_weights
from gptqmodel.utils import hf as hf_utils


def _write_generation_config(directory, payload):
    path = directory / "generation_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _tiny_causal_lm():
    """Build a tiny GPT-2 on CPU so ``generate()`` needs no download."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    return GPT2LMHeadModel(config).eval()


class _ConfigOnlyModel:
    class _Config:
        def save_pretrained(self, save_dir):
            from pathlib import Path

            (Path(save_dir) / "config.json").write_text(
                json.dumps({"architectures": ["ConfigOnlyModel"]}),
                encoding="utf-8",
            )

    config = _Config()
    dtype = torch.float32

    def __init__(self, generation_config):
        self.generation_config = generation_config

    @staticmethod
    def can_generate():
        return True


def test_absent_do_sample_stays_unset_and_generates_greedily(tmp_path, monkeypatch):
    _write_generation_config(
        tmp_path,
        {"max_new_tokens": 3, "pad_token_id": 0, "eos_token_id": 1},
    )

    cfg = hf_utils._load_sanitized_generation_config(str(tmp_path))

    assert cfg is not None
    assert cfg.do_sample is not True
    assert cfg.get_generation_mode() is GenerationMode.GREEDY_SEARCH

    model = _tiny_causal_lm()
    model.generation_config = cfg

    def _fail_if_sampled(*_args, **_kwargs):
        raise AssertionError("generate() sampled even though do_sample was absent")

    monkeypatch.setattr(torch, "multinomial", _fail_if_sampled)
    model.generate(torch.tensor([[2, 3, 4]]), max_new_tokens=3)


def test_explicit_do_sample_true_is_preserved(tmp_path):
    _write_generation_config(
        tmp_path,
        {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
    )

    cfg = hf_utils._load_sanitized_generation_config(str(tmp_path))

    assert cfg.do_sample is True
    assert cfg.get_generation_mode() is GenerationMode.SAMPLE

    hf_utils.autofix_hf_generation_config(cfg)
    assert cfg.do_sample is True


def test_explicit_do_sample_false_is_preserved_without_sampling_fields(tmp_path):
    _write_generation_config(
        tmp_path,
        {"do_sample": False, "max_new_tokens": 3},
    )

    cfg = hf_utils._load_sanitized_generation_config(str(tmp_path))

    assert cfg.do_sample is False


def test_explicit_greedy_with_sampling_fields_is_repaired(tmp_path):
    """The old dead block's intent: only repair a self-contradicting config."""
    _write_generation_config(
        tmp_path,
        {"do_sample": False, "temperature": 0.5},
    )

    cfg = hf_utils._load_sanitized_generation_config(str(tmp_path))

    assert cfg.do_sample is True


def test_checkpoint_write_does_not_add_do_sample(tmp_path):
    model = _ConfigOnlyModel(GenerationConfig(max_new_tokens=5))

    _save_model_configs_without_weights(model, str(tmp_path))

    path = tmp_path / "generation_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("do_sample") is not True
    assert "do_sample" not in payload

    # The writer's post-save sanitizer must not add it either.
    assert hf_utils.sanitize_generation_config_file(str(path)) is False
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "do_sample" not in payload


def test_checkpoint_write_preserves_absent_do_sample_with_sampling_fields(tmp_path):
    """An absent ``do_sample`` must not be rewritten just to satisfy HF."""
    model = _ConfigOnlyModel(GenerationConfig(max_new_tokens=5, temperature=0.7))

    _save_model_configs_without_weights(model, str(tmp_path))

    payload = json.loads((tmp_path / "generation_config.json").read_text(encoding="utf-8"))
    assert "do_sample" not in payload
    assert payload["temperature"] == 0.7


def test_loaded_absent_do_sample_roundtrips_without_do_sample(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_generation_config(
        source,
        {
            "_from_model_config": True,
            "max_new_tokens": 7,
            "pad_token_id": 0,
            "eos_token_id": 1,
        },
    )

    cfg = hf_utils._load_sanitized_generation_config(str(source))
    assert cfg.do_sample is not True

    output = tmp_path / "output"
    cfg.save_pretrained(output)
    payload = json.loads((output / "generation_config.json").read_text(encoding="utf-8"))
    assert "do_sample" not in payload


def test_saved_explicit_greedy_with_sampling_fields_is_repaired(tmp_path):
    path = _write_generation_config(
        tmp_path,
        {"do_sample": False, "temperature": 0.5},
    )

    assert hf_utils.sanitize_generation_config_file(str(path)) is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["do_sample"] is True
