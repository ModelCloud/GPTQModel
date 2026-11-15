# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Legacy remote-code `generate()` contracts under transformers 5.

Issue #4121 item 3: #4108 fixed `generate()` only for ChatGLM. Every other
legacy family hits the same transformers-5 gaps, each from a different place:

* DeciLM / Baichuan2 index the cache as a tuple: `past_key_values[0][0]`
  (`'DynamicCache' object is not subscriptable`).
* MiniCPM-2B-dpo reads `past_key_values.seen_tokens` (attribute removed).
* EXAONE-3 reads `cache_position.shape[0]` while transformers 5 passes `None`.
* Nanbeige4.2 calls `get_max_length()` on a cache with no layers yet
  (`max() iterable argument is empty`).
* PanGu-alpha sizes the cache from `config.num_hidden_layers`, which its
  `GPTPanguConfig` does not declare (it only has `num_layers`).

The fix is one shared adapter, not one patch per family, so these tests pin the
adapter contract rather than a family name.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from gptqmodel.utils.hf import (
    _normalize_remote_code_config_compat,
    _patch_transformers_remote_code_compat,
    install_legacy_generation_compat,
)


OURO_PATH = Path("/monster/data/model/Ouro-1.4B")

_REMOTE_MODULE = "transformers_modules.gptqmodel_fixture.modeling_fixture"


def _remote_class(body: dict) -> type:
    """Build a class that looks like a `trust_remote_code` modeling module.

    The adapter only wraps methods that live in the remote modeling module, so
    the fixture methods carry that module name too.
    """

    cls = type("FixtureRemoteForCausalLM", (object,), {})
    cls.__module__ = _REMOTE_MODULE
    for name, member in body.items():
        if isinstance(member, types.FunctionType):
            member = types.FunctionType(
                member.__code__,
                member.__globals__,
                member.__name__,
                member.__defaults__,
                member.__closure__,
            )
            member.__module__ = _REMOTE_MODULE
        setattr(cls, name, member)
    return cls


def _legacy_prepare(self, input_ids, past_key_values=None, attention_mask=None,
                    cache_position=None, **kwargs):
    """The transformers-4 shape every family above copies."""

    self.calls.append(
        {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
        }
    )
    if past_key_values is not None:
        # DeciLM/Baichuan2: index the cache as a tuple of (key, value) pairs.
        past_length = past_key_values[0][0].shape[2]
        assert past_length == past_key_values.seen_tokens
        input_ids = input_ids[:, past_length:]
    if cache_position is None:
        cache_position = torch.arange(
            0 if past_key_values is None else past_key_values.seen_tokens,
            input_ids.shape[-1],
            device=input_ids.device,
        )
    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
    }


def _filled_cache(layers: int = 2, seq_len: int = 5) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx in range(layers):
        keys = torch.zeros(1, 2, seq_len, 4)
        values = torch.zeros(1, 2, seq_len, 4)
        cache.update(keys, values, layer_idx)
    return cache


def test_legacy_layer_count_alias_covers_num_layers_and_n_layer():
    for field in ("num_layers", "n_layer", "n_layers"):
        config = PretrainedConfig(model_type="fixture", **{field: 7})
        _normalize_remote_code_config_compat(config)
        assert config.num_hidden_layers == 7, field
        assert config.attribute_map["num_hidden_layers"] == field
        # The checkpoint's own field name survives serialization.
        serialized = config.to_dict()
        assert field in serialized
        assert "num_hidden_layers" not in serialized


def test_legacy_layer_count_alias_leaves_declared_config_alone():
    config = PretrainedConfig(
        model_type="fixture", num_hidden_layers=3, num_layers=9
    )
    _normalize_remote_code_config_compat(config)
    assert config.num_hidden_layers == 3
    assert "num_hidden_layers" not in config.attribute_map


def test_chatglm_alias_uses_the_shared_layer_count_helper():
    """#4108's ChatGLM aliases must survive the generalization to all configs."""

    class _FakeChatGlmConfig(PretrainedConfig):
        model_type = "chatglm"

    config = _FakeChatGlmConfig(seq_length=2048, num_layers=28)
    _normalize_remote_code_config_compat(config)
    assert config.max_length == 2048
    assert config.num_hidden_layers == 28
    # `max_length` is a compatibility view, not a generation parameter.
    assert "max_length" not in type(config)._get_generation_parameters(config)


def test_pangu_style_config_resolves_num_hidden_layers():
    """`GPTPanguConfig` declares `num_layers` only; DynamicCache needs this."""

    config = PretrainedConfig(model_type="gpt_pangu", num_layers=32)
    _normalize_remote_code_config_compat(config)
    assert config.num_hidden_layers == 32


def test_empty_cache_is_passed_as_none_and_cache_position_is_synthesized():
    cls = _remote_class({"prepare_inputs_for_generation": _legacy_prepare})
    assert install_legacy_generation_compat(cls()) is True

    model = cls()
    model.calls = []
    out = model.prepare_inputs_for_generation(
        torch.tensor([[1, 2, 3, 4]]), past_key_values=DynamicCache()
    )
    call = model.calls[0]
    assert call["past_key_values"] is None
    assert tuple(out["input_ids"].shape) == (1, 4)
    assert out["cache_position"].tolist() == [0, 1, 2, 3]


def test_filled_cache_keeps_tuple_indexing_and_seen_tokens():
    cls = _remote_class({"prepare_inputs_for_generation": _legacy_prepare})
    install_legacy_generation_compat(cls())

    model = cls()
    model.calls = []
    out = model.prepare_inputs_for_generation(
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), past_key_values=_filled_cache(seq_len=5)
    )
    call = model.calls[0]
    assert call["past_key_values"] is not None
    # Only the unprocessed tokens survive the slice.
    assert tuple(out["input_ids"].shape) == (1, 2)
    assert out["cache_position"].tolist() == [5, 6]


def test_install_is_idempotent_for_repeated_loads():
    cls = _remote_class({"prepare_inputs_for_generation": _legacy_prepare})
    assert install_legacy_generation_compat(cls()) is True
    wrapped = cls.prepare_inputs_for_generation
    assert install_legacy_generation_compat(cls()) is False
    assert cls.prepare_inputs_for_generation is wrapped


def test_native_transformers_class_is_not_wrapped():
    class Native:
        def prepare_inputs_for_generation(self, input_ids, next_sequence_length=None,
                                          past_key_values=None, **kwargs):
            return {"input_ids": input_ids}

    assert install_legacy_generation_compat(Native()) is False


def test_dynamic_cache_legacy_surface():
    _patch_transformers_remote_code_compat()
    cache = _filled_cache(layers=3, seq_len=5)

    # `past_key_values[0][0].shape[2]` (DeciLM / Baichuan2).
    assert cache[0][0].shape[2] == 5
    assert cache[-1][1].shape[2] == 5
    assert len(cache[:2]) == 2

    # `past_key_values.seen_tokens` (MiniCPM-2B-dpo).
    assert cache.seen_tokens == 5

    # `get_usable_length` reports the pre-5.x "unbounded" result instead of
    # `max_length - new_seq_length` with the -1 sentinel.
    assert cache.get_usable_length(1, 0) == 5


def test_empty_cache_tuple_indexing_reports_missing_state():
    _patch_transformers_remote_code_compat()
    with pytest.raises(IndexError):
        DynamicCache()[0]


@pytest.mark.skipif(
    not OURO_PATH.is_dir(), reason=f"local Ouro fixture is unavailable: {OURO_PATH}"
)
def test_native_load_runs_before_model_load_hook(tmp_path):
    """Issue #4121 item 2: the native branch must call `before_model_load`.

    Ouro's remote forward forwards `input_embeds`/`cache_position` to
    `create_causal_mask`; only the definition's `before_model_load` adapts
    those names. A native `GPTQModel.load(quantize_config=None)` raised
    `TypeError: create_causal_mask() got an unexpected keyword argument
    'input_embeds'` before the hook was wired into that branch.
    """

    from gptqmodel import GPTQModel
    from tests.tiny_remote_models import build_tiny_remote_model

    path = build_tiny_remote_model(
        OURO_PATH,
        tmp_path / "tiny_ouro",
        {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "total_ut_steps": 4,
            "layer_types": ["full_attention", "full_attention"],
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
    )

    model = GPTQModel.load(
        str(path),
        quantize_config=None,
        trust_remote_code=True,
        device="cpu",
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    out = model.generate(input_ids, max_new_tokens=3, do_sample=False, num_beams=1)
    assert out.shape[0] == 1
    assert out.shape[1] > input_ids.shape[1]
