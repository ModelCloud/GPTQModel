# GPU=-1
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Gemma 2 long-context guard: uninitialized backend and rejection paths.

The 27B-sized guard (more than 42 decoder layers) only admits quantized
vLLM/SGLang engines.  A plain Transformers model must be rejected with the
documented ``SUPPORT_ERR`` message instead of leaking ``UnboundLocalError``.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

from gptqmodel.models.definitions.gemma2 import SUPPORT_ERR, Gemma2QModel
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils import sglang as sglang_utils
from gptqmodel.utils import vllm as vllm_utils


def _gemma2_config(num_hidden_layers: int) -> Gemma2Config:
    return Gemma2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )


def _meta_model(num_hidden_layers: int):
    with torch.device("meta"):
        return Gemma2ForCausalLM(_gemma2_config(num_hidden_layers))


def _quantize_config() -> QuantizeConfig:
    return QuantizeConfig(bits=4, group_size=128)


class _FakeEngine:
    def __init__(self, config):
        self.config = config


class _FakeVLLM:
    """Imports nothing from the real vLLM package."""

    def __init__(self):
        self.LLM = _FakeEngine


@pytest.mark.parametrize("num_hidden_layers", [43, 46])
def test_plain_transformers_gemma2_guard_raises_support_error(num_hidden_layers):
    with pytest.raises(ValueError) as excinfo:
        Gemma2QModel(_meta_model(num_hidden_layers), quantized=True, quantize_config=_quantize_config())

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == SUPPORT_ERR


def test_unquantized_long_gemma2_still_only_warns():
    wrapper = Gemma2QModel(_meta_model(43), quantized=False, quantize_config=_quantize_config())

    assert wrapper.quantized is False
    assert wrapper.model.config.num_hidden_layers == 43


def test_quantized_vllm_engine_still_admitted(monkeypatch):
    fake = _FakeVLLM()
    monkeypatch.setattr(vllm_utils, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(vllm_utils, "LLM", fake.LLM)

    engine = fake.LLM(_gemma2_config(43))
    wrapper = Gemma2QModel(engine, quantized=True, quantize_config=_quantize_config())

    assert wrapper.model is engine


def test_quantized_sglang_engine_still_admitted(monkeypatch):
    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(sglang_utils, "sgl", SimpleNamespace(Runtime=type("_FakeRuntime", (), {})))

    engine = SimpleNamespace(config=_gemma2_config(43))
    # The loader marks engine-API runtimes with this attribute.
    setattr(engine, sglang_utils._ENGINE_MARKER, True)
    wrapper = Gemma2QModel(engine, quantized=True, quantize_config=_quantize_config())

    assert wrapper.model is engine


def test_vllm_available_but_transformers_model_is_rejected(monkeypatch):
    fake = _FakeVLLM()
    monkeypatch.setattr(vllm_utils, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(vllm_utils, "LLM", fake.LLM)

    with pytest.raises(ValueError) as excinfo:
        Gemma2QModel(_meta_model(43), quantized=True, quantize_config=_quantize_config())

    assert str(excinfo.value) == SUPPORT_ERR


def test_backend_detection_helpers_ignore_unavailable_engines(monkeypatch):
    monkeypatch.setattr(vllm_utils, "VLLM_AVAILABLE", False)
    monkeypatch.setattr(vllm_utils, "LLM", None)
    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", False)

    assert vllm_utils.is_vllm_model(object()) is False
    assert sglang_utils.is_sglang_model(object()) is False
