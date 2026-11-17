# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""The estimate has to be reachable from a loaded model, not just from a log line.

Most users load a quantized checkpoint rather than quantize one, and the only
place the KV-cache cost appeared was the end of a quantization run. These tests
cover the public method, the dtype it inherits from the weights, and the line
``load()`` now emits.
"""

import types

import torch

from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.kv_cache import KVCacheEstimate


def _stub(*, layers=2, heads=4, kv_heads=2, hidden=64, dtype=torch.bfloat16):
    config = types.SimpleNamespace(
        hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads,
    )
    parameters = [torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=dtype)]
    inner = types.SimpleNamespace(config=config, parameters=lambda: iter(parameters))
    return types.SimpleNamespace(model=inner)


def test_estimate_inherits_the_weight_dtype():
    stub = _stub(dtype=torch.bfloat16)
    estimate = BaseQModel.kv_cache_estimate(stub)
    assert isinstance(estimate, KVCacheEstimate)
    assert estimate.dtype == torch.bfloat16
    # 2 layers * 2 (K and V) * 2 kv_heads * 16 head_dim * 2 bytes
    assert estimate.bytes_per_token == 2 * 2 * 2 * 16 * 2


def test_explicit_dtype_overrides_the_weights():
    stub = _stub(dtype=torch.bfloat16)
    estimate = BaseQModel.kv_cache_estimate(stub, dtype=torch.float32)
    assert estimate.dtype == torch.float32
    assert estimate.bytes_per_token == 2 * 2 * 2 * 16 * 4


def test_non_floating_parameters_are_skipped():
    """Quantized checkpoints hold integer tensors; the cache follows activations."""

    config = types.SimpleNamespace(hidden_size=64, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2)
    inner = types.SimpleNamespace(
        config=config,
        parameters=lambda: iter([torch.zeros(1, dtype=torch.int64)]),
    )
    estimate = BaseQModel.kv_cache_estimate(types.SimpleNamespace(model=inner))
    # No floating parameter at all: fall back to the config dtype (unset -> fp16).
    assert estimate.dtype == torch.float16


def test_model_without_a_decoder_config_returns_none():
    inner = types.SimpleNamespace(config=types.SimpleNamespace(), parameters=lambda: iter([]))
    assert BaseQModel.kv_cache_estimate(types.SimpleNamespace(model=inner)) is None


def test_logging_is_skipped_when_the_config_cannot_be_resolved(capsys):
    inner = types.SimpleNamespace(config=types.SimpleNamespace(), parameters=lambda: iter([]))
    BaseQModel.log_kv_cache_estimate(types.SimpleNamespace(model=inner))
    assert "KV cache:" not in capsys.readouterr().out


def test_load_emits_the_estimate(capsys, tmp_path):
    """The documented entry point must tell a loader what context costs."""

    from pathlib import Path

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from gptqmodel import GPTQModel

    source = Path(tmp_path) / "native"
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64)
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(source)
    tokenizer = Tokenizer(WordLevel(vocab={"<pad>": 0, "<s>": 1, "eos": 2}, unk_token="<pad>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
                            bos_token="<s>", eos_token="eos").save_pretrained(source)

    model = GPTQModel.load(str(source), device="cpu")
    captured = capsys.readouterr()
    assert "KV cache:" in captured.out + captured.err

    estimate = model.kv_cache_estimate()
    assert estimate is not None
    assert estimate.growing_layers == config.num_hidden_layers


def test_kv_cache_plan_takes_an_explicit_budget():
    stub = _stub()
    budget, rows, estimate = BaseQModel.kv_cache_plan(stub, vram_gib=1)
    assert budget == 1024 ** 3
    assert rows and estimate.context_limit is None
    assert all(context > 0 for _, context in rows)


def test_kv_cache_plan_uses_free_vram_when_no_budget_is_given(monkeypatch):
    import gptqmodel.models.base as base

    monkeypatch.setattr(base, "_free_vram_budget", lambda reserve=0.1: 2 * 1024 ** 3)
    budget, rows, _estimate = BaseQModel.kv_cache_plan(_stub())
    assert budget == 2 * 1024 ** 3
    assert rows


def test_kv_cache_plan_returns_none_without_a_budget(monkeypatch):
    import gptqmodel.models.base as base

    monkeypatch.setattr(base, "_free_vram_budget", lambda reserve=0.1: None)
    assert BaseQModel.kv_cache_plan(_stub()) is None


def test_free_vram_budget_applies_the_reserve(monkeypatch):
    import gptqmodel.models.base as base

    monkeypatch.setattr(base.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base.torch.cuda, "mem_get_info", lambda: (10 * 1024 ** 3, 24 * 1024 ** 3))
    assert base._free_vram_budget(0.1) == int(9 * 1024 ** 3)
    assert base._free_vram_budget(0.0) == 10 * 1024 ** 3


def test_logging_adds_the_plan_when_a_budget_is_known(monkeypatch, capsys):
    import gptqmodel.models.base as base

    monkeypatch.setattr(base, "_free_vram_budget", lambda reserve=0.1: 1024 ** 3)
    BaseQModel.log_kv_cache_estimate(_stub())
    out = capsys.readouterr().out
    assert "KV cache:" in out
    assert "KV plan:" in out


def test_logging_skips_the_plan_when_asked(monkeypatch, capsys):
    import gptqmodel.models.base as base

    monkeypatch.setattr(base, "_free_vram_budget", lambda reserve=0.1: 1024 ** 3)
    BaseQModel.log_kv_cache_estimate(_stub(), plan=False)
    out = capsys.readouterr().out
    assert "KV cache:" in out
    assert "KV plan:" not in out
