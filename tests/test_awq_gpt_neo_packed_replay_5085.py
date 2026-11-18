# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression for #5085: packed AWQ replay must respect the position limit.

AWQ samples the rows of several calibration documents into one
``[1, retained, hidden]`` replay sequence (``_token_row_plan_for_module`` at
capture time since #5175, ``_pack_token_rows`` on the legacy fallback), so with
``batch_size > 1`` the packed length is about ``batch_size * seq_len``. A
block-diagonal mask and the retained rows' positions keep that semantically
correct for mask-driven attention, but ``GPTNeoSelfAttention`` does not read
its causal structure from ``attention_mask``. It slices a fixed
``[max_position_embeddings, max_position_embeddings]`` causal buffer:

    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

When the packed replay is longer than ``max_position_embeddings`` the slice is
clipped and ``torch.where`` raises a plain ``RuntimeError`` that the configured
scale-group fallback could not catch, so ``quantize()`` aborted in layer 0.

These nodes pin both halves of the fix on a tiny native CPU GPT-Neo:

* the token-row budget is capped at the model's declared sequence limit, and
* a replay that still raises a tensor-geometry error becomes an
  ``_AWQScaleGroupError`` so the configured fallback applies instead of
  aborting the run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import GPTNeoConfig, GPTNeoForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.looper.awq_processor import (
    AWQProcessor,
    _AWQScaleGroupError,
    _is_replay_geometry_error,
)
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.model import resolve_position_limit


_VOCAB = 128
_HIDDEN = 32
_LIMIT = 64
_ROW_TOKENS = 40


def _tiny_gpt_neo(tmp_path):
    """Two blocks (global + local attention) with GPT-Neo's real mask geometry."""

    model_dir = tmp_path / "tiny-gpt-neo"
    torch.manual_seed(0)
    config = GPTNeoConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=_LIMIT,
        window_size=8,
        intermediate_size=64,
        attention_types=[[["global", "local"], 1]],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    GPTNeoForCausalLM(config).save_pretrained(model_dir)
    backend = Tokenizer(
        models.WordLevel(vocab={f"t{i}": i for i in range(_VOCAB)}, unk_token="t0")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="t0",
        pad_token="t0",
        bos_token="t1",
        eos_token="t2",
    ).save_pretrained(model_dir)
    return model_dir


def _calibration(rows: int = 4, length: int = _ROW_TOKENS) -> list:
    """Each row fits the limit; two rows in one batch do not."""

    generator = torch.Generator().manual_seed(5085)
    return [
        {
            "input_ids": torch.randint(3, _VOCAB, (length,), generator=generator).tolist(),
            "attention_mask": [1] * length,
        }
        for _ in range(rows)
    ]


def _awq_config(**kwargs) -> AWQConfig:
    return AWQConfig(bits=4, group_size=16, device="cpu", **kwargs)


def _load(model_dir):
    return GPTQModel.load(
        str(model_dir), _awq_config(), dtype=torch.bfloat16, backend=BACKEND.AWQ_TORCH
    )


def _token_row_plans(monkeypatch):
    """Record the capture-time token-row plan of every module the search sees.

    #5175 moved token-row sampling from replay (``_pack_token_rows``) to
    capture (``_token_row_plan_for_module``), so this is the plan that decides
    how many rows the replay receives and it must stay inside the position
    limit.
    """

    seen = []
    original = AWQProcessor._token_row_plan_for_module

    def spy(self, module_name, hidden, configured_max_tokens=None):
        plan = original(self, module_name, hidden, configured_max_tokens)
        if plan is not None:
            seen.append(plan)
        return plan

    monkeypatch.setattr(AWQProcessor, "_token_row_plan_for_module", spy)
    return seen


def test_position_limit_resolves_each_declared_key():
    assert resolve_position_limit(SimpleNamespace(max_position_embeddings=_LIMIT)) == _LIMIT
    # GPT-2 declares `n_positions`, MPT `max_seq_len`.
    assert resolve_position_limit({"n_positions": 32, "seq_length": 4096}) == 32
    assert resolve_position_limit({"max_seq_len": 48}) == 48
    # Non-positive or missing values are not limits.
    assert resolve_position_limit(SimpleNamespace(max_position_embeddings=0)) is None
    assert resolve_position_limit(SimpleNamespace()) is None
    assert resolve_position_limit(None) is None
    # A wrapper that nests the decoder config still resolves the real limit.
    assert resolve_position_limit({"text_config": {"max_position_embeddings": 24}}) == 24


def test_token_row_budget_is_capped_at_the_position_limit():
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.gptq_model = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=_LIMIT))
    )
    processor.model = None
    assert processor._position_limit() == _LIMIT
    assert processor._cap_token_rows(200) == _LIMIT
    assert processor._cap_token_rows(16) == 16
    # No declared limit keeps the unclamped budget; an empty sample stays empty.
    processor_no_limit = AWQProcessor.__new__(AWQProcessor)
    processor_no_limit.gptq_model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))
    processor_no_limit.model = None
    assert processor_no_limit._cap_token_rows(200) == 200
    assert processor_no_limit._cap_token_rows(0) == 0


def test_pack_token_rows_keeps_the_cap_it_is_given():
    """The legacy packer (no-provenance fallback) still honours ``max_tokens``."""

    tensors = [
        torch.zeros(1, _ROW_TOKENS, 4),
        torch.zeros(1, _ROW_TOKENS, 4),
    ]
    packed, raw_tokens = AWQProcessor._pack_token_rows(tensors, max_tokens=_LIMIT)

    assert raw_tokens == 2 * _ROW_TOKENS
    assert packed.shape == (1, _LIMIT, 4)


@pytest.mark.cpu
@pytest.mark.integration
def test_awq_gpt_neo_batch_of_two_quantizes_within_the_limit(tmp_path, monkeypatch):
    """Two 40-token rows pack to 80 tokens; the 64-position limit forbids it.

    Before the fix this raised ``RuntimeError: The size of tensor a (64) must
    match the size of tensor b (80) at non-singleton dimension 3`` in layer 0.
    """

    model_dir = _tiny_gpt_neo(tmp_path)
    plans = _token_row_plans(monkeypatch)
    model = _load(model_dir)
    model.quantize(
        CalibrationConfig(dataset=_calibration(), batch_size=2, min_length=1),
        backend=BACKEND.AWQ_TORCH,
    )

    assert plans, "the capture must plan token rows"
    # Two 40-token documents per batch ask for 80 replayed rows; only the cap
    # to the model's 64 declared positions keeps that legal.
    assert max(plan.budget for plan in plans) > _LIMIT
    assert max(plan.max_tokens for plan in plans) == _LIMIT
    for plan in plans:
        assert plan.max_tokens <= _LIMIT
        assert sum(plan.quotas) <= _LIMIT
    # Every quantizable GPT-Neo leaf is packed per block: q/k/v, out_proj,
    # c_fc and c_proj.
    packed = [m for m in model.model.modules() if m.__class__.__name__ == "AwqTorchLinear"]
    assert len(packed) == 6 * 2
    with torch.inference_mode():
        logits = model.model(input_ids=torch.randint(3, _VOCAB, (1, 32))).logits
    assert torch.isfinite(logits).all()


@pytest.mark.cpu
@pytest.mark.integration
def test_control_batch_of_one_keeps_every_row(tmp_path, monkeypatch):
    """A single document below the limit still contributes all of its rows."""

    model_dir = _tiny_gpt_neo(tmp_path)
    plans = _token_row_plans(monkeypatch)
    model = _load(model_dir)
    model.quantize(
        CalibrationConfig(dataset=_calibration(rows=1), batch_size=1, min_length=1),
        backend=BACKEND.AWQ_TORCH,
    )

    assert plans
    for plan in plans:
        assert plan.max_tokens == _ROW_TOKENS
        assert plan.quotas == [_ROW_TOKENS]


def test_replay_geometry_error_becomes_a_scale_group_error(tmp_path, monkeypatch, caplog):
    """A replay that still overruns the module's fixed context must not abort.

    A model whose real mask is bounded below the configured value (or a
    definition that declares its own ``max_tokens``) can still produce an
    over-long replay. Skipping the cap reproduces that: the 80-row packed
    sequence is fed to GPT-Neo's 64-row causal buffer again. The fallback
    contract differs from a plain ``RuntimeError``:

    * ``fallback=None`` propagates ``_AWQScaleGroupError``;
    * a configured fallback quantizes the layer plainly in the same run.
    """

    assert _is_replay_geometry_error(
        RuntimeError(
            "The size of tensor a (64) must match the size of tensor b (80) "
            "at non-singleton dimension 3"
        )
    )
    # Unrelated runtime failures keep propagating.
    assert not _is_replay_geometry_error(RuntimeError("out of memory"))
    assert not _is_replay_geometry_error(ValueError("size of tensor a"))

    model_dir = _tiny_gpt_neo(tmp_path)
    monkeypatch.setattr(AWQProcessor, "_cap_token_rows", lambda self, budget: budget)
    calibration = CalibrationConfig(dataset=_calibration(), batch_size=2, min_length=1)

    strict = GPTQModel.load(
        str(model_dir), _awq_config(fallback=None), dtype=torch.bfloat16, backend=BACKEND.AWQ_TORCH
    )
    with pytest.raises(_AWQScaleGroupError, match="packed token rows"):
        strict.quantize(calibration, backend=BACKEND.AWQ_TORCH)

    tolerant = _load(model_dir)
    tolerant.quantize(calibration, backend=BACKEND.AWQ_TORCH)

    assert "configured plain-quantization fallback" in caplog.text
    packed = [m for m in tolerant.model.modules() if m.__class__.__name__ == "AwqTorchLinear"]
    assert len(packed) == 6 * 2, "the fallback must still quantize every leaf"
