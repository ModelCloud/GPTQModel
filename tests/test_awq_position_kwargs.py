# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ replay must not reuse another sample's position scaffolding.

Issue #4093 item 4: ``AWQProcessor._prepare_module_forward`` only added
``position_ids`` when the key was *absent*. Captured kwargs from a 430-token
sample were therefore replayed with a 522-token activation (EXAONE-3.0), and
48 vs 20 for tiny DeciLM/Nanbeige, crashing in ``apply_rotary_pos_emb``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.awq_processor import AWQProcessor


class _FakeRotary(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.register_buffer("inv_freq", torch.arange(dim, dtype=torch.float32))
        self.dim = dim

    def forward(self, x, position_ids):
        batch = position_ids.shape[0]
        seq = position_ids.shape[-1]
        cos = torch.ones(batch, seq, self.dim)
        sin = torch.zeros(batch, seq, self.dim)
        return cos, sin


class _Attention(nn.Module):
    def __init__(self, dim: int = 8, *, module_rotary: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        if module_rotary:
            self.rotary_emb = _FakeRotary(dim)

    def forward(self, hidden_states, position_ids=None, position_embeddings=None):
        raise AssertionError("only the replay kwargs are under test")


def _processor() -> AWQProcessor:
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.gptq_model = SimpleNamespace(
        rotary_embedding=None,
        awq_preserve_explicit_position_embeddings=False,
    )
    processor.model = SimpleNamespace()
    return processor


def _stale_position_ids(seq_len: int) -> dict:
    return {"position_ids": torch.arange(seq_len).unsqueeze(0)}


def test_stale_position_ids_are_rebuilt_for_the_replayed_sample():
    processor = _processor()
    module = _Attention()
    x = torch.randn(20, 8)

    new_x, kwargs = processor._prepare_module_forward(
        x, module, _stale_position_ids(48)
    )

    assert new_x.shape == (1, 20, 8)
    position_ids = kwargs["position_ids"]
    assert position_ids.shape == (1, 20)
    assert position_ids[0, -1].item() == 19


def test_matching_position_ids_are_preserved():
    processor = _processor()
    module = _Attention()
    x = torch.randn(20, 8)
    captured = torch.arange(100, 120).unsqueeze(0)

    _, kwargs = processor._prepare_module_forward(
        x, module, {"position_ids": captured}
    )

    assert torch.equal(kwargs["position_ids"], captured)


def test_mismatched_position_embeddings_are_dropped_without_rotary():
    processor = _processor()
    module = _Attention()
    x = torch.randn(20, 8)
    stale = (torch.ones(1, 48, 8), torch.zeros(1, 48, 8))

    _, kwargs = processor._prepare_module_forward(
        x,
        module,
        {"position_ids": torch.arange(48).unsqueeze(0), "position_embeddings": stale},
    )

    # No rotary can rebuild them, so the foreign cos/sin must not be replayed.
    assert "position_embeddings" not in kwargs
    assert kwargs["position_ids"].shape == (1, 20)


def test_module_local_rotary_rebuilds_position_embeddings():
    """Nanbeige/DeciLM keep the rotary on the attention, not the model root."""

    processor = _processor()
    module = _Attention(module_rotary=True)
    x = torch.randn(20, 8)
    stale = (torch.ones(1, 48, 8), torch.zeros(1, 48, 8))

    _, kwargs = processor._prepare_module_forward(
        x,
        module,
        {"position_ids": torch.arange(48).unsqueeze(0), "position_embeddings": stale},
    )

    position_embeddings = kwargs["position_embeddings"]
    assert position_embeddings[0].shape == (1, 20, 8)
    assert position_embeddings[1].shape == (1, 20, 8)
    assert kwargs["position_ids"].shape == (1, 20)


REMOTE_AWQ_FAMILIES = {
    "exaone": (
        Path("/monster/data/model/EXAONE-3.0-7.8B-Instruct"),
        {
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "layer_norm_epsilon": 1e-5,
            "max_position_embeddings": 64,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        None,
        "transformer.h.0.attn.attention.q_proj",
    ),
    "decilm": (
        Path("/monster/data/model/DeciLM-7B"),
        {
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            # DeciLM carries a per-layer KV-head list, not one scalar.
            "num_key_value_heads_per_layer": [4, 4],
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 64,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        None,
        "model.layers.0.self_attn.q_proj",
    ),
    "nanbeige": (
        Path("/monster/data/model/Nanbeige4.2-3B"),
        {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "num_loops": 1,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 64,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        None,
        "model.layers.0.self_attn.q_proj",
    ),
}


@pytest.mark.parametrize("family", sorted(REMOTE_AWQ_FAMILIES))
def test_variable_length_awq_runs_on_remote_family(family, tmp_path):
    """AWQ must not replay one sample's ``position_ids`` with another's ``x``."""

    source, overrides, class_ref, q_proj_path = REMOTE_AWQ_FAMILIES[family]
    if not source.is_dir():
        pytest.skip(f"local {family} fixture is unavailable: {source}")

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import AWQConfig
    from tests.tiny_remote_models import build_tiny_remote_model

    path = build_tiny_remote_model(
        source,
        tmp_path / "tiny",
        overrides,
        model_class_ref=class_ref,
        dtype=torch.float16,
    )
    model = GPTQModel.load(
        str(path),
        quantize_config=AWQConfig(bits=4, group_size=32, device="cpu"),
        trust_remote_code=True,
        # The Torch AWQ linear only supports fp16/bf16.
        dtype=torch.float16,
    )
    generator = torch.Generator().manual_seed(0)
    calibration = [
        {
            "input_ids": torch.randint(4, 128, (1, length), generator=generator),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }
        for length in (48, 20)
    ]
    # Sorted descending: the 48-token capture is replayed with the 20-token
    # activation, which is exactly what crashed before the fix.
    model.quantize(calibration, batch_size=1, calibration_sort="desc")

    q_proj = model.model.get_submodule(q_proj_path)
    assert not isinstance(q_proj, nn.Linear)
