# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""TeleChat2 must quantize ``query``/``key_value`` and survive AWQ replay.

Issue #4093 item 3: ``definitions/telechat2.py`` declared only
``self_attention.dense``, so the attention's input projections stayed in the
source dtype while the coverage report still claimed every declared spec
matched. AWQ then crashed replaying ``TelechatAttention`` because its forward
needs ``residual`` and ``attention_mask``, which the per-linear activation
capture never records.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.models.definitions.telechat2 import TeleChat2QModel


TELEHCAT2_PATH = Path("/monster/data/model/TeleChat2-7B")


def test_telechat2_tree_declares_attention_input_projections():
    """The tree must own every quantizable Linear in the attention block."""

    config = SimpleNamespace(model_type="telechat", n_layer=2, num_key_value_heads=2)
    blocks = TeleChat2QModel.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat = {name for block in blocks for name in block}

    assert {
        "self_attention.query",
        "self_attention.key_value",
        "self_attention.dense",
    } <= flat
    # q/kv share the attention input; `dense` consumes the attention output.
    assert any(
        {"self_attention.query", "self_attention.key_value"} <= set(block)
        for block in blocks
    )
    assert any(block == ["self_attention.dense"] for block in blocks)


def test_telechat2_awq_replay_supplies_residual_and_mask():
    """The AWQ block replay must pass the arguments TeleChat's forward requires."""

    class _Attention(nn.Module):
        def forward(self, hidden_states, residual, attention_mask, position_ids=None):
            raise AssertionError("replay kwargs are what this test exercises")

    instance = TeleChat2QModel.__new__(TeleChat2QModel)
    x = torch.randn(5, 8)
    kwargs = {"position_ids": torch.arange(5).unsqueeze(0)}

    new_x, new_kwargs = TeleChat2QModel.awq_module_replay_inputs(
        instance, _Attention(), x, kwargs
    )

    # The per-linear capture is 2-D; the attention forward is 3-D.
    assert new_x.dim() == 3
    assert new_x.shape[-2] == 5
    assert torch.equal(new_kwargs["residual"], new_x)
    mask = new_kwargs["attention_mask"]
    assert mask.dtype == torch.bool
    assert mask.shape == (5, 5)
    # Causal: strictly upper triangle is masked.
    assert torch.equal(mask, mask.triu(1))
    assert not bool(mask.tril(0).any())


@pytest.mark.skipif(
    not TELEHCAT2_PATH.is_dir(),
    reason=f"local TeleChat2 fixture is unavailable: {TELEHCAT2_PATH}",
)
def test_telechat2_awq_quantizes_attention_projections(tmp_path, monkeypatch):
    """End-to-end: AWQ runs, and no quantizable attention Linear stays dense."""

    from gptqmodel import GPTQModel
    from gptqmodel.looper.module_coverage import LayerModuleCoverageTracker
    from gptqmodel.quantization.config import AWQConfig
    from tests.tiny_remote_models import build_tiny_remote_model

    trackers = []
    original_init = LayerModuleCoverageTracker.__init__

    def tracking_init(self):
        original_init(self)
        trackers.append(self)

    monkeypatch.setattr(LayerModuleCoverageTracker, "__init__", tracking_init)

    path = build_tiny_remote_model(
        TELEHCAT2_PATH,
        tmp_path / "tiny",
        {
            "vocab_size": 128,
            # AWQ's scale search requires the block output width to be a
            # multiple of 64 (`awq_processor.py` oc_batch_size).
            "hidden_size": 128,
            "n_layer": 2,
            "n_head": 4,
            "num_key_value_heads": 4,
            "ffn_hidden_size": 256,
            "layer_norm_epsilon": 1e-5,
            "flash_attn": False,
            "pretraining_tp": 1,
            "training_seqlen": 32,
            "base_seqlen": 32,
            "seq_length": 32,
            "max_position_embeddings": 64,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 3,
        },
        model_class_ref="modeling_telechat2.Telechat2ForCausalLM",
        architecture="Telechat2ForCausalLM",
    )

    model = GPTQModel.load(
        str(path),
        quantize_config=AWQConfig(bits=4, group_size=32, device="cpu"),
        trust_remote_code=True,
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(0)
    calibration = [
        {
            "input_ids": torch.randint(4, 128, (1, 24), generator=generator),
            "attention_mask": torch.ones(1, 24, dtype=torch.long),
        }
        for _ in range(2)
    ]
    model.quantize(calibration, batch_size=1)

    attention = model.model.transformer.h[0].self_attention
    for name in ("query", "key_value", "dense"):
        module = attention.get_submodule(name)
        assert not isinstance(module, nn.Linear), (
            f"self_attention.{name} stayed in the source dtype"
        )

    # The reporting bug: every declared spec matched while two Linears were
    # never declared. With the fixed tree the tracker must have no such hole.
    assert trackers, "quantization did not create a coverage tracker"
    for tracker in trackers:
        assert tracker.undeclared_snapshot() == {}
