# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #4874: the AWQ replay must resolve Laguna's rotary label.

``LagunaRotaryEmbedding.forward(x, position_ids, layer_type)`` is typed, but
``LagunaAttention`` stores only ``is_local_attention`` / ``sliding_window``; the
``full_attention`` / ``sliding_attention`` label lives in
``config.layer_types[layer_idx]``. AWQ's replay rebuilds ``position_embeddings``
for each calibration sample, so it needs that label. Without it the rebuild
returned ``None``, the captured pair was dropped for a different-length sample,
and ``LagunaAttention.forward`` raised
``TypeError: missing 1 required positional argument: 'position_embeddings'``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import torch
import torch.nn as nn

from gptqmodel.looper.awq_processor import AWQProcessor


class _TypedRotary(nn.Module):
    def __init__(self, layer_types=("full_attention", "sliding_attention")):
        super().__init__()
        self.layer_types = list(layer_types)
        self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))

    def forward(self, x, position_ids, layer_type=None):
        if layer_type not in self.layer_types:
            raise KeyError(layer_type)
        shape = (*position_ids.shape, 4)
        return torch.ones(shape), torch.zeros(shape)


class _LagunaAttention(nn.Module):
    """No ``layer_type``/``attention_type``/``rope_layer_type`` attribute."""

    def __init__(self, *, local: bool):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.is_local_attention = local
        self.sliding_window = 16

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        raise AssertionError("only the replay scaffolding is under test")


def _processor(rotary, *, layer_types=None, layer_index=None):
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.gptq_model = SimpleNamespace(
        rotary_embedding=None,
        awq_preserve_explicit_position_embeddings=False,
    )
    processor.model = SimpleNamespace(
        rotary_emb=rotary,
        config=SimpleNamespace(layer_types=layer_types) if layer_types is not None else SimpleNamespace(),
    )
    processor._rotary_lock = threading.Lock()
    processor._rotary_cache = {}
    processor._rotary_source_id = None
    processor._scale_context = threading.local()
    if layer_index is not None:
        processor._scale_context.layer_index = layer_index
    return processor


def test_selector_falls_back_to_the_config_layer_type():
    rotary = _TypedRotary()
    attention = _LagunaAttention(local=True)

    processor = _processor(
        rotary, layer_types=["full_attention", "sliding_attention"], layer_index=1
    )

    assert processor._rotary_layer_type_kwargs(rotary, attention) == {
        "layer_type": "sliding_attention"
    }


def test_selector_uses_the_layer_index_of_the_layer_being_scaled():
    rotary = _TypedRotary()
    attention = _LagunaAttention(local=False)

    processor = _processor(
        rotary, layer_types=["full_attention", "sliding_attention"], layer_index=0
    )

    assert processor._rotary_layer_type_kwargs(rotary, attention) == {
        "layer_type": "full_attention"
    }


def test_selector_ignores_a_config_label_the_rotary_cannot_resolve():
    rotary = _TypedRotary()
    attention = _LagunaAttention(local=True)

    processor = _processor(rotary, layer_types=["full_attention", "compressed"], layer_index=1)

    assert processor._rotary_layer_type_kwargs(rotary, attention) is None


def test_selector_keeps_the_attention_attribute_first():
    rotary = _TypedRotary()
    attention = _LagunaAttention(local=False)
    attention.layer_type = "sliding_attention"

    processor = _processor(
        rotary, layer_types=["full_attention", "full_attention"], layer_index=1
    )

    assert processor._rotary_layer_type_kwargs(rotary, attention) == {
        "layer_type": "sliding_attention"
    }


def test_selector_without_a_layer_index_keeps_the_previous_behaviour():
    rotary = _TypedRotary()
    attention = _LagunaAttention(local=True)

    processor = _processor(rotary, layer_types=["full_attention", "sliding_attention"])

    assert processor._rotary_layer_type_kwargs(rotary, attention) is None


def test_awq_quantize_on_laguna_with_varying_lengths_completes(tmp_path):
    import transformers

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
    from tests.tiny_remote_models import build_wordlevel_tokenizer

    config_cls = getattr(transformers, "LagunaConfig", None)
    model_cls = getattr(transformers, "LagunaForCausalLM", None)
    if config_cls is None or model_cls is None:
        import pytest

        pytest.skip("transformers does not provide Laguna")

    model_dir = tmp_path / "laguna"
    torch.manual_seed(0)
    model_cls(
        config_cls(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            layer_types=["full_attention", "sliding_attention"],
            mlp_layer_types=["dense", "sparse"],
            num_attention_heads_per_layer=[4, 4],
            sliding_window=16,
            pad_token_id=0,
        )
    ).to(torch.float16).save_pretrained(model_dir)
    build_wordlevel_tokenizer().save_pretrained(model_dir)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=AWQConfig(bits=4, group_size=32, device="cpu"),
        dtype=torch.float16,
    )

    generator = torch.Generator().manual_seed(0)
    rows = []
    for length in (64, 48, 80, 56, 72, 40):
        input_ids = torch.randint(5, 128, (1, length), generator=generator)
        rows.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})

    model.quantize(CalibrationConfig(dataset=rows, batch_size=4, min_length=1))

    logged = {(int(row["layer"]), str(row["module"])) for row in model.quant_log}
    for layer_index in range(2):
        for projection in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            assert (layer_index, projection) in logged
