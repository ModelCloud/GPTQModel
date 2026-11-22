# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ scale-search rotary selection for decoupled rope types (#4695).

DeepSeek V4 keeps architecture layer types (``sliding_attention`` /
``compressed_sparse_attention``) separate from the rope labels its rotary
indexes (``main`` / ``compress``). ``AWQProcessor._prepare_module_forward``
used to rebuild ``position_embeddings`` with the architecture label, which
raised ``KeyError: 'sliding_attention'`` in the first scale search, and it
always returned a tuple even though V4's attention indexes a mapping.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from gptqmodel import GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
from tests.tiny_remote_models import build_wordlevel_tokenizer


class _TypedRotary(nn.Module):
    """Minimal multi-label rotary that rejects unknown layer types."""

    def __init__(self, layer_types=("main", "compress")):
        super().__init__()
        self.layer_types = list(layer_types)
        self.rope_type = dict.fromkeys(self.layer_types, "default")
        self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))

    def forward(self, x, position_ids, layer_type=None):
        if layer_type not in self.rope_type:
            raise KeyError(layer_type)
        shape = (*position_ids.shape, 4)
        return torch.ones(shape), torch.zeros(shape)


class _DecoupledAttention(nn.Module):
    def __init__(self, *, rope_layer_type="main"):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.layer_type = "sliding_attention"
        self.rope_layer_type = rope_layer_type

    def forward(self, hidden_states, position_embeddings, position_ids):
        raise AssertionError("only the replay scaffolding is under test")


def _processor(rotary: nn.Module) -> AWQProcessor:
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.gptq_model = SimpleNamespace(
        rotary_embedding=None,
        awq_preserve_explicit_position_embeddings=False,
    )
    processor.model = SimpleNamespace(rotary_emb=rotary)
    processor._rotary_lock = threading.Lock()
    processor._rotary_cache = {}
    processor._rotary_source_id = None
    return processor


@pytest.mark.parametrize(
    ("rope_layer_type", "expected"),
    (("main", {"layer_type": "main"}), ("compress", {"layer_type": "compress"})),
)
def test_selector_prefers_decoupled_rope_layer_type(rope_layer_type, expected):
    rotary = _TypedRotary()
    attention = _DecoupledAttention(rope_layer_type=rope_layer_type)

    assert _processor(rotary)._rotary_layer_type_kwargs(rotary, attention) == expected


def test_architecture_layer_type_is_not_passed_when_rotary_cannot_resolve_it():
    rotary = _TypedRotary()
    attention = _DecoupledAttention()
    del attention.rope_layer_type

    assert _processor(rotary)._rotary_layer_type_kwargs(rotary, attention) is None


def test_mapping_position_embeddings_are_rebuilt_per_rope_type():
    rotary = _TypedRotary()
    attention = _DecoupledAttention()
    processor = _processor(rotary)
    stale = {
        name: (torch.ones(1, 48, 4), torch.zeros(1, 48, 4))
        for name in rotary.layer_types
    }

    new_x, kwargs = processor._prepare_module_forward(
        torch.randn(20, 8),
        attention,
        {
            "position_ids": torch.arange(48).unsqueeze(0),
            "position_embeddings": stale,
        },
    )

    assert new_x.shape == (1, 20, 8)
    assert kwargs["position_ids"].shape == (1, 20)
    position_embeddings = kwargs["position_embeddings"]
    assert set(position_embeddings) == {"main", "compress"}
    for cos, sin in position_embeddings.values():
        assert cos.shape == (1, 20, 4)
        assert sin.shape == (1, 20, 4)


def test_unresolvable_selector_keeps_matching_captured_embeddings():
    rotary = _TypedRotary()
    attention = _DecoupledAttention()
    del attention.rope_layer_type
    captured = (torch.ones(1, 20, 4), torch.zeros(1, 20, 4))

    _, kwargs = _processor(rotary)._prepare_module_forward(
        torch.randn(20, 8),
        attention,
        {
            "position_ids": torch.arange(20).unsqueeze(0),
            "position_embeddings": captured,
        },
    )

    assert kwargs["position_embeddings"] is captured


class _BatchEcho(nn.Module):
    def forward(self, x, position_embeddings=None):
        assert position_embeddings is not None
        for cos, sin in position_embeddings.values():
            assert cos.shape[0] == x.shape[0]
            assert sin.shape[0] == x.shape[0]
        return x


def test_chunked_replay_slices_mapping_position_embeddings():
    processor = _processor(_TypedRotary())
    processor._quant_batch_size = 1
    embeddings = {
        name: (torch.ones(2, 3, 4), torch.zeros(2, 3, 4))
        for name in ("main", "compress")
    }

    outputs = list(
        processor._iter_prepared_forward_outputs(
            torch.randn(2, 3, 4),
            _BatchEcho(),
            {"position_embeddings": embeddings},
        )
    )

    assert [tuple(output.shape) for output in outputs] == [(1, 3, 4), (1, 3, 4)]


def _tiny_v4_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        n_routed_experts=3,
        num_experts_per_tok=2,
        hc_mult=2,
        o_groups=2,
        o_lora_rank=8,
        sliding_window=8,
        # Mixed architecture types exercise both rope labels.
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "compressed_sparse_attention",
        ],
        mlp_layer_types=["moe"] * 3,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=4,
        dspark_block_size=5,
        dspark_noise_token_id=31,
        partial_rotary_factor=0.5,
        dtype="bfloat16",
    )


@pytest.mark.cpu
def test_awq_quantizes_tiny_deepseek_v4_with_decoupled_rope_types(tmp_path):
    config = _tiny_v4_config()
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    source = tmp_path / "source"
    DeepseekV4ForCausalLM(config).save_pretrained(source)
    build_wordlevel_tokenizer(vocab_size=32).save_pretrained(source)

    model = GPTQModel.load(
        str(source),
        quantize_config=AWQConfig(
            bits=4,
            group_size=16,
            device="cpu",
            offload_to_disk=False,
        ),
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    calibration = [
        {
            "input_ids": torch.randint(4, 32, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        for _ in range(2)
    ]

    model.quantize(CalibrationConfig(dataset=calibration, batch_size=1))

    assert any(
        isinstance(module, AwqTorchLinear) for module in model.model.modules()
    )
    with torch.inference_mode():
        logits = model.model(**calibration[0], use_cache=False).logits
    assert logits.shape == (1, 16, 32)
    assert torch.isfinite(logits).all()
