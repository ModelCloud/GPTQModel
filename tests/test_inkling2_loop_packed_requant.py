# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Pre-quantized Inkling 2 Mini experts are defused and re-quantizable.

A checkpoint that already stores NVFP4/MXFP8 experts is re-quantized, never
passed through: every expert becomes a real ``nn.Linear`` so calibration is
collected per module, its forward stays on the packed W4A16/W8A16 kernel, and
the dense bf16 source is materialized only when a processor asks for
``weight``.
"""

from __future__ import annotations

import torch
from torch import nn

from gptqmodel.models.inkling2_loop import experts as E
from gptqmodel.models.inkling2_loop.configuration import Inkling2LoopTextConfig
from gptqmodel.utils.model import find_modules


def _text_config(**overrides) -> Inkling2LoopTextConfig:
    payload = {
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 16,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8,
        "vocab_size": 64,
        "unpadded_vocab_size": 64,
        "n_layers_per_attn_res_block": 3,
        "d_rel": 2,
        "rel_extent": 4,
        "sliding_window_size": 4,
    }
    payload.update(overrides)
    return Inkling2LoopTextConfig(**payload)


def _fill_nvfp4(container: E.Inkling2LoopPackedExperts, seed: int = 0) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in ("gate_up_weight", "down_weight"):
            target = getattr(container, name)
            codes = torch.randint(
                0, 256, target.shape, generator=generator, dtype=torch.int64
            ).to(torch.uint8)
            target.copy_(codes)
        for name in ("gate_up_scales", "down_scales"):
            scales = getattr(container, name)
            scales.copy_(torch.full(scales.shape, 126, dtype=torch.uint8).view(torch.float8_e4m3fn))
        for name in (
            "gate_up_amax",
            "down_amax",
            "gate_up_outer_scale",
            "down_outer_scale",
        ):
            getattr(container, name).fill_(1.0)


def _route_inputs(config, dtype=torch.float32):
    hidden = torch.randn(4, config.hidden_size, dtype=dtype)
    top_k_index = torch.tensor([[0], [1], [0], [1]])
    top_k_weights = torch.tensor([[0.7], [0.3], [0.5], [0.5]], dtype=dtype)
    return hidden, top_k_index, top_k_weights


def test_packed_experts_defuse_into_linear_leaves_without_decoding():
    config = _text_config(experts_formats=["nvfp4"])
    container = E.build_experts(config, 0)
    assert isinstance(container, E.Inkling2LoopPackedExperts)
    _fill_nvfp4(container)

    hidden, top_k_index, top_k_weights = _route_inputs(config)
    reference = container(hidden, top_k_index, top_k_weights)

    assert container.defuse() is container
    assert container.defused
    # Fused storage is gone; the leaves hold views of the same checkpoint bytes.
    assert [name for name, _ in container.named_parameters() if "." not in name] == []

    for index in range(config.n_routed_experts):
        expert = getattr(container, str(index))
        for projection, (out_features, in_features) in (
            ("gate_proj", (config.moe_intermediate_size, config.hidden_size)),
            ("up_proj", (config.moe_intermediate_size, config.hidden_size)),
            ("down_proj", (config.hidden_size, config.moe_intermediate_size)),
        ):
            leaf = getattr(expert, projection)
            assert isinstance(leaf, nn.Linear), projection
            assert leaf.in_features == in_features
            assert leaf.out_features == out_features
            assert leaf.packed_weight is not None

    # Defusion is a view split, so the routed output is unchanged.
    torch.testing.assert_close(
        container(hidden, top_k_index, top_k_weights), reference, rtol=1e-3, atol=1e-3
    )


def test_packed_expert_linear_forwards_on_kernel_and_defers_dense_weight(monkeypatch):
    leaf = E.Inkling2LoopPackedExpertLinear(
        in_features=32,
        out_features=16,
        fmt="nvfp4",
        weight=torch.zeros(16, 16, dtype=torch.uint8),
        scales=torch.full((16, 2), 126, dtype=torch.uint8).view(torch.float8_e4m3fn),
        global_scale=torch.tensor(1.0),
    )

    calls = []
    monkeypatch.setattr(E, "scaled_wxa16_available", lambda device=None: True)

    def fake_kernel(x, w_packed, w_scales, global_scale=1.0, **kwargs):
        calls.append((tuple(x.shape), tuple(w_packed.shape)))
        return torch.zeros(x.shape[:-1] + (w_packed.shape[0],), dtype=x.dtype)

    monkeypatch.setattr(E, "nvfp4_w4a16_linear", fake_kernel)

    out = leaf(torch.randn(3, 32, dtype=torch.bfloat16))
    assert calls == [((3, 32), (16, 16))]
    assert out.shape == (3, 16)
    # The kernel forward never materializes the dense source.
    assert "_dense_weight" not in leaf._parameters

    weight = leaf.weight
    assert weight.dtype == torch.bfloat16
    assert tuple(weight.shape) == (16, 32)
    assert leaf._parameters["_dense_weight"] is weight


def test_packed_expert_linear_falls_back_to_dense_bf16_without_kernel(monkeypatch):
    leaf = E.Inkling2LoopPackedExpertLinear(
        in_features=32,
        out_features=16,
        fmt="mxfp8",
        weight=torch.zeros(16, 32, dtype=torch.float8_e4m3fn),
        scales=torch.full((16, 1), 127, dtype=torch.uint8),
    )
    monkeypatch.setattr(E, "scaled_wxa16_available", lambda device=None: False)

    x = torch.randn(2, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(
        leaf(x), torch.nn.functional.linear(x, leaf.dequantize_weight())
    )


def test_defused_packed_layers_resolve_quantizable_specs():
    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.models.inkling2_loop import modeling

    definition = MODEL_MAP["inkling2_loop"]
    config = _text_config(num_hidden_layers=2, experts_formats=["nvfp4", "mxfp8"])

    with torch.device("meta"):
        model = modeling.Inkling2LoopTextModel(config)

    layers = model.layers
    assert E.defuse_packed_experts(layers[0]) == 1
    assert E.defuse_packed_experts(layers[0]) == 0  # idempotent

    resolved = set(find_modules(layers[0]))
    for index in range(config.n_routed_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            path = f"mlp.experts.{index}.{projection}"
            assert path in resolved, path

    blocks = definition.build_layer_modules(definition.module_tree, include_capture_only=True)
    expert_blocks = [block for block in blocks if any("mlp.experts." in name for name in block)]
    assert expert_blocks, "expert projections must stay in the quantization plan"
    assert any("mlp.experts.{expert_index}.down_proj" in block for block in expert_blocks)


def test_packed_experts_refuse_restores_the_fused_checkpoint_storage():
    config = _text_config(experts_formats=["nvfp4"])
    container = E.build_experts(config, 0)
    _fill_nvfp4(container)
    hidden, top_k_index, top_k_weights = _route_inputs(config)
    reference = container(hidden, top_k_index, top_k_weights)
    original = {
        name: getattr(container, name).detach().clone()
        for name in (
            "gate_up_weight",
            "gate_up_scales",
            "down_weight",
            "down_scales",
            "gate_up_amax",
            "down_amax",
            "gate_up_outer_scale",
            "down_outer_scale",
        )
    }

    container.defuse()
    assert container.refuse() is container
    assert not container.defused
    assert not hasattr(container, "0")
    for name, value in original.items():
        restored = getattr(container, name)
        assert restored.dtype == value.dtype, name
        assert torch.equal(restored.detach(), value), name

    torch.testing.assert_close(
        container(hidden, top_k_index, top_k_weights), reference, rtol=1e-3, atol=1e-3
    )


def test_after_quantize_declares_the_storage_a_reload_will_build():
    from gptqmodel.models.definitions.inkling2_loop import (
        _layer_index_from_name,
        declare_expert_storage,
    )
    from gptqmodel.models.inkling2_loop import modeling

    config = _text_config(num_hidden_layers=2, experts_formats=["nvfp4", "mxfp8"])
    with torch.device("meta"):
        model = modeling.Inkling2LoopTextModel(config)

    # Layer 0 was defused and its leaves replaced by quantized modules.
    E.defuse_packed_experts(model.layers[0])
    for index in range(config.n_routed_experts):
        expert = getattr(model.layers[0].mlp.experts, str(index))
        for projection, (out_features, in_features) in (
            ("gate_proj", (config.moe_intermediate_size, config.hidden_size)),
            ("up_proj", (config.moe_intermediate_size, config.hidden_size)),
            ("down_proj", (config.hidden_size, config.moe_intermediate_size)),
        ):
            setattr(
                expert,
                projection,
                nn.Linear(in_features, out_features, bias=False, device="meta"),
            )
    # Layer 1 was defused but the plan skipped it, so it keeps packed storage.
    E.defuse_packed_experts(model.layers[1])

    assert _layer_index_from_name("model.language_model.layers.7.mlp.experts") == 7
    declare_expert_storage(model)

    assert config.experts_formats == [None, "mxfp8"]
    assert not model.layers[1].mlp.experts.defused
    assert model.layers[1].mlp.experts.gate_up_weight is not None


def test_partially_quantized_expert_layer_is_reported_not_saved_mixed():
    from gptqmodel.models.definitions.inkling2_loop import declare_expert_storage
    from gptqmodel.models.inkling2_loop import modeling

    config = _text_config(num_hidden_layers=1, experts_formats=["nvfp4"])
    with torch.device("meta"):
        model = modeling.Inkling2LoopTextModel(config)

    E.defuse_packed_experts(model.layers[0])
    expert = getattr(model.layers[0].mlp.experts, "0")
    expert.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False, device="meta")

    try:
        declare_expert_storage(model)
    except RuntimeError as error:
        assert "cannot be re-fused" in str(error)
    else:  # pragma: no cover - the mixed layout must never be accepted silently
        raise AssertionError("mixed packed/quantized expert layer was accepted")
