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

import pytest
import torch
from torch import nn

from gptqmodel.models.ports.inkling2_loop import experts as E
from gptqmodel.models.ports.inkling2_loop.configuration import Inkling2LoopTextConfig
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


@pytest.mark.parametrize("m,k,num_experts", [(0, 8, 128), (3, 0, 128), (4, 8, 3), (97, 8, 128)])
def test_cpu_expert_routes_match_dense_mask_order(m, k, num_experts):
    generator = torch.Generator().manual_seed(92)
    selected = torch.randint(num_experts, (m, k), generator=generator)
    if m and k:
        selected[0, 1] = selected[0, 0]
    weights = torch.randn(m, k, generator=generator)
    hidden = torch.randn(m, 32, generator=generator)
    mask = nn.functional.one_hot(selected, num_classes=num_experts).permute(2, 1, 0)
    groups = tuple(E._route(hidden, selected, weights, num_experts))
    active = mask.flatten(1).any(1).nonzero().flatten().tolist()
    assert [expert for expert, _, _ in groups] == active
    for expert, tokens, slots in groups:
        expected_slots, expected_tokens = torch.where(mask[expert])
        assert torch.equal(tokens, expected_tokens)
        assert torch.equal(slots, expected_slots)
        assert torch.equal(weights[tokens, slots], weights[expected_tokens, expected_slots])


@pytest.mark.parametrize("invalid,message", [(-1, "non-negative"), (128, "smaller")])
def test_cpu_expert_routes_reject_invalid_ids(invalid, message):
    with pytest.raises(RuntimeError, match=message):
        tuple(E._route(torch.zeros(1, 32), torch.tensor([[invalid]]), torch.ones(1, 1), 128))


@pytest.mark.parametrize("fmt", ("nvfp4", "mxfp8"))
def test_released_expert_gate_up_rows_follow_checkpoint_pair_order(fmt):
    config = _text_config(n_routed_experts=1, experts_formats=[fmt], intermediate_size=32, moe_intermediate_size=32)
    container = E.build_experts(config, 0)
    # Physical rows encode g0=1, u0=2, g1=-1, u1=4 in one input channel.
    # The down projection uses two intermediate channels to expose either
    # a wrong row order or a premature BF16 cast of the routing multiplier.
    with torch.no_grad():
        container.gate_up_weight.zero_()
        container.down_weight.zero_()
        if fmt == "nvfp4":
            container.gate_up_weight[0, :4, 0] = torch.tensor([2, 4, 10, 6], dtype=torch.uint8)
            container.down_weight[0, 0, 0] = 0x12
            for name in ("gate_up_scales", "down_scales", "gate_up_amax", "down_amax",
                         "gate_up_outer_scale", "down_outer_scale"):
                getattr(container, name).fill_(1.0)
        else:
            container.gate_up_weight[0, :4, 0].copy_(torch.tensor([1, 2, -1, 4], dtype=torch.float8_e4m3fn))
            container.down_weight[0, 0, :2].copy_(torch.tensor([1.0, 0.5], dtype=torch.float8_e4m3fn))
            container.gate_up_scales.fill_(127)
            container.down_scales.fill_(127)

    x = torch.zeros((1, config.hidden_size), dtype=torch.bfloat16)
    x[0, 0] = 1.25
    expert = torch.zeros((1, 1), dtype=torch.long)
    gamma = torch.tensor([[0.4731]], dtype=torch.float32)
    gate, up = torch.tensor([1.25, -1.25]), torch.tensor([2.5, 5.0])
    activated = (4 * (gate / 4).tanh() * gate.sigmoid() * (25 * (up / 25).tanh()) * gamma).bfloat16()
    expected = torch.zeros_like(x)
    expected[0, 0] = activated[0, 0].float() + 0.5 * activated[0, 1].float()

    raw = {name: tensor.detach().view(torch.uint8).clone() for name, tensor in container.named_parameters()}
    torch.testing.assert_close(container(x, expert, gamma), expected, rtol=0, atol=0)
    exported = container.dequantize()
    assert exported.gate_up_proj[0, [0, 1, 32, 33], 0].tolist() == [1.0, -1.0, 2.0, 4.0]
    container.defuse()
    leaf = getattr(container, "0")
    assert leaf.gate_proj.dequantize_weight()[:2, 0].tolist() == [1.0, -1.0]
    assert leaf.up_proj.dequantize_weight()[:2, 0].tolist() == [2.0, 4.0]
    torch.testing.assert_close(container(x, expert, gamma), expected, rtol=0, atol=0)
    container.refuse()
    for name, tensor in container.named_parameters():
        assert torch.equal(tensor.detach().view(torch.uint8), raw[name]), name
    torch.testing.assert_close(container(x, expert, gamma), expected, rtol=0, atol=0)


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


def _fill_mxfp8(container: E.Inkling2LoopPackedExperts) -> None:
    with torch.no_grad():
        for name in ("gate_up_weight", "down_weight"):
            getattr(container, name).fill_(1.0)
        for name in ("gate_up_scales", "down_scales"):
            getattr(container, name).fill_(127)


def test_packed_experts_are_iterable_and_apply_the_softcapped_gate():
    """#6108: the defused packed container must behave like the dense one."""

    config = _text_config(experts_formats=["mxfp8"], intermediate_size=32)
    container = E.build_experts(config, 0)
    assert isinstance(container, E.Inkling2LoopPackedExperts)
    _fill_mxfp8(container)

    assert len(container) == 0
    assert list(container) == []
    container.defuse()

    expected_experts = [getattr(container, str(index)) for index in range(config.n_routed_experts)]
    assert len(container) == config.n_routed_experts
    assert list(container) == expected_experts

    gate_up = torch.randn(5, 2 * config.moe_intermediate_size)
    routing_weight = torch.rand(5, 1)
    actual = container._apply_gate(gate_up, routing_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    expected = (
        4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25)
    ) * routing_weight
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    dense = E.Inkling2LoopExperts(config)
    torch.testing.assert_close(
        actual, dense._apply_gate(gate_up, routing_weight), atol=1e-6, rtol=1e-6
    )
    plain_swiglu = nn.functional.silu(gate) * up * routing_weight
    assert not torch.allclose(actual, plain_swiglu), (
        "the packed gate must use the model's softcapped SwiGLU, not plain SwiGLU"
    )


def test_packed_experts_bypass_runs_every_expert_with_softcapped_activation():
    """#6108: ExpertsRoutingBypass must reach every packed expert projection."""

    from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

    config = _text_config(experts_formats=["mxfp8"], intermediate_size=32)
    container = E.build_experts(config, 0)
    _fill_mxfp8(container)
    container.defuse()

    block = nn.Module()
    block.experts = container
    hooks = GateUpDownMoELifecycleHooks()
    # The lifecycle hooks normally resolve this from the model's module plan.
    # A one-container fixture binds the same container name directly.
    hooks.expert_block_names = ("experts",)

    subset = {}
    for index in range(config.n_routed_experts):
        expert = getattr(container, str(index))
        for projection in ("gate_proj", "up_proj", "down_proj"):
            subset[f"mlp.experts.{index}.{projection}"] = getattr(expert, projection)
    names = list(subset)
    hidden = torch.randn(2, 5, config.hidden_size)

    gate_calls = {}
    gate_outputs = {}
    up_outputs = {}
    down_inputs = {}
    handles = []

    def record_gate(index):
        def hook(module, args, output):
            gate_calls[index] = gate_calls.get(index, 0) + 1
            gate_outputs[index] = output.detach().clone()

        return hook

    def record_up(index):
        def hook(module, args, output):
            up_outputs[index] = output.detach().clone()

        return hook

    def record_down(index):
        def hook(module, args):
            down_inputs[index] = args[0].detach().clone()

        return hook

    for index in range(config.n_routed_experts):
        expert = getattr(container, str(index))
        handles.append(expert.gate_proj.register_forward_hook(record_gate(index)))
        handles.append(expert.up_proj.register_forward_hook(record_up(index)))
        handles.append(expert.down_proj.register_forward_pre_hook(record_down(index)))

    class _Processor:
        def record_moe_root_input_feature(self, *args, **kwargs):
            return None

    class _Looper:
        def _set_processor_hooks_paused(self, *args, **kwargs):
            return None

    original_calls = []

    def original_forward(hidden_states, **kwargs):
        original_calls.append(hidden_states.shape)
        return torch.zeros_like(hidden_states)

    try:
        with torch.inference_mode():
            output = hooks.forward_to_all_experts(
                moe_block=block,
                hidden_states=hidden,
                processor=_Processor(),
                subset=subset,
                ordered_module_names=names,
                original_forward=original_forward,
                model_class=None,
                module_looper=_Looper(),
                moe_block_prefix="mlp",
            )
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == hidden.shape
    assert original_calls == [hidden.shape]
    assert gate_calls == dict.fromkeys(range(config.n_routed_experts), 1), (
        "the bypass lifecycle must run every packed expert on the block input"
    )
    for index in range(config.n_routed_experts):
        gate, up = gate_outputs[index], up_outputs[index]
        expected = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25)
        torch.testing.assert_close(down_inputs[index], expected, atol=1e-6, rtol=1e-6)


def test_moe_bypass_warns_when_the_experts_container_cannot_be_iterated(monkeypatch):
    """#6108: a dropped bypass request must not be silent."""

    from gptqmodel.models import moe_lifecycle

    hooks = moe_lifecycle.GateUpDownMoELifecycleHooks()
    hooks.expert_block_names = ("experts",)
    block = nn.Module()
    block.experts = nn.Module()
    subset = {
        f"mlp.experts.0.{projection}": nn.Identity()
        for projection in ("gate_proj", "up_proj", "down_proj")
    }

    warnings = []

    class _Log:
        def warn(self, message):
            warnings.append(message)

    monkeypatch.setattr(moe_lifecycle, "log", _Log())
    original_calls = []

    def original_forward(hidden_states, **kwargs):
        original_calls.append(hidden_states.shape)
        return hidden_states

    output = hooks.forward_to_all_experts(
        moe_block=block,
        hidden_states=torch.randn(1, 4, 8),
        processor=object(),
        subset=subset,
        ordered_module_names=list(subset),
        original_forward=original_forward,
        model_class=None,
        module_looper=object(),
        moe_block_prefix="mlp",
    )

    assert output.shape == (1, 4, 8)
    assert original_calls == [(1, 4, 8)]
    assert warnings, "dropping ExpertsRoutingBypass must be reported"
    assert "not iterable" in warnings[0]
    assert "ExpertsRoutingBypass" in warnings[0]


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
    from gptqmodel.models.ports.inkling2_loop import modeling

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
    from gptqmodel.models.ports.inkling2_loop import modeling

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
    from gptqmodel.models.ports.inkling2_loop import modeling

    config = _text_config(num_hidden_layers=1, experts_formats=["nvfp4"])
    with torch.device("meta"):
        model = modeling.Inkling2LoopTextModel(config)

    E.defuse_packed_experts(model.layers[0])
    expert = getattr(model.layers[0].mlp.experts, "0")
    expert.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False, device="meta")

    with pytest.raises(RuntimeError) as error:
        declare_expert_storage(model)
    message = str(error.value)
    assert "cannot be re-fused" in message
    assert "layers.0.mlp.experts" in message
    assert "contracts/model/refusion.md" in message


class _KernelStub(nn.Module):
    """Stand-in for an installed quantized kernel module."""

    def __init__(self, bits=4, group_size=128):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.sym = True
        self.desc_act = False
        self.pack_dtype = torch.int32
        self.scales_dtype = torch.float16
        self.qzero_dtype = torch.int32
        self.in_features = 32
        self.out_features = 8
        self.qweight = nn.Parameter(torch.zeros(4, 1, dtype=torch.int32), requires_grad=False)
        self.scales = nn.Parameter(torch.zeros(1, 1, dtype=torch.float16), requires_grad=False)


def test_refusion_signature_separates_packed_dense_and_kernel_modules():
    from gptqmodel.utils.refusion import quantization_signature

    config = _text_config(experts_formats=["nvfp4"])
    container = E.build_experts(config, 0)
    _fill_nvfp4(container)
    container.defuse()
    packed = quantization_signature(getattr(container, "0").gate_proj)
    assert (packed.kind, packed.format, packed.bits) == ("packed", "nvfp4", 4)
    assert packed.values_per_scale == 16

    mxfp8 = E.Inkling2LoopPackedExpertLinear(
        in_features=32,
        out_features=16,
        fmt="mxfp8",
        weight=torch.zeros(16, 32, dtype=torch.float8_e4m3fn),
        scales=torch.zeros(16, 1, dtype=torch.uint8),
    )
    assert quantization_signature(mxfp8).bits == 8
    assert quantization_signature(nn.Linear(4, 4)).kind == "dense"
    assert quantization_signature(_KernelStub()).kind == "quantized"
    assert quantization_signature(nn.LayerNorm(4)).kind == "unknown"


def test_refusion_rejects_a_leaf_with_a_different_quantization():
    """Different format/bit width/block size may never share one fused tensor."""

    from gptqmodel.utils.refusion import RefusionContractError

    config = _text_config(experts_formats=["nvfp4"])
    container = E.build_experts(config, 0)
    _fill_nvfp4(container)
    container.defuse()

    # One leaf that reports a different block-scaled format (8-bit instead of
    # 4-bit): the fused container cannot describe both.
    getattr(container, "1").up_proj.fmt = "mxfp8"

    with pytest.raises(RefusionContractError) as error:
        container.refuse("model.language_model.layers.0.mlp.experts")

    message = str(error.value)
    assert "different quantization" in message
    assert "contracts/model/refusion.md" in message
    assert "nvfp4" in message and "mxfp8" in message
    assert "expert 1.up_proj" in message

    # Nothing was mutated: the layer stays defused and unsavable as packed.
    assert container.defused
    assert not hasattr(container, "gate_up_weight")
    for index in range(config.n_routed_experts):
        expert = getattr(container, str(index))
        for projection in ("gate_proj", "up_proj", "down_proj"):
            assert getattr(expert, projection) is not None


def test_refusion_rejects_leaves_that_disagree_with_the_container_declaration():
    """Every leaf agreeing is not enough: the declaration must match too."""

    from gptqmodel.utils.refusion import RefusionContractError

    config = _text_config(experts_formats=["nvfp4"])
    container = E.build_experts(config, 0)
    _fill_nvfp4(container)
    container.defuse()
    for index in range(config.n_routed_experts):
        expert = getattr(container, str(index))
        for projection in ("gate_proj", "up_proj", "down_proj"):
            getattr(expert, projection).fmt = "mxfp8"

    with pytest.raises(RefusionContractError) as error:
        container.refuse("layers.0.mlp.experts")

    message = str(error.value)
    assert "its leaves are mxfp8" in message
    assert "container declares nvfp4" in message
    assert container.defused


def test_refusion_rejects_a_quantized_leaf():
    """A quantized kernel module cannot be merged back into a packed container."""

    from gptqmodel.utils.refusion import RefusionContractError

    config = _text_config(experts_formats=["nvfp4"])
    container = E.build_experts(config, 0)
    _fill_nvfp4(container)
    container.defuse()
    getattr(container, "0").gate_proj = _KernelStub()

    with pytest.raises(RefusionContractError) as error:
        container.refuse("layers.0.mlp.experts")

    message = str(error.value)
    assert "layers.0.mlp.experts cannot be re-fused" in message
    assert "expert 0.gate_proj already holds _KernelStub" in message
    assert "bits=4" in message
    assert "contracts/model/refusion.md" in message
    assert container.defused
    assert not hasattr(container, "gate_up_weight")


def test_checkpoint_conversion_reverses_into_clean_save_keys():
    """`save_pretrained` reverts the mapping; the reverse must yield checkpoint keys."""

    from gptqmodel.models.ports.inkling2_loop.conversion import build_checkpoint_mapping

    mapping = build_checkpoint_mapping()
    cases = {
        "model.llm.embed.weight": "model.language_model.embed_tokens.weight",
        "model.llm.unembed.weight": "lm_head.weight",
        "model.llm.layers.0.mlp.experts.w13_weight": (
            "model.language_model.layers.0.mlp.experts.gate_up_weight"
        ),
        "model.llm.attn_res.attn_queries.3": (
            "model.language_model.layers.4.attention_input.query"
        ),
        "model.llm.attn_res.mlp_queries.5": "model.language_model.layers.5.mlp_input.query",
        "model.visual.backbone.layers.2.ln_1.weight": (
            "model.vision_tower.layers.2.layernorm_before.weight"
        ),
    }
    for checkpoint_key, runtime_key in cases.items():
        forward = checkpoint_key
        for rule in mapping:
            forward, _ = rule.rename_source_key(forward)
        assert forward == runtime_key, checkpoint_key

        back = forward
        for rule in mapping:
            back, _ = rule.reverse_transform().rename_source_key(back)
        assert back == checkpoint_key, runtime_key


class _ExpertsHost(nn.Module):
    """Minimal host so Defuser can walk a model that owns expert containers."""

    def __init__(self, config, experts):
        super().__init__()
        self.config = config
        language_model = nn.Module()
        language_model.layers = nn.ModuleList()
        for module in experts:
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.experts = module
            language_model.layers.append(layer)
        self.model = nn.Module()
        self.model.language_model = language_model


def test_defuser_leaves_a_packed_checkpoint_payload_untouched():
    """Defuser must not decode or split packed experts, and must still expand dense ones.

    ``Inkling2LoopPackedExperts`` is deliberately not decorated with
    ``@use_experts_implementation``: Defuser's generic pass only defuses
    decorator-backed containers, so a checkpoint's NVFP4/MXFP8 bytes survive
    load. ``defuse_packed_experts`` splits them per layer, right before the layer
    is planned, and ``refuse()`` re-fuses the layers the plan skipped.
    """

    import defuser

    from gptqmodel.models.ports.inkling2_loop.configuration import Inkling2LoopConfig

    config = Inkling2LoopConfig(
        text_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
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
            "experts_formats": ["nvfp4", None],
        },
        vision_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "projection_hidden_dim": 16,
            "text_hidden_size": 32,
        },
    )
    text_config = config.text_config
    packed = E.build_experts(text_config, 0)
    dense = E.build_experts(text_config, 1)
    assert isinstance(packed, E.Inkling2LoopPackedExperts)
    assert isinstance(dense, E.Inkling2LoopExperts)
    _fill_nvfp4(packed)

    payload = {name: getattr(packed, name) for name in ("gate_up_weight", "down_weight")}
    pointers = {name: tensor.data_ptr() for name, tensor in payload.items()}
    sums = {name: tensor.float().sum().item() for name, tensor in payload.items()}

    host = _ExpertsHost(config, [packed, dense])
    assert defuser.convert_model(host, cleanup_original=False) is True

    # Packed container: same objects, same bytes, still fused.
    assert host.model.language_model.layers[0].mlp.experts is packed
    assert packed.defused is False
    assert not hasattr(packed, "0")
    for name, tensor in payload.items():
        current = getattr(packed, name)
        assert current is tensor, name
        assert current.data_ptr() == pointers[name], name
        assert current.float().sum().item() == sums[name], name

    # Dense container: Defuser expands it into per-expert linears.
    assert host.model.language_model.layers[1].mlp.experts is dense
    assert getattr(dense, "gate_up_proj", None) is None
    assert getattr(dense, "down_proj", None) is None
    for index in range(text_config.n_routed_experts):
        expert = getattr(dense, str(index))
        for projection, (out_features, in_features) in (
            ("gate_proj", (text_config.moe_intermediate_size, text_config.hidden_size)),
            ("up_proj", (text_config.moe_intermediate_size, text_config.hidden_size)),
            ("down_proj", (text_config.hidden_size, text_config.moe_intermediate_size)),
        ):
            leaf = getattr(expert, projection)
            assert (leaf.out_features, leaf.in_features) == (out_features, in_features), projection
