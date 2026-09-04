# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import MethodType
from typing import ClassVar

import pytest
import torch
from torch import nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.nn_modules.qvq_grouped_runtime import (
    _is_exact_silu_activation,
    install_qvq_hopper_groups,
    qvq_grouped_runtime_telemetry,
    uninstall_qvq_hopper_groups,
)
from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    unpack_qvq_binary_bank_ids,
)
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
from gptqmodel.utils.qvq_wgmma_cuda import (
    qvq_h100_grouped_ordered_split_counts,
    qvq_p32_window_wgmma_m16_tma_ordered_split,
)


def _child(
    name: str,
    *,
    in_features: int = 256,
    out_features: int = 256,
    bits: float = 3.0,
    su: torch.Tensor | None = None,
    alt_id: int = 1,
    seed: int = 1,
    device: torch.device | str = "cpu",
    input_hadamard: bool = True,
    output_hadamard: bool = True,
) -> QVQLinear:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    tiles = (in_features // 16) * (out_features // 16)
    words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    if su is None:
        su = torch.randn(in_features, generator=generator, device=device)
    tensors = {
        "trellis": torch.randint(
            0,
            1 << 32,
            (tiles, words),
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).to(torch.int32),
        "SU": su.clone(),
        "SV": torch.randn(out_features, generator=generator, device=device),
        "bias": torch.randn(out_features, generator=generator, device=device),
        "bank_ids": pack_qvq_binary_bank_ids(
            torch.randint(
                0,
                2,
                (tiles * 8,),
                generator=generator,
                device=device,
                dtype=torch.uint8,
            )
        ),
        "bank_alt_id": torch.tensor([alt_id], dtype=torch.uint8, device=device),
    }
    return QVQLinear.from_tensors(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        name=name,
        tensors=tensors,
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    ).eval()


class _Attention(nn.Module):
    def __init__(self, children: tuple[QVQLinear, QVQLinear, QVQLinear]):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = children


class _MLP(nn.Module):
    def __init__(self, children: tuple[QVQLinear, QVQLinear]):
        super().__init__()
        self.gate_proj, self.up_proj = children


def test_exact_silu_activation_recognition_is_narrow():
    from transformers.activations import SiLUActivation

    assert _is_exact_silu_activation(torch.nn.functional.silu)
    assert _is_exact_silu_activation(nn.SiLU())
    assert _is_exact_silu_activation(SiLUActivation())
    assert not _is_exact_silu_activation(nn.SiLU(inplace=True))
    assert not _is_exact_silu_activation(nn.GELU())
    assert not _is_exact_silu_activation(lambda value: torch.nn.functional.silu(value))


@pytest.mark.parametrize(
    ("transition_bits", "expected"),
    (
        (4, ((10, 20, 20), (10, 20), (10, 10))),
        (5, ((10, 20, 20), (10, 20), (10, 10))),
        (6, ((10, 20, 20), (4, 20), (10, 10))),
        (7, ((4, 20, 20), (4, 4), (5, 5))),
    ),
)
def test_qwen38_h100_grouped_schedules_preserve_child_split_policies(
    transition_bits, expected
):
    shapes = (
        (12288, 1024, 1024),
        (10240, 6144),
        (17408, 17408),
    )
    actual = tuple(
        qvq_h100_grouped_ordered_split_counts(
            device_name="NVIDIA H100 80GB HBM3",
            compute_capability=(9, 0),
            in_features=5120,
            out_features=shape,
            transition_bits=transition_bits,
        )
        for shape in shapes
    )
    assert actual == expected
    assert qvq_h100_grouped_ordered_split_counts(
        device_name="NVIDIA H200",
        compute_capability=(9, 0),
        in_features=5120,
        out_features=shapes[0],
        transition_bits=transition_bits,
    ) is None


def test_r0_installs_only_bit_identical_input_transforms():
    shared = torch.randn(256, generator=torch.Generator().manual_seed(10))
    accepted = _Attention(
        tuple(
            _child(name, su=shared, alt_id=index + 1, seed=20 + index)
            for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
        )
    )
    rejected = _MLP(
        (
            _child("gate_proj", su=shared, seed=30),
            _child("up_proj", su=shared.clone().add_(1), seed=31),
        )
    )
    model = nn.ModuleDict({"attention": accepted, "mlp": rejected})

    counts = install_qvq_hopper_groups(model)

    assert counts == {"qkv": 1, "gate_up": 0}
    assert all(
        getattr(child, "_gptqmodel_qvq_grouped_runtime", None) is not None
        for child in (accepted.q_proj, accepted.k_proj, accepted.v_proj)
    )
    assert not hasattr(rejected.gate_proj, "_gptqmodel_qvq_grouped_runtime")
    assert install_qvq_hopper_groups(model) == {"qkv": 0, "gate_up": 0}
    assert uninstall_qvq_hopper_groups(model) == 1
    assert not hasattr(accepted.q_proj, "_gptqmodel_qvq_grouped_runtime")


def test_r0_accepts_child_local_output_axes_but_rejects_mixed_input_axes():
    shared = torch.ones(256)
    accepted_children = tuple(
        _child(name, su=shared, alt_id=index + 1, seed=35 + index)
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    accepted_children[2].output_hadamard = False
    accepted = _Attention(accepted_children)
    assert install_qvq_hopper_groups(accepted, gate_up=False) == {"qkv": 1}
    assert uninstall_qvq_hopper_groups(accepted) == 1

    rejected_children = tuple(
        _child(name, su=shared, alt_id=index + 1, seed=38 + index)
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    rejected_children[1].input_hadamard = False
    rejected = _Attention(rejected_children)
    assert install_qvq_hopper_groups(rejected, gate_up=False) == {"qkv": 0}


def test_sibling_lifecycle_fires_once_and_never_returns_stale_output(monkeypatch):
    shared = torch.ones(256)
    attention = _Attention(
        tuple(
            _child(name, su=shared, alt_id=index + 1, seed=40 + index)
            for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
        )
    )
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}
    runtime = attention.q_proj._gptqmodel_qvq_grouped_runtime
    launches = []

    monkeypatch.setattr(runtime, "_runtime_eligible", lambda x: None)

    def execute(x):
        launches.append(x)
        return tuple(
            torch.full((*x.shape[:-1], child.out_features), index + 1.0)
            for index, child in enumerate(
                (attention.q_proj, attention.k_proj, attention.v_proj)
            )
        )

    monkeypatch.setattr(runtime, "_execute", execute)

    def plain(value):
        def forward(_self, x):
            return torch.full((*x.shape[:-1], 256), value)

        return forward

    for index, child in enumerate(
        (attention.q_proj, attention.k_proj, attention.v_proj)
    ):
        child._gptqmodel_qvq_grouped_original_forward = MethodType(
            plain(90 + index), child
        )

    x = torch.randn(2, 256)
    outputs = (attention.q_proj(x), attention.k_proj(x), attention.v_proj(x))
    assert len(launches) == 1
    assert [output[0, 0].item() for output in outputs] == [1, 2, 3]
    assert runtime.telemetry.grouped_launches == 1
    assert runtime.telemetry.sibling_cache_hits == 2

    # A non-primary call outside a cycle uses its original forward.
    assert attention.k_proj(x)[0, 0].item() == 91
    # Starting a new primary abandons any incomplete old cycle. A later
    # out-of-order V must use its own forward, never the abandoned cache.
    assert attention.q_proj(x)[0, 0].item() == 1
    assert attention.v_proj(x)[0, 0].item() == 92
    assert runtime.telemetry.stale_cycles == 1
    assert runtime.telemetry.plain_fallbacks == 2


def test_base_fuse_uses_architecture_roles_and_preserves_qvq_checkpoint_buffers():
    shared = torch.ones(256)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.alpha = _child("alpha", su=shared, alt_id=1, seed=70)
            self.attention.beta = _child("beta", su=shared, alt_id=2, seed=71)
            self.attention.gamma = _child("gamma", su=shared, alt_id=3, seed=72)
            self.mlp = nn.Module()
            self.mlp.first = _child("first", su=shared, alt_id=1, seed=73)
            self.mlp.second = _child("second", su=shared, alt_id=3, seed=74)

    class HFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class QModel(BaseQModel):
        module_tree: ClassVar[list] = [
            "model",
            "layers",
            "#",
            {
                "attention": ("gamma:0:v", "alpha:0:q", "beta:0:k"),
                "mlp": ("second:0:up", "first:0:gate"),
            },
        ]

        def __init__(self, model):
            nn.Module.__init__(self)
            self.model = model
            self.quantized = True
            self.load_quantized_model = True

    model = HFModel().eval()
    qmodel = QModel(model)
    q_trellis = model.layers[0].attention.alpha.trellis
    gate_trellis = model.layers[0].mlp.first.trellis

    counts = qmodel.fuse(free_original_weights=True)

    assert counts == {"qkv": 1, "gate_up": 1}
    assert model.layers[0].attention.alpha.trellis is q_trellis
    assert model.layers[0].mlp.first.trellis is gate_trellis
    state = model.state_dict()
    assert torch.equal(state["layers.0.attention.alpha.trellis"], q_trellis)
    assert torch.equal(state["layers.0.mlp.first.trellis"], gate_trellis)


def test_base_fuse_honors_qvq_only_architecture_group_declarations():
    shared = torch.ones(256)

    class HFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = nn.Module()
            self.linear_attn.packed = _child("packed", su=shared, seed=75)
            self.linear_attn.gate = _child("gate", su=shared, seed=76)

    class QModel(BaseQModel):
        qvq_grouped_p32_candidates = {
            "qkv": (("packed", "gate"),),
        }

        def __init__(self, model):
            nn.Module.__init__(self)
            self.model = model
            self.quantized = True
            self.load_quantized_model = True

    model = HFModel().eval()
    counts = QModel(model).fuse(
        gate_up=False,
        free_original_weights=False,
    )

    assert counts == {"qkv": 1}
    runtime = model.linear_attn.packed._gptqmodel_qvq_grouped_runtime
    assert runtime.member_names == ("packed", "gate")


def _h100_device() -> torch.device | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return None
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) == (9, 0) and "H100" in properties.name:
        return torch.device("cuda", 0)
    return None


def _independent_ordered_qkv_reference(
    children: tuple[QVQLinear, ...], x: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Reference the production split-8 schedule without grouped packing."""

    from gptqmodel.utils.qvq_cuda import _pgc16_levels

    rows = x.numel() // children[0].in_features
    transformed = children[0]._qvq_prepare_inference_input(
        x.reshape(rows, children[0].in_features), torch.float16
    )
    padded = torch.zeros(
        (16, children[0].in_features), device=x.device, dtype=torch.float16
    )
    padded[:rows].copy_(transformed)
    levels = _pgc16_levels(x.device, children[0].codebook_version)
    outputs = []
    for child in children:
        tile_count = (child.in_features // 16) * (child.out_features // 16)
        selectors = pack_qvq_binary_bank_ids(
            unpack_qvq_binary_bank_ids(child.bank_ids, tile_count * 8)
        ).to(device=x.device)
        window = child._prepare_hopper_p32_window(x.device)
        inner = qvq_p32_window_wgmma_m16_tma_ordered_split(
            padded,
            window,
            levels,
            selectors,
            child.bits,
            out_features=child.out_features,
            bank_alt_id=int(child.bank_alt_id.detach().item()),
            split_count=8,
        )
        outputs.append(
            child._qvq_recover_inference_output(inner[:rows], torch.float16)
            .reshape(*x.shape[:-1], child.out_features)
            .to(x.dtype)
        )
    return tuple(outputs)


@pytest.mark.parametrize(
    ("category", "widths"),
    (
        ("qkv", (2048, 512, 512)),
        ("gate_up", (8192, 8192)),
    ),
)
@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
@pytest.mark.parametrize("logical_m", (1, 2, 4, 8, 16))
def test_production_group_is_exact_and_storage_neutral_at_llama32_1b_shapes(
    category, widths, bits, logical_m
):
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    in_features = 2048
    shared = torch.randn(
        in_features,
        generator=torch.Generator(device=device).manual_seed(100),
        device=device,
    )
    names = (
        ("q_proj", "k_proj", "v_proj")
        if category == "qkv"
        else ("gate_proj", "up_proj")
    )
    children = tuple(
        _child(
            name,
            in_features=in_features,
            out_features=width,
                bits=bits,
            su=shared,
            alt_id=index + 1,
            seed=110 + index,
            device=device,
        )
        for index, (name, width) in enumerate(zip(names, widths, strict=True))
    )
    if category == "qkv":
        children[2].output_hadamard = False
    model = _Attention(children) if category == "qkv" else _MLP(children)
    x = (
        torch.randn(
            (logical_m, in_features),
            generator=torch.Generator(device=device).manual_seed(120 + logical_m),
            device=device,
        )
        * 0.02
    ).half()

    with torch.inference_mode():
        previous = tuple(child(x) for child in children)
        expected = (
            _independent_ordered_qkv_reference(children, x)
            if category == "qkv"
            else previous
        )
    assert all(child._qvq_cuda_window_cache is not None for child in children)

    counts = install_qvq_hopper_groups(
        model, qkv=category == "qkv", gate_up=category == "gate_up"
    )
    assert counts[category] == 1
    with torch.inference_mode():
        actual = tuple(getattr(model, name)(x) for name in names)

    assert all(
        torch.equal(output, reference)
        for output, reference in zip(actual, expected, strict=True)
    )
    # Split-K changes FP32 parenthesization relative to the previous split-1
    # path.  Its exact reference is the same ordered schedule executed by
    # independent children, not the arithmetically different split-1 result.
    assert all(torch.isfinite(output).all() for output in actual)
    assert all(child._qvq_cuda_window_cache is None for child in children)
    telemetry = qvq_grouped_runtime_telemetry(model)
    assert len(telemetry) == 1
    expected_window_bytes = sum(
        child.trellis.numel() * child.trellis.element_size() for child in children
    )
    assert telemetry[0]["grouped_launches"] == 1
    assert telemetry[0]["payload_builds"] == 1
    assert telemetry[0]["grouped_window_bytes"] == expected_window_bytes
    assert telemetry[0]["child_window_bytes_avoided"] == expected_window_bytes
    assert telemetry[0]["h100_direct_padded_input_launches"] == (
        1 if logical_m < 16 else 0
    )
    if category == "gate_up":
        assert telemetry[0]["paired_recovery_launches"] == 1
        assert telemetry[0]["independent_recovery_children"] == 0
        assert telemetry[0]["h100_fp16_recovery_store_launches"] == 0
        assert telemetry[0]["h100_w25_n128_gate_up_launches"] == (
            1 if bits == 2.5 else 0
        )
    else:
        assert telemetry[0]["paired_recovery_launches"] == 0
        assert telemetry[0]["independent_recovery_children"] == 3
        assert telemetry[0]["active_split_counts"] == (8, 8, 8)
        assert telemetry[0]["ordered_split_launches"] == 1
        assert telemetry[0]["h100_fp16_recovery_store_launches"] == 2
        assert telemetry[0]["h100_w25_n128_gate_up_launches"] == 0


def test_unequal_gate_up_widths_retain_exact_independent_recovery():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    shared = torch.ones(256, device=device)
    children = (
        _child(
            "gate_proj",
            out_features=256,
            su=shared,
            seed=180,
            device=device,
        ),
        _child(
            "up_proj",
            out_features=512,
            su=shared,
            seed=181,
            device=device,
        ),
    )
    mlp = _MLP(children)
    x = torch.randn((2, 256), device=device, dtype=torch.float16) * 0.02
    with torch.inference_mode():
        expected = tuple(child(x) for child in children)
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}
    with torch.inference_mode():
        actual = (mlp.gate_proj(x), mlp.up_proj(x))

    assert all(
        torch.equal(output, reference)
        for output, reference in zip(actual, expected, strict=True)
    )
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["paired_recovery_launches"] == 0
    assert telemetry["independent_recovery_children"] == 2


def test_warmed_production_group_is_cuda_graph_capturable():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    shared = torch.ones(256, device=device)
    children = tuple(
        _child(
            name,
            su=shared,
            alt_id=index + 1,
            seed=200 + index,
            device=device,
        )
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    attention = _Attention(children)
    install_qvq_hopper_groups(attention, gate_up=False)
    static_input = torch.randn((1, 256), device=device, dtype=torch.float16) * 0.02

    with torch.inference_mode():
        # Build the extension, grouped payload, dtype caches, and allocator
        # pools before capture. Replays execute only captured CUDA work.
        tuple(
            getattr(attention, name)(static_input)
            for name in ("q_proj", "k_proj", "v_proj")
        )
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tuple(
                getattr(attention, name)(static_input)
                for name in ("q_proj", "k_proj", "v_proj")
            )
        graph.replay()
        torch.cuda.synchronize(device)

    assert all(torch.isfinite(output).all() for output in captured)
    telemetry = qvq_grouped_runtime_telemetry(attention)[0]
    # Python runs for warmup and capture; graph replay intentionally does not
    # alter host telemetry.
    assert telemetry["grouped_launches"] == 2
    assert telemetry["sibling_cache_hits"] == 4


def test_qwen38_full_attention_group_runs_measured_schedule_in_cuda_graph():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    shared = torch.ones(5120, device=device)
    widths = (12288, 1024, 1024)
    names = ("q_proj", "k_proj", "v_proj")
    children = tuple(
        _child(
            name,
            in_features=5120,
            out_features=width,
            bits=3,
            su=shared,
            alt_id=index + 1,
            seed=20260910 + index,
            device=device,
            output_hadamard=name != "v_proj",
        )
        for index, (name, width) in enumerate(zip(names, widths, strict=True))
    )
    attention = _Attention(children)
    static_input = torch.randn(
        (1, 5120), device=device, dtype=torch.float16
    ) * 0.02
    with torch.inference_mode():
        plain = tuple(child(static_input).clone() for child in children)
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}
    with torch.inference_mode():
        eager = tuple(getattr(attention, name)(static_input) for name in names)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tuple(
                getattr(attention, name)(static_input) for name in names
            )
        graph.replay()
        torch.cuda.synchronize(device)

    for actual, replayed, reference in zip(eager, captured, plain, strict=True):
        assert torch.equal(replayed, actual)
        torch.testing.assert_close(actual, reference, rtol=0, atol=2e-3)
    telemetry = qvq_grouped_runtime_telemetry(attention)[0]
    assert telemetry["active_split_counts"] == (10, 20, 20)
    assert telemetry["grouped_launches"] == 2
    assert telemetry["plain_fallbacks"] == 0


def test_qwen38_folded_mlp_is_fused_and_cuda_graph_replay_exact():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")

    class QwenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.ones(5120, device=device)
            self.gate_proj = _child(
                "gate_proj",
                in_features=5120,
                out_features=17408,
                bits=3,
                su=shared,
                seed=20260920,
                device=device,
                output_hadamard=False,
            )
            self.up_proj = _child(
                "up_proj",
                in_features=5120,
                out_features=17408,
                bits=3,
                su=shared,
                seed=20260921,
                device=device,
                output_hadamard=False,
            )
            self.down_proj = _child(
                "down_proj",
                in_features=17408,
                out_features=5120,
                bits=3,
                seed=20260922,
                device=device,
                input_hadamard=False,
            )
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(
                self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            )

    mlp = QwenMLP().eval()
    with torch.no_grad():
        for child in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            child.SV.fill_(0.002)
            child.bias.zero_()
    static_input = torch.randn(
        (1, 5120), device=device, dtype=torch.float16
    ) * 0.02
    with torch.inference_mode():
        plain = mlp(static_input).clone()
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}
    assert hasattr(mlp, "_gptqmodel_qvq_fused_mlp_runtime")
    with torch.inference_mode():
        eager = mlp(static_input)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mlp(static_input)
        graph.replay()
        torch.cuda.synchronize(device)

    assert torch.equal(captured, eager)
    torch.testing.assert_close(eager, plain, rtol=0, atol=2e-3)
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["active_split_counts"] == (10, 10)
    assert telemetry["fused_mlp_launches"] == 2
    assert telemetry["h100_folded_qwen_mlp_launches"] == 2
    assert telemetry["h100_folded_qwen_fused_precondition_launches"] == 2
    assert telemetry["h100_folded_qwen_fused_ordered_reduction_launches"] == 2
    assert telemetry["h100_qwen_w3_ordered_decode_prefetch_launches"] == 2
    assert telemetry["h100_qwen_composite_down_recovery_launches"] == 2
    assert telemetry["plain_fallbacks"] == 0
    assert telemetry["fused_mlp_fallbacks"] == 0


def test_unwarmed_group_fails_closed_to_graph_safe_children(monkeypatch):
    """Capture must never run R0 tensor comparisons or payload repacking."""

    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    shared = torch.ones(256, device=device)
    children = tuple(
        _child(
            name,
            su=shared,
            alt_id=index + 1,
            seed=230 + index,
            device=device,
        )
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    attention = _Attention(children)
    static_input = torch.randn((1, 256), device=device, dtype=torch.float16) * 0.02
    with torch.inference_mode():
        expected = tuple(child(static_input).clone() for child in children)
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}
    runtime = attention.q_proj._gptqmodel_qvq_grouped_runtime
    runtime.invalidate()

    def forbidden_payload_build(*_args, **_kwargs):
        raise AssertionError("payload construction ran during CUDA Graph capture")

    monkeypatch.setattr(runtime, "_build_payload", forbidden_payload_build)

    with torch.inference_mode():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tuple(
                getattr(attention, name)(static_input)
                for name in ("q_proj", "k_proj", "v_proj")
            )
        graph.replay()
        torch.cuda.synchronize(device)

    assert all(
        torch.equal(output, reference)
        for output, reference in zip(captured, expected, strict=True)
    )
    telemetry = qvq_grouped_runtime_telemetry(attention)[0]
    assert telemetry["grouped_launches"] == 0
    assert telemetry["plain_fallbacks"] == 3


def test_warmed_gate_up_paired_recovery_is_cuda_graph_capturable():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    shared = torch.ones(256, device=device)
    children = tuple(
        _child(name, su=shared, alt_id=index + 1, seed=250 + index, device=device)
        for index, name in enumerate(("gate_proj", "up_proj"))
    )
    mlp = _MLP(children)
    install_qvq_hopper_groups(mlp, qkv=False)
    static_input = torch.randn((1, 256), device=device, dtype=torch.float16) * 0.02

    with torch.inference_mode():
        expected = (mlp.gate_proj(static_input), mlp.up_proj(static_input))
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = (mlp.gate_proj(static_input), mlp.up_proj(static_input))
        graph.replay()
        torch.cuda.synchronize(device)

    assert torch.equal(captured[0], expected[0])
    assert torch.equal(captured[1], expected[1])
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["paired_recovery_launches"] == 2
    assert telemetry["independent_recovery_children"] == 0


def test_fused_mlp_lifecycle_flag_fallback_and_uninstall_are_exact():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")

    class LlamaLikeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.ones(256, device=device)
            self.gate_proj = _child(
                "gate_proj", su=shared, alt_id=1, seed=270, device=device
            )
            self.up_proj = _child(
                "up_proj", su=shared, alt_id=2, seed=271, device=device
            )
            self.down_proj = _child(
                "down_proj", su=shared, alt_id=3, seed=272, device=device
            )
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    disabled = LlamaLikeMLP().eval()
    assert install_qvq_hopper_groups(disabled, qkv=False, gate_up_activation=False) == {
        "gate_up": 1
    }
    assert not hasattr(disabled, "_gptqmodel_qvq_fused_mlp_runtime")
    assert uninstall_qvq_hopper_groups(disabled) == 1

    mlp = LlamaLikeMLP().eval()
    x = torch.randn((2, 256), device=device, dtype=torch.float16) * 0.02
    fallback_x = torch.randn((17, 256), device=device, dtype=torch.float16) * 0.02
    with torch.inference_mode():
        expected = mlp(x)
        expected_fallback = mlp(fallback_x)
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}
    assert hasattr(mlp, "_gptqmodel_qvq_fused_mlp_runtime")
    with torch.inference_mode():
        actual = mlp(x)
        actual_fallback = mlp(fallback_x)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_fallback, expected_fallback)
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["fused_mlp_launches"] == 1
    assert telemetry["fused_mlp_fallbacks"] == 1
    assert uninstall_qvq_hopper_groups(mlp) == 1
    assert not hasattr(mlp, "_gptqmodel_qvq_fused_mlp_runtime")
    with torch.inference_mode():
        assert torch.equal(mlp(x), expected)


def test_real_llama32_layer_logits_and_cached_generation_are_exact():
    """Exercise the production coordinator inside Transformers' real Llama graph."""

    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        torch_dtype=torch.float16,
    )
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float16).eval()
    layer = model.model.layers[0]
    qkv_su = torch.ones(2048, device=device)
    gate_up_su = torch.ones(2048, device=device)
    replacements = {
        "q_proj": _child(
            "q_proj",
            in_features=2048,
            out_features=2048,
            su=qkv_su,
            alt_id=1,
            seed=300,
            device=device,
        ),
        "k_proj": _child(
            "k_proj",
            in_features=2048,
            out_features=512,
            su=qkv_su,
            alt_id=2,
            seed=301,
            device=device,
        ),
        "v_proj": _child(
            "v_proj",
            in_features=2048,
            out_features=512,
            su=qkv_su,
            alt_id=3,
            seed=302,
            device=device,
        ),
        "gate_proj": _child(
            "gate_proj",
            in_features=2048,
            out_features=8192,
            su=gate_up_su,
            alt_id=1,
            seed=303,
            device=device,
        ),
        "up_proj": _child(
            "up_proj",
            in_features=2048,
            out_features=8192,
            su=gate_up_su,
            alt_id=3,
            seed=304,
            device=device,
        ),
        "down_proj": _child(
            "down_proj",
            in_features=8192,
            out_features=2048,
            su=torch.ones(8192, device=device),
            alt_id=2,
            seed=305,
            device=device,
        ),
    }
    replacements["v_proj"].output_hadamard = False
    # Keep the random synthetic quantized layer in the ordinary activation
    # range so the test measures execution equivalence rather than overflow.
    with torch.no_grad():
        for child in replacements.values():
            child.SV.fill_(0.002)
            child.bias.zero_()
    layer.self_attn.q_proj = replacements["q_proj"]
    layer.self_attn.k_proj = replacements["k_proj"]
    layer.self_attn.v_proj = replacements["v_proj"]
    layer.mlp.gate_proj = replacements["gate_proj"]
    layer.mlp.up_proj = replacements["up_proj"]
    layer.mlp.down_proj = replacements["down_proj"]

    # Sixteen prompt rows exercise the measured Phase-64 paired-recovery-tile
    # policy; cached decoding then returns to the ordinary one-row path.
    input_ids = torch.tensor(
        [[1, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]],
        device=device,
    )
    generation_ids = input_ids[:, :4]
    with torch.inference_mode():
        expected_logits = model(input_ids=input_ids, use_cache=False).logits
        expected_tokens = model.generate(
            input_ids=generation_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )

    counts = install_qvq_hopper_groups(model)
    assert counts == {"qkv": 1, "gate_up": 1}
    with torch.inference_mode():
        actual_logits = model(input_ids=input_ids, use_cache=False).logits
        actual_tokens = model.generate(
            input_ids=generation_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )
        repeated_logits = model(input_ids=input_ids, use_cache=False).logits

    assert torch.equal(actual_logits, repeated_logits)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=2e-3)
    assert torch.equal(actual_tokens, expected_tokens)
    telemetry = qvq_grouped_runtime_telemetry(model)
    assert {entry["category"] for entry in telemetry} == {"qkv", "gate_up"}
    assert all(entry["grouped_launches"] >= 4 for entry in telemetry)
    assert all(entry["plain_fallbacks"] == 0 for entry in telemetry)
    assert all(
        entry["h100_direct_padded_input_launches"] >= 3 for entry in telemetry
    )
    assert all(
        entry["h100_multiblock_input_hadamard_launches"] >= 4
        for entry in telemetry
    )
    assert all(entry["h100_fp16_recovery_store_launches"] >= 4 for entry in telemetry)
    gate_up_telemetry = next(
        entry for entry in telemetry if entry["category"] == "gate_up"
    )
    assert gate_up_telemetry["fused_mlp_launches"] >= 4
    assert gate_up_telemetry["fused_mlp_fallbacks"] == 0
    assert gate_up_telemetry["h100_multiblock_recovery_launches"] >= 4
    assert gate_up_telemetry["h100_warp_recovery_low_launches"] >= 4
    assert gate_up_telemetry["h100_fused_recovery_precondition_launches"] >= 4
    assert gate_up_telemetry["h100_paired_recovery_tiles_launches"] >= 1
    assert gate_up_telemetry["h100_bounded_recovery_rounding_launches"] >= 1
    assert gate_up_telemetry["h100_packed_gate_up_recovery_launches"] >= 1
    assert gate_up_telemetry["h100_multiblock_precondition_launches"] >= 4
    assert gate_up_telemetry["h100_half2_precondition_high_launches"] >= 4
    assert gate_up_telemetry["h100_fused_silu_precondition_low_launches"] >= 4
    assert gate_up_telemetry["h100_half2_precondition_low_launches"] >= 4
    assert gate_up_telemetry["h100_direct_padded_precondition_launches"] >= 3
    assert gate_up_telemetry["h100_fused_down_reduction_recovery_launches"] >= 4
    assert gate_up_telemetry["h100_multiblock_down_recovery_launches"] >= 4
