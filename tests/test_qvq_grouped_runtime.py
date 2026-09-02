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
    install_qvq_hopper_groups,
    qvq_grouped_runtime_telemetry,
    uninstall_qvq_hopper_groups,
)
from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile


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
    ).eval()


class _Attention(nn.Module):
    def __init__(self, children: tuple[QVQLinear, QVQLinear, QVQLinear]):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = children


class _MLP(nn.Module):
    def __init__(self, children: tuple[QVQLinear, QVQLinear]):
        super().__init__()
        self.gate_proj, self.up_proj = children


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


def _h100_device() -> torch.device | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return None
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) == (9, 0) and "H100" in properties.name:
        return torch.device("cuda", 0)
    return None


@pytest.mark.parametrize(
    ("category", "widths"),
    (
        ("qkv", (2048, 512, 512)),
        ("gate_up", (8192, 8192)),
    ),
)
@pytest.mark.parametrize("logical_m", (1, 2, 4, 8, 16))
def test_production_group_is_exact_and_storage_neutral_at_llama32_1b_shapes(
    category, widths, logical_m
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
            bits=3,
            su=shared,
            alt_id=index + 1,
            seed=110 + index,
            device=device,
        )
        for index, (name, width) in enumerate(zip(names, widths, strict=True))
    )
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
        expected = tuple(child(x) for child in children)
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
    }
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

    input_ids = torch.tensor([[1, 7, 11, 19]], device=device)
    with torch.inference_mode():
        expected_logits = model(input_ids=input_ids, use_cache=False).logits
        expected_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )

    counts = install_qvq_hopper_groups(model)
    assert counts == {"qkv": 1, "gate_up": 1}
    with torch.inference_mode():
        actual_logits = model(input_ids=input_ids, use_cache=False).logits
        actual_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )

    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_tokens, expected_tokens)
    telemetry = qvq_grouped_runtime_telemetry(model)
    assert {entry["category"] for entry in telemetry} == {"qkv", "gate_up"}
    assert all(entry["grouped_launches"] >= 4 for entry in telemetry)
    assert all(entry["plain_fallbacks"] == 0 for entry in telemetry)
