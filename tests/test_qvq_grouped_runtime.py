# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import replace
from types import MethodType
from typing import ClassVar

import pytest
import torch
from torch import nn

from gptqmodel.models.base import BaseQModel, _qvq_quantization_group_candidates
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.nn_modules.qvq_grouped_runtime import (
    QVQHopperGroupedRuntime,
    _child_window_source,
    _is_exact_silu_activation,
    _source_key,
    _validate_static_group,
    install_qvq_hopper_groups,
    qvq_grouped_runtime_telemetry,
    uninstall_qvq_hopper_groups,
)
from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    repack_p32_planar_to_window,
    unpack_qvq_binary_bank_ids,
)
from gptqmodel.quantization.qvq_rank8 import (
    P32WindowConfig,
    prepare_rank8,
    window_kernel_candidates,
)
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
from gptqmodel.utils.qvq_wgmma_cuda import (
    qvq_fp16_to_fp8_e5m2_clamped,
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
    activation: dict | bool | None = None,
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
        activation=activation,
    ).eval()


class _Attention(nn.Module):
    def __init__(self, children: tuple[QVQLinear, QVQLinear, QVQLinear]):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = children


class _MLP(nn.Module):
    def __init__(self, children: tuple[QVQLinear, QVQLinear]):
        super().__init__()
        self.gate_proj, self.up_proj = children


def _h200_device() -> torch.device | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return None
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) == (9, 0) and "H200" in properties.name:
        return torch.device("cuda", 0)
    return None


def test_exact_silu_activation_recognition_is_narrow():
    from transformers.activations import SiLUActivation

    assert _is_exact_silu_activation(torch.nn.functional.silu)
    assert _is_exact_silu_activation(nn.SiLU())
    assert _is_exact_silu_activation(SiLUActivation())
    assert not _is_exact_silu_activation(nn.SiLU(inplace=True))
    assert not _is_exact_silu_activation(nn.GELU())
    assert not _is_exact_silu_activation(lambda value: torch.nn.functional.silu(value))


def test_group_validation_uses_window_only_child_storage():
    shared = torch.ones(256)
    children = tuple(
        _child(name, su=shared, seed=91 + index)
        for index, name in enumerate(("q_proj", "k_proj"))
    )
    for child in children:
        planar = child.trellis
        child.window_words = repack_p32_planar_to_window(planar, bits=child.bits)
        child.window_only = True
        child.trellis = None

    resolved = _validate_static_group(children)
    assert resolved == children
    assert all(_child_window_source(child) is child.window_words for child in children)
    key_before = _source_key(children)
    children[0].window_words.add_(1)
    assert _source_key(children) != key_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_correction_off_ignores_fused_recovery_metadata():
    """All-off grouped candidates must not touch absent rank8 factors."""
    device = _h200_device() or _h100_device()
    if device is None:
        pytest.skip("requires an H100 or H200 SM90 validation device")
    shared = torch.ones(2048, device=device)
    children = tuple(
        _child(
            name,
            in_features=2048,
            out_features=256,
            su=shared,
            seed=20261100 + index,
            device=device,
            input_hadamard=False,
            output_hadamard=False,
        )
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    policy = P32WindowConfig(
        recovery_mode="off", recovery_kernel="fused_epilogue"
    )
    for child in children:
        prepare_rank8(child, policy)
    parent = _Attention(children)
    assert install_qvq_hopper_groups(parent, gate_up=False) == {"qkv": 1}
    x = torch.randn(1, 2048, device=device, dtype=torch.float16) * 0.01
    outputs = (parent.q_proj(x), parent.k_proj(x), parent.v_proj(x))
    assert all(torch.isfinite(output).all() for output in outputs)


def test_window_only_dense_reference_reconstructs_planar_temporarily():
    child = _child("q_proj", seed=93)
    expected = child.get_inner_weight_tensor()
    x = torch.randn(2, 256)
    expected_output = child(x).clone()
    child.window_words = repack_p32_planar_to_window(child.trellis, bits=child.bits)
    child.window_only = True
    child.trellis = None

    actual = child.get_inner_weight_tensor()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(child(x), expected_output, rtol=0, atol=0)


def test_window_planar_fallback_cache_is_transient_across_copy():
    child = _child("q_proj", device="cpu", seed=96)
    child.window_words = repack_p32_planar_to_window(child.trellis, bits=child.bits)
    child.window_only = True
    child.trellis = None
    fallback = child._prepare_planar_fallback()
    assert fallback.shape == child.window_words.shape
    assert child._qvq_planar_fallback_cache is not None
    cloned = copy.deepcopy(child)
    assert cloned._qvq_planar_fallback_cache is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_h200_window_prepare_releases_planar_source_in_inference():
    device = _h200_device()
    if device is None:
        pytest.skip("H200 required")
    child = _child("q_proj", device=device, seed=97)
    assert not child.window_only and child.trellis is not None
    with torch.inference_mode():
        window = child._prepare_hopper_p32_window(device)
    assert child.window_only
    assert child.trellis is None
    assert child.window_words is window
    # A second preparation reuses the inference tensor by identity without
    # trying to read its unavailable version counter.
    assert child._prepare_hopper_p32_window(device) is window


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_h200_grouped_window_only_payload_is_exact_and_graph_safe():
    device = _h200_device()
    if device is None:
        pytest.skip("H200 required")
    children = tuple(
        _child(name, device=device, seed=101 + index)
        for index, name in enumerate(("q_proj", "k_proj"))
    )
    for child in children[1:]:
        child.SU.copy_(children[0].SU)
    x = torch.randn(16, 256, device=device, dtype=torch.float16) * 0.01
    with torch.inference_mode():
        references = tuple(child(x).clone() for child in children)
        for child in children:
            # Direct inference now transfers planar ownership to the lossless
            # window cache before this explicit window-only check.
            if not child.window_only:
                child.window_words = repack_p32_planar_to_window(
                    child.trellis, bits=child.bits
                )
                child.window_only = True
                child.trellis = None
                child.post_init()
        parent = nn.Module()
        parent.gate_proj, parent.up_proj = children
        assert install_qvq_hopper_groups(parent, qkv=False, gate_up=True)["gate_up"] == 1
        outputs = (parent.gate_proj(x), parent.up_proj(x))
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = (parent.gate_proj(x), parent.up_proj(x))
        for _ in range(3):
            graph.replay()
            for output, reference in zip(captured, references, strict=True):
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
    for output, reference in zip(outputs, references, strict=True):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    runtime = children[0]._gptqmodel_qvq_grouped_runtime
    assert runtime.telemetry.payload_builds == 1
    assert runtime.telemetry.grouped_launches >= 2
    assert children[0].trellis is None and children[1].trellis is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_h200_grouped_explicit_bm_bn_policy_is_graph_safe():
    """Grouped launches must honor one explicit BM/BN geometry for every child."""
    device = _h200_device()
    if device is None:
        pytest.skip("H200 required")
    children = tuple(
        _child(name, device=device, seed=121 + index)
        for index, name in enumerate(("gate_proj", "up_proj"))
    )
    for child in children[1:]:
        child.SU.copy_(children[0].SU)
    config = P32WindowConfig(
        algorithm="hopper_m16",
        block_m=64,
        block_n=64,
        block_k=256,
        warp_groups=1,
        split_k=1,
        recovery_mode="off",
    )
    x = torch.randn(64, 256, device=device, dtype=torch.float16) * 0.01
    with torch.inference_mode():
        references = tuple(child(x).clone() for child in children)
        for child in children:
            prepare_rank8(child, config)
        parent = nn.Module()
        parent.gate_proj, parent.up_proj = children
        assert install_qvq_hopper_groups(parent, qkv=False, gate_up=True)["gate_up"] == 1
        outputs = (parent.gate_proj(x), parent.up_proj(x))
        for output, reference in zip(outputs, references, strict=True):
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = (parent.gate_proj(x), parent.up_proj(x))
        for _ in range(3):
            graph.replay()
            for output, reference in zip(captured, outputs, strict=True):
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
    runtime = children[0]._gptqmodel_qvq_grouped_runtime
    assert runtime.telemetry.grouped_launches >= 2
    assert runtime.telemetry.plain_fallbacks == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_h200_window_only_rank8_capture_matches_planar_reference():
    device = _h200_device()
    if device is None:
        pytest.skip("H200 required")
    from test_qvq_window_recovery import _kernel_rank8

    child = _child("q_proj", device=device, seed=111)
    _kernel_rank8(child)
    config = P32WindowConfig(recovery_mode="on")
    x = torch.randn(16, 256, device=device, dtype=torch.float16) * 0.01
    with torch.inference_mode():
        prepare_rank8(child, config)
        expected = child(x).clone()
        if not child.window_only:
            child.window_words = repack_p32_planar_to_window(
                child.trellis, bits=child.bits
            )
            child.window_only = True
            child.trellis = None
            child.post_init()
        prepare_rank8(child, config)
        candidates = window_kernel_candidates(child, m=16)
        assert candidates[0].algorithm == "hopper_m16"
        assert all(
            candidate.algorithm != "production_window" for candidate in candidates
        )
        actual = child(x)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = child(x)
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert child.trellis is None


def test_quantization_uses_the_same_role_groups_as_runtime_fusion():
    tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1:o"),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
        },
    ]
    merged = _qvq_quantization_group_candidates(
        tree,
        {"qkv": (("in_proj_qkv", "in_proj_z"),)},
    )
    assert merged == {
        "qkv": (
            ("in_proj_qkv", "in_proj_z"),
            ("q_proj", "k_proj", "v_proj"),
        ),
        "gate_up": (("gate_proj", "up_proj"),),
    }


@pytest.mark.parametrize(
    ("transition_bits", "expected"),
    (
        (4, ((10, 20, 20), (10, 20), (5, 5))),
        (5, ((10, 20, 20), (10, 20), (5, 5))),
        (6, ((10, 20, 20), (4, 20), (5, 5))),
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
    assert (
        qvq_h100_grouped_ordered_split_counts(
            device_name="NVIDIA H200",
            compute_capability=(9, 0),
            in_features=5120,
            out_features=shapes[0],
            transition_bits=transition_bits,
        )
        is None
    )


@pytest.mark.parametrize(
    ("transition_bits", "expected"),
    (
        (4, ((1, 1, 1), (1, 1))),
        (5, ((5, 10, 10), (1, 1))),
        (6, ((1, 1, 1), (2, 2))),
        (7, ((2, 10, 10), (2, 2))),
    ),
)
def test_qwen38_flash_next_h100_grouped_schedules_preserve_child_reductions(
    transition_bits, expected
):
    shapes = ((12288, 512, 512), (10240, 6144))

    actual = tuple(
        qvq_h100_grouped_ordered_split_counts(
            device_name="NVIDIA H100",
            compute_capability=(9, 0),
            in_features=2560,
            out_features=shape,
            transition_bits=transition_bits,
        )
        for shape in shapes
    )
    assert actual == expected
    assert (
        qvq_h100_grouped_ordered_split_counts(
            device_name="NVIDIA H200",
            compute_capability=(9, 0),
            in_features=2560,
            out_features=shapes[0],
            transition_bits=transition_bits,
        )
        is None
    )


def test_h100_large_m_chunk_candidates_are_bounded_and_configurable(monkeypatch):
    monkeypatch.delenv("QVQ_HOPPER_LARGE_M_CHUNK_CANDIDATES", raising=False)
    assert QVQHopperGroupedRuntime._large_m_chunk_candidates() == (
        512,
        1024,
        2048,
        4096,
    )
    monkeypatch.setenv(
        "QVQ_HOPPER_LARGE_M_CHUNK_CANDIDATES", "2048,invalid,4096,2048,8192"
    )
    assert QVQHopperGroupedRuntime._large_m_chunk_candidates() == (2048, 4096)
    monkeypatch.setenv("QVQ_HOPPER_LARGE_M_CHUNK_CANDIDATES", "invalid,8192")
    assert QVQHopperGroupedRuntime._large_m_chunk_candidates() == (
        512,
        1024,
        2048,
        4096,
    )


def test_h100_fp8_prefill_conversion_saturates_and_replays_cuda_graph():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    values = torch.tensor(
        (-65504, -57344, -1, 0, 1, 57344, 65504),
        device=device,
        dtype=torch.float16,
    )
    expected = values.clamp(-57344, 57344).to(torch.float8_e5m2)
    with torch.inference_mode():
        eager = qvq_fp16_to_fp8_e5m2_clamped(values)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = qvq_fp16_to_fp8_e5m2_clamped(values)
        graph.replay()
        torch.cuda.synchronize(device)
    assert torch.equal(eager.float(), expected.float())
    assert torch.equal(captured.float(), expected.float())


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


def test_r0_accepts_shared_a8_contract_and_rejects_mixed_activation_state():
    shared = torch.ones(256)
    accepted = _Attention(
        tuple(
            _child(
                name,
                su=shared,
                alt_id=index + 1,
                seed=90 + index,
                activation=True,
            )
            for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
        )
    )
    assert install_qvq_hopper_groups(accepted, gate_up=False) == {"qkv": 1}
    assert uninstall_qvq_hopper_groups(accepted) == 1

    rejected_children = tuple(
        _child(
            name,
            su=shared,
            alt_id=index + 1,
            seed=95 + index,
            activation=True,
        )
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    rejected_children[1].activation = None
    assert install_qvq_hopper_groups(
        _Attention(rejected_children), gate_up=False
    ) == {"qkv": 0}


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("logical_m", (1, 2, 4, 8, 16))
def test_h200_grouped_a8_executes_true_fp8_children_and_matches_independent_outputs(
    logical_m, dtype
):
    device = _h200_device()
    if device is None:
        pytest.skip("requires the exclusive H200 validation device")
    shared = torch.randn(256, device=device)
    children = tuple(
        _child(
            name,
            su=shared,
            alt_id=index + 1,
            seed=105 + index,
            device=device,
            activation=True,
        )
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    attention = _Attention(children)
    x = (torch.randn((logical_m, 256), device=device) * 0.02).to(dtype)
    if dtype == torch.bfloat16:
        # Finite BF16 values above FP16's range must survive the shared
        # SU/Hadamard transform before dynamic E4M3 row scaling.
        x[0].fill_(65536.0)

    with torch.inference_mode():
        expected = tuple(child(x).clone() for child in children)
    assert all(torch.isfinite(output).all() for output in expected)
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}
    with torch.inference_mode():
        actual = tuple(
            getattr(attention, name)(x)
            for name in ("q_proj", "k_proj", "v_proj")
        )

    assert all(
        torch.equal(output, reference)
        for output, reference in zip(actual, expected, strict=True)
    )
    telemetry = qvq_grouped_runtime_telemetry(attention)[0]
    assert telemetry["grouped_launches"] == 1
    assert telemetry["grouped_a8_launches"] == 1
    assert telemetry["shared_fp8_quantizations"] == 1
    assert telemetry["fp8_independent_child_launches"] == 3
    assert telemetry["payload_builds"] == 0
    assert telemetry["grouped_window_bytes"] == 0
    for child in children:
        fp8_telemetry = child.qvq_fp8_kernel_telemetry()
        assert fp8_telemetry["executed"] == 2
        assert fp8_telemetry["fallback"] == 0


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


def test_grouped_hopper_policy_accepts_split_tuple_and_rejects_unimplemented_geometry():
    shared = torch.randn(256)
    children = tuple(
        _child(name, su=shared, seed=71 + index)
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    attention = _Attention(children)
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}
    runtime = attention.q_proj._gptqmodel_qvq_grouped_runtime
    base = P32WindowConfig()
    for child in children:
        child._p32_window_config = base
    direct = replace(
        base,
        algorithm="hopper_direct_decode_mma",
        block_m=64,
        block_n=64,
        warp_groups=1,
    )
    for child in children:
        child._p32_window_config = direct
    reason = runtime._runtime_eligible(torch.randn(1, 256))
    assert reason == "grouped Hopper tuning requires hopper_m16 for every child; direct BM/BN geometry is single-child only"

    grouped_geometry = replace(
        base,
        algorithm="hopper_m16",
        block_m=64,
        block_n=128,
        warp_groups=2,
    )
    for child in children:
        child._p32_window_config = grouped_geometry
    reason = runtime._runtime_eligible(torch.randn(64, 256))
    assert reason == "grouped Hopper BN128 requires a single-segment specialization"

    split_policy = replace(direct, algorithm="hopper_m16", block_m=0, block_n=0, warp_groups=0)
    for child in children:
        child._p32_window_config = split_policy
    reason = runtime._runtime_eligible(torch.randn(1, 256))
    assert reason == "grouped Hopper requires FP16 or BF16 CUDA activations"

    unsupported_projection = replace(
        split_policy, recovery_projection="tensor_core"
    )
    for child in children:
        child._p32_rank8_enabled = True
        child._p32_window_config = unsupported_projection
    reason = runtime._runtime_eligible(torch.randn(1, 256))
    assert reason == (
        "grouped rank8 projection supports separate_reference, "
        "concurrent_reference, input_fused or project_output_fused only"
    )
    concurrent_policy = replace(split_policy, recovery_projection="concurrent_reference")
    for child in children:
        child._p32_window_config = concurrent_policy
    reason = runtime._runtime_eligible(torch.randn(1, 256, dtype=torch.bfloat16))
    # The concurrent producer has an FP16 projection contract; BF16 must not
    # silently degrade to the separate child projection.
    assert reason == "grouped concurrent rank8 projection requires FP16 activations"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("projection", "recovery_kernel"),
    (
        ("concurrent_reference", "separate_reference"),
        ("input_fused", "separate_reference"),
        ("project_output_fused", "fused_epilogue"),
    ),
)
def test_grouped_rank8_reuses_transform_and_replays_graph(
    projection, recovery_kernel
):
    """Grouped rank8 producer policies must join the decode graph safely."""

    device = _h200_device() or _h100_device()
    if device is None:
        pytest.skip("requires an H100 or H200 SM90 validation device")
    from test_qvq_window_recovery import _kernel_rank8
    from gptqmodel.quantization.qvq_rank8 import P32WindowConfig, prepare_rank8

    torch.backends.cuda.matmul.allow_tf32 = False
    shared = torch.ones(256, device=device)
    children = tuple(
        _child(
            name,
            su=shared,
            alt_id=index + 1,
            seed=20261040 + index,
            device=device,
        )
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    config = P32WindowConfig(
        recovery_mode="on",
        recovery_projection=projection,
        recovery_kernel=recovery_kernel,
    )
    for child in children:
        _kernel_rank8(child)
        prepare_rank8(child, config)
    attention = _Attention(children)
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}
    x = torch.randn((64, 256), device=device, dtype=torch.float16) * 0.02

    with torch.inference_mode():
        eager = tuple(
            getattr(attention, name)(x).clone()
            for name in ("q_proj", "k_proj", "v_proj")
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tuple(
                getattr(attention, name)(x)
                for name in ("q_proj", "k_proj", "v_proj")
            )
        graph.replay()
        torch.cuda.synchronize(device)

    for actual, replayed in zip(eager, captured, strict=True):
        assert torch.equal(replayed, actual)
        assert torch.isfinite(actual).all()
    if projection == "concurrent_reference":
        assert all(
            len(child._qvq_rank8_concurrent_cache) >= 1 for child in children
        )


def test_grouped_policy_rejects_m_outside_prepared_range_before_device_dispatch():
    """A shape-tuned grouped policy cannot silently run at another M."""
    shared = torch.randn(256)
    children = tuple(
        _child(name, su=shared, seed=141 + index)
        for index, name in enumerate(("gate_proj", "up_proj"))
    )
    mlp = _MLP(children)
    assert install_qvq_hopper_groups(mlp, qkv=False, gate_up=True) == {"gate_up": 1}
    policy = replace(
        P32WindowConfig(algorithm="hopper_m16", recovery_mode="off"),
        min_m=64,
        max_m=64,
    )
    for child in children:
        child._p32_window_config = policy
    runtime = children[0]._gptqmodel_qvq_grouped_runtime
    reason = runtime._runtime_eligible(torch.randn(32, 256))
    assert reason == "grouped child 0 M=32 is outside its prepared policy range [64, 64]"

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
        qvq_grouped_p32_candidates: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
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
        (child.window_words if child.window_only else child.trellis).numel()
        * (child.window_words if child.window_only else child.trellis).element_size()
        for child in children
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


@pytest.mark.parametrize("logical_m", (1, 32, 64, 128, 256, 512, 1024, 2048, 4096))
def test_warmed_production_group_is_cuda_graph_capturable(logical_m):
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
    static_input = (
        torch.randn((logical_m, 256), device=device, dtype=torch.float16) * 0.02
    )

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
    static_input = torch.randn((1, 5120), device=device, dtype=torch.float16) * 0.02
    with torch.inference_mode():
        plain = tuple(child(static_input).clone() for child in children)
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}
    with torch.inference_mode():
        eager = tuple(getattr(attention, name)(static_input) for name in names)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tuple(getattr(attention, name)(static_input) for name in names)
        graph.replay()
        torch.cuda.synchronize(device)

    for actual, replayed, reference in zip(eager, captured, plain, strict=True):
        assert torch.equal(replayed, actual)
        torch.testing.assert_close(actual, reference, rtol=0, atol=2e-3)
    telemetry = qvq_grouped_runtime_telemetry(attention)[0]
    assert telemetry["active_split_counts"] == (10, 20, 20)
    assert telemetry["grouped_launches"] == 2
    assert telemetry["h100_qwen_composite_input_launches"] == 0
    assert telemetry["plain_fallbacks"] == 0


@pytest.mark.parametrize(
    ("bits", "expected_splits"),
    ((2.0, (10, 20)), (2.5, (10, 20)), (3.0, (4, 20))),
)
def test_qwen38_linear_input_group_uses_fixed_grid_in_cuda_graph(bits, expected_splits):
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")

    names = ("in_proj_qkv", "in_proj_z")
    widths = (10240, 6144)
    shared = torch.ones(5120, device=device)
    children = tuple(
        _child(
            name,
            in_features=5120,
            out_features=width,
            bits=bits,
            su=shared,
            alt_id=index + 1,
            seed=20260970 + index,
            device=device,
        )
        for index, (name, width) in enumerate(zip(names, widths, strict=True))
    )
    parent = nn.Module()
    for name, child in zip(names, children, strict=True):
        setattr(parent, name, child)
    with torch.no_grad():
        for child in children:
            child.SV.fill_(0.002)
            child.bias.zero_()
    static_input = torch.randn((8, 5120), device=device, dtype=torch.float16) * 0.02
    with torch.inference_mode():
        plain = tuple(child(static_input).clone() for child in children)
    assert install_qvq_hopper_groups(
        parent,
        qkv_candidates=(names,),
        qkv=True,
        gate_up=False,
    ) == {"qkv": 1}
    with torch.inference_mode():
        eager = tuple(getattr(parent, name)(static_input) for name in names)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tuple(getattr(parent, name)(static_input) for name in names)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize(device)

    for actual, replayed, reference in zip(eager, captured, plain, strict=True):
        assert torch.equal(replayed, actual)
        torch.testing.assert_close(actual, reference, rtol=0, atol=2e-3)
    telemetry = qvq_grouped_runtime_telemetry(parent)[0]
    assert telemetry["active_split_counts"] == expected_splits
    assert telemetry["grouped_launches"] == 2
    assert telemetry["h100_qwen_fixed_linear_grid_launches"] == 2
    assert telemetry["h100_qwen_linear_decode_prefetch_launches"] == (
        0 if bits == 2.5 else 2
    )
    assert telemetry["h100_qwen_linear_composite_recovery_launches"] == 4
    assert telemetry["h100_qwen_linear_multiblock_recovery_launches"] == 4
    assert telemetry["plain_fallbacks"] == 0


@pytest.mark.parametrize("bits", (2.0, 2.5, 3.0))
def test_qwen38_folded_mlp_is_fused_and_cuda_graph_safe(bits):
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
                bits=bits,
                su=shared,
                seed=20260920,
                device=device,
                output_hadamard=False,
            )
            self.up_proj = _child(
                "up_proj",
                in_features=5120,
                out_features=17408,
                bits=bits,
                su=shared,
                seed=20260921,
                device=device,
                output_hadamard=False,
            )
            self.down_proj = _child(
                "down_proj",
                in_features=17408,
                out_features=5120,
                bits=bits,
                seed=20260922,
                device=device,
                input_hadamard=False,
            )
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    mlp = QwenMLP().eval()
    with torch.no_grad():
        for child in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            child.SV.fill_(0.002)
            child.bias.zero_()
    static_input = torch.randn((8, 5120), device=device, dtype=torch.float16) * 0.02
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

    if bits == 3.0:
        assert torch.equal(captured, eager)
    else:
        torch.testing.assert_close(captured, eager, rtol=0, atol=2e-3)
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.isfinite(captured).all()
        if bits == 3.0:
            assert torch.equal(captured, eager)
        else:
            torch.testing.assert_close(captured, eager, rtol=0, atol=2e-3)
    torch.testing.assert_close(eager, plain, rtol=0, atol=2e-3)
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["active_split_counts"] == (5, 5)
    assert telemetry["fused_mlp_launches"] == 2
    assert telemetry["h100_folded_qwen_mlp_launches"] == 2
    assert telemetry["h100_folded_qwen_fused_precondition_launches"] == 2
    assert telemetry["h100_folded_qwen_fused_ordered_reduction_launches"] == 2
    assert telemetry["h100_qwen_w3_ordered_decode_prefetch_launches"] == (
        2 if bits == 3.0 else 0
    )
    assert telemetry["h100_qwen_ordered_decode_prefetch_launches"] == 2
    assert telemetry["h100_qwen_w3_down_decode_prefetch_launches"] == (
        2 if bits == 3.0 else 0
    )
    assert telemetry["h100_qwen_down_decode_prefetch_launches"] == 2
    assert telemetry["h100_qwen_fixed_ordered_grid_launches"] == 2
    assert telemetry["h100_qwen_composite_down_recovery_launches"] == 2
    assert telemetry["h100_qwen_ordered_composite_down_recovery_launches"] == (
        2 if bits == 3.0 else 0
    )
    assert telemetry["h100_qwen_composite_input_launches"] == 2
    assert telemetry["plain_fallbacks"] == 0
    assert telemetry["fused_mlp_fallbacks"] == 0


@pytest.mark.parametrize("recovery_kernel", ["separate_reference", "fused_epilogue"])
def test_rank8_grouped_mlp_includes_down_correction_and_graph_replay(
    recovery_kernel,
):
    """Gate/up and down rank-8 corrections share one captured MLP path."""

    device = _h100_device() or _h200_device()
    if device is None:
        pytest.skip("requires an H100 or H200 SM90 validation device")

    from test_qvq_window_recovery import _kernel_rank8

    from gptqmodel.quantization.qvq_rank8 import P32WindowConfig, prepare_rank8

    class Rank8MLP(nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.ones(256, device=device)
            self.gate_proj = _child(
                "gate_proj",
                in_features=256,
                out_features=256,
                bits=3,
                su=shared,
                seed=20260930,
                device=device,
                output_hadamard=False,
            )
            self.up_proj = _child(
                "up_proj",
                in_features=256,
                out_features=256,
                bits=3,
                su=shared,
                seed=20260931,
                device=device,
                output_hadamard=False,
            )
            self.down_proj = _child(
                "down_proj",
                in_features=256,
                out_features=256,
                bits=3,
                su=torch.ones(256, device=device),
                seed=20260932,
                device=device,
                input_hadamard=False,
            )
            with torch.no_grad():
                for child in (self.gate_proj, self.up_proj, self.down_proj):
                    child.SV.fill_(0.002)
                    child.bias.zero_()
            for child in (self.gate_proj, self.up_proj, self.down_proj):
                _kernel_rank8(child)
                prepare_rank8(
                    child,
                    P32WindowConfig(
                        recovery_mode="on", recovery_kernel=recovery_kernel
                    ),
                )
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(
                self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            )

    mlp = Rank8MLP().eval()
    static_input = torch.randn(
        (2, 256), device=device, dtype=torch.float16
    ) * 0.02
    with torch.inference_mode():
        reference = mlp(static_input).clone()
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}
    assert hasattr(mlp, "_gptqmodel_qvq_fused_mlp_runtime")

    with torch.inference_mode():
        eager = mlp(static_input).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mlp(static_input)
        graph.replay()
        torch.cuda.synchronize(device)

    torch.testing.assert_close(eager, reference, rtol=0, atol=2e-3)
    assert torch.equal(captured, eager)
    for _ in range(4):
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(captured, eager)
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["fused_mlp_launches"] == 2
    assert telemetry["fused_mlp_fallbacks"] == 0
    assert telemetry["independent_recovery_children"] >= 4


@pytest.mark.parametrize(
    "projection", ("input_fused", "concurrent_reference", "project_output_fused")
)
def test_fused_mlp_rejects_unsupported_down_rank8_projection(projection):
    """Unsupported down policies fail before grouped dispatch."""

    device = _h100_device() or _h200_device()
    if device is None:
        pytest.skip("requires an H100 or H200 SM90 validation device")

    from test_qvq_window_recovery import _kernel_rank8
    from gptqmodel.quantization.qvq_rank8 import P32WindowConfig, prepare_rank8

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.ones(256, device=device)
            self.gate_proj = _child(
                "gate_proj", in_features=256, out_features=256, su=shared,
                seed=20261120, device=device, output_hadamard=False,
            )
            self.up_proj = _child(
                "up_proj", in_features=256, out_features=256, su=shared,
                seed=20261121, device=device, output_hadamard=False,
            )
            self.down_proj = _child(
                "down_proj", in_features=256, out_features=256,
                seed=20261122, device=device, input_hadamard=False,
            )
            _kernel_rank8(self.down_proj)
            prepare_rank8(
                self.down_proj,
                P32WindowConfig(recovery_mode="on", recovery_projection=projection),
            )
            self.act_fn = nn.SiLU()

    mlp = MLP().eval()
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}
    x = torch.randn((2, 256), device=device, dtype=torch.float16)
    runtime = mlp.gate_proj._gptqmodel_qvq_grouped_runtime
    runtime._configure_mlp_fusion(mlp, mlp.down_proj, mlp.act_fn)
    reason = runtime._mlp_rejection(x)
    assert reason == (
        f"fused MLP down rank8 projection {projection} is unsupported; use "
        "separate_reference or tensor_core"
    )


@pytest.mark.parametrize("bits, expected_fused_tiles", [(2.0, 0), (3.0, 2)])
def test_qwen38_m32_rate_specific_path_is_cuda_graph_safe(
    bits, expected_fused_tiles
):
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")

    class QwenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.ones(5120, device=device)
            self.gate_proj = _child(
                "gate_proj", in_features=5120, out_features=17408, bits=bits,
                su=shared, seed=20261020, device=device, output_hadamard=False,
            )
            self.up_proj = _child(
                "up_proj", in_features=5120, out_features=17408, bits=bits,
                su=shared, seed=20261021, device=device, output_hadamard=False,
            )
            self.down_proj = _child(
                "down_proj", in_features=17408, out_features=5120, bits=bits,
                seed=20261022, device=device, input_hadamard=False,
            )
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    mlp = QwenMLP().eval()
    with torch.no_grad():
        for child in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            child.SV.fill_(0.002)
            child.bias.zero_()
    x = torch.randn((32, 5120), device=device, dtype=torch.float16) * 0.02
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}
    with torch.inference_mode():
        eager = mlp(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mlp(x)
        graph.replay()
        torch.cuda.synchronize(device)

    torch.testing.assert_close(eager, captured, rtol=0, atol=0)
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["h100_qwen_m32_fused_tiles"] == expected_fused_tiles
    assert telemetry["h100_qwen_large_m_direct_down_launches"] == (
        0 if expected_fused_tiles else 2
    )
    assert telemetry["fused_mlp_fallbacks"] == 0


def test_qwen38_m128_gate_up_uses_unsplit_graph_safe_payload():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")

    class QwenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.ones(5120, device=device)
            self.gate_proj = _child(
                "gate_proj", in_features=5120, out_features=17408, bits=3,
                su=shared, seed=20261023, device=device, output_hadamard=False,
            )
            self.up_proj = _child(
                "up_proj", in_features=5120, out_features=17408, bits=3,
                su=shared, seed=20261024, device=device, output_hadamard=False,
            )
            self.down_proj = _child(
                "down_proj", in_features=17408, out_features=5120, bits=3,
                seed=20261025, device=device, input_hadamard=False,
            )
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    mlp = QwenMLP().eval()
    with torch.no_grad():
        for child in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            child.SV.fill_(0.002)
            child.bias.zero_()
    x = torch.randn((128, 5120), device=device, dtype=torch.float16) * 0.02
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}
    with torch.inference_mode():
        eager = mlp(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mlp(x)
        graph.replay()
        torch.cuda.synchronize(device)

    torch.testing.assert_close(eager, captured, rtol=0, atol=0)
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["active_split_counts"] == (5, 5)
    assert telemetry["h100_qwen_large_m_unsplit_gate_up_launches"] == 2
    assert telemetry["h100_qwen_large_m_direct_down_launches"] == 2
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


def test_h100_large_m_wide_gate_up_runtime_is_exact_graph_safe_and_observable():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    shared = torch.ones(2048, device=device)
    children = tuple(
        _child(
            name,
            in_features=2048,
            out_features=8192,
            bits=3,
            su=shared,
            alt_id=alt_id,
            seed=260 + index,
            device=device,
        )
        for index, (name, alt_id) in enumerate(
            (("gate_proj", 1), ("up_proj", 3))
        )
    )
    mlp = _MLP(children)
    static_input = torch.randn((128, 2048), device=device, dtype=torch.float16) * 0.02
    with torch.inference_mode():
        expected = (mlp.gate_proj(static_input), mlp.up_proj(static_input))
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}

    with torch.inference_mode():
        eager = (mlp.gate_proj(static_input), mlp.up_proj(static_input))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = (mlp.gate_proj(static_input), mlp.up_proj(static_input))
        graph.replay()
        torch.cuda.synchronize(device)

    assert all(
        torch.equal(output, reference)
        for output, reference in zip(eager, expected, strict=True)
    )
    assert all(
        torch.equal(output, reference)
        for output, reference in zip(captured, expected, strict=True)
    )
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["h100_wide_reuse_gate_up_launches"] == 2
    assert telemetry["h100_reuse8_gate_up_launches"] == 2
    assert telemetry["plain_fallbacks"] == 0


@pytest.mark.parametrize("logical_m", (512, 1024, 2048, 4096))
def test_h100_reuse11_runtime_is_exact_graph_safe_and_observable(logical_m):
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    shared = torch.ones(2048, device=device)
    children = tuple(
        _child(
            name,
            in_features=2048,
            out_features=8192,
            bits=3,
            su=shared,
            alt_id=alt_id,
            seed=280 + index,
            device=device,
        )
        for index, (name, alt_id) in enumerate(
            (("gate_proj", 1), ("up_proj", 3))
        )
    )
    mlp = _MLP(children)
    static_input = (
        torch.randn((logical_m, 2048), device=device, dtype=torch.float16) * 0.02
    )
    with torch.inference_mode():
        expected = (mlp.gate_proj(static_input), mlp.up_proj(static_input))
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}

    with torch.inference_mode():
        eager = (mlp.gate_proj(static_input), mlp.up_proj(static_input))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = (mlp.gate_proj(static_input), mlp.up_proj(static_input))
        graph.replay()
        torch.cuda.synchronize(device)

    assert all(
        torch.equal(output, reference)
        for output, reference in zip(eager, expected, strict=True)
    )
    assert all(
        torch.equal(output, reference)
        for output, reference in zip(captured, expected, strict=True)
    )
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["h100_wide_reuse_gate_up_launches"] == 2
    assert telemetry["h100_reuse11_gate_up_launches"] == 2
    assert telemetry["h100_reuse8_gate_up_launches"] == 0
    assert telemetry["plain_fallbacks"] == 0


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
    large_m_x = torch.randn((17, 256), device=device, dtype=torch.float16) * 0.02
    with torch.inference_mode():
        expected = mlp(x)
        expected_large_m = mlp(large_m_x)
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}
    assert hasattr(mlp, "_gptqmodel_qvq_fused_mlp_runtime")
    with torch.inference_mode():
        actual = mlp(x)
        actual_large_m = mlp(large_m_x)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_large_m, expected_large_m)
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["fused_mlp_launches"] == 2
    assert telemetry["fused_mlp_fallbacks"] == 0
    assert uninstall_qvq_hopper_groups(mlp) == 1
    assert not hasattr(mlp, "_gptqmodel_qvq_fused_mlp_runtime")
    with torch.inference_mode():
        assert torch.equal(mlp(x), expected)


@pytest.mark.parametrize(
    "logical_rows", (32, 64, 128, 256, 512, 1024, 2048, 4096, 4097)
)
def test_large_m_fused_mlp_uses_native_row_reuse_and_replays_cuda_graph(
    logical_rows,
):
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")

    class LlamaLikeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            shared_input = torch.ones(256, device=device)
            self.gate_proj = _child(
                "gate_proj", su=shared_input, alt_id=1, seed=280, device=device
            )
            self.up_proj = _child(
                "up_proj", su=shared_input, alt_id=2, seed=281, device=device
            )
            self.down_proj = _child("down_proj", alt_id=3, seed=282, device=device)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    mlp = LlamaLikeMLP().eval()
    static_input = (
        torch.randn((logical_rows, 256), device=device, dtype=torch.float16) * 0.02
    )
    with torch.inference_mode():
        plain = mlp(static_input).clone() if logical_rows <= 4096 else None
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}

    with torch.inference_mode():
        graph = torch.cuda.CUDAGraph()
        if logical_rows > 4096:
            # Warm the canonical payload and child kernel caches while leaving
            # the >4096 chunk-plan cache cold. Capture cannot time candidates,
            # so it must record the fixed 4096-row plan without caching it;
            # the next eager call remains free to autotune the measured target.
            mlp(static_input[:16])
            with torch.cuda.graph(graph):
                captured = mlp(static_input)
            eager = mlp(static_input)
        else:
            eager = mlp(static_input)
            with torch.cuda.graph(graph):
                captured = mlp(static_input)
        graph.replay()
        torch.cuda.synchronize(device)

    if logical_rows <= 4096:
        assert torch.equal(eager, plain)
    assert torch.equal(captured, eager)
    if logical_rows > 4096:
        runtime = mlp._gptqmodel_qvq_fused_mlp_runtime
        with torch.inference_mode():
            for chunk_rows in (512, 1024, 2048, 4096):
                assert torch.equal(
                    runtime._execute_mlp_chunked(static_input, chunk_rows), eager
                )
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, eager)
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["fused_mlp_launches"] == (3 if logical_rows > 4096 else 2)
    assert telemetry["fused_mlp_fallbacks"] == 0
    assert telemetry["plain_fallbacks"] == 0
    if logical_rows > 4096:
        assert telemetry["h100_large_m_chunk_autotunes"] == 1
        assert telemetry["h100_large_m_chunked_mlp_launches"] == 2
        assert telemetry["h100_large_m_chunk_rows"] in (512, 1024, 2048, 4096)


def test_h100_grouped_projection_above_4096_autotunes_and_replays_cuda_graph():
    """Generic grouped QKV must multiplex rows without changing child output."""

    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    shared = torch.ones(256, device=device)
    children = tuple(
        _child(
            name,
            su=shared,
            alt_id=index + 1,
            seed=290 + index,
            device=device,
        )
        for index, name in enumerate(("q_proj", "k_proj", "v_proj"))
    )
    children[2].output_hadamard = False
    attention = _Attention(children)
    static_input = (
        torch.randn((4097, 256), device=device, dtype=torch.float16) * 0.02
    )
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}
    runtime = attention.q_proj._gptqmodel_qvq_grouped_runtime

    def execute_group():
        return tuple(
            getattr(attention, name)(static_input)
            for name in ("q_proj", "k_proj", "v_proj")
        )

    with torch.inference_mode():
        # Warm canonical payload/kernel state on a normal row tile. Keep the
        # >4096 plan cold so capture records the conservative 4096-row path
        # and the following eager call remains free to measure all targets.
        warm = static_input[:16]
        tuple(
            getattr(attention, name)(warm)
            for name in ("q_proj", "k_proj", "v_proj")
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = execute_group()
        eager = execute_group()
        graph.replay()
        torch.cuda.synchronize(device)

        assert all(
            torch.equal(output, reference)
            for output, reference in zip(captured, eager, strict=True)
        )
        for chunk_rows in (512, 1024, 2048, 4096):
            candidate = runtime._execute_group_chunked(
                static_input, chunk_rows, recover=True
            )
            assert all(
                torch.equal(output, reference)
                for output, reference in zip(candidate, eager, strict=True)
            )
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)

    assert all(
        torch.equal(output, reference)
        for output, reference in zip(captured, eager, strict=True)
    )
    telemetry = qvq_grouped_runtime_telemetry(attention)[0]
    assert telemetry["h100_large_m_chunk_autotunes"] == 1
    assert telemetry["h100_large_m_chunked_group_launches"] == 2
    assert telemetry["h100_large_m_group_chunk_rows"] in (512, 1024, 2048, 4096)
    assert telemetry["plain_fallbacks"] == 0


def test_h100_m8192_qkv_uses_folded_fp8_prefill_and_replays_cuda_graph():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    from gptqmodel.nn_modules.qlinear.qvq import qvq_dense_oracle_forward

    shared = torch.ones(2048, device=device)
    widths = (2048, 512, 512)
    children = tuple(
        _child(
            name,
            in_features=2048,
            out_features=width,
            bits=3,
            su=shared,
            alt_id=index + 1,
            seed=300 + index,
            device=device,
        )
        for index, (name, width) in enumerate(
            zip(("q_proj", "k_proj", "v_proj"), widths, strict=True)
        )
    )
    children[2].output_hadamard = False
    for child in children:
        child.bias = None
        child.SV.fill_(0.002)
        child._dtype_cache_clear()
    attention = _Attention(children)
    static_input = (
        torch.randn((8192, 2048), device=device, dtype=torch.float16) * 0.02
    )
    with torch.inference_mode():
        expected = tuple(
            qvq_dense_oracle_forward(child, static_input, device=device)
            for child in children
        )
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}

    def execute_group():
        return tuple(
            getattr(attention, name)(static_input)
            for name in ("q_proj", "k_proj", "v_proj")
        )

    with torch.inference_mode():
        eager = execute_group()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = execute_group()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)

    for actual, reference in zip(eager, expected, strict=True):
        torch.testing.assert_close(actual.float(), reference, rtol=0, atol=2e-3)
    assert all(
        torch.equal(actual, reference)
        for actual, reference in zip(captured, eager, strict=True)
    )
    telemetry = qvq_grouped_runtime_telemetry(attention)[0]
    assert telemetry["h100_fp8_prefill_launches"] == 2
    assert 6 * 1024 * 1024 < telemetry["h100_fp8_prefill_bytes"] < 7 * 1024 * 1024
    assert telemetry["h100_large_m_chunked_group_launches"] == 0
    assert telemetry["plain_fallbacks"] == 0


def test_h100_m512_mlp_uses_versioned_fp8_prefill_and_replays_cuda_graph(
    monkeypatch,
):
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    monkeypatch.setenv("QVQ_HOPPER_FP8_MLP_PREFILL", "1")

    class LlamaLikeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.ones(2048, device=device)
            self.gate_proj = _child(
                "gate_proj", in_features=2048, out_features=8192,
                bits=3, su=shared, alt_id=1, seed=340, device=device,
            )
            self.up_proj = _child(
                "up_proj", in_features=2048, out_features=8192,
                bits=3, su=shared, alt_id=3, seed=341, device=device,
            )
            self.down_proj = _child(
                "down_proj", in_features=8192, out_features=2048,
                bits=3, su=torch.ones(8192, device=device), alt_id=2,
                seed=342, device=device,
            )
            for child in (self.gate_proj, self.up_proj, self.down_proj):
                child.bias = None
                child.SV.fill_(0.002)
                child._dtype_cache_clear()
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    mlp = LlamaLikeMLP().eval()
    static_input = torch.randn((512, 2048), device=device, dtype=torch.float16)
    with torch.inference_mode():
        expected = mlp(static_input)
    assert install_qvq_hopper_groups(mlp, qkv=False) == {"gate_up": 1}

    with torch.inference_mode():
        eager = mlp(static_input)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mlp(static_input)
        graph.replay()
        torch.cuda.synchronize(device)

    torch.testing.assert_close(eager.float(), expected.float(), rtol=0, atol=2e-3)
    relative_l2 = torch.linalg.vector_norm(
        eager.float() - expected.float()
    ) / torch.linalg.vector_norm(expected.float())
    assert relative_l2.item() < 0.08
    assert torch.equal(captured, eager)
    telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
    assert telemetry["h100_fp8_prefill_launches"] == 2
    assert telemetry["h100_fp8_mlp_down_launches"] == 2
    assert telemetry["h100_fp8_mlp_fused_silu_launches"] == 2
    assert telemetry["h100_fp8_prefill_bytes"] == 2048 * 16384 + 16384 * 4 + 4
    assert telemetry["h100_fp8_mlp_down_bytes"] == 8192 * 2048 + 2048 * 4 + 4
    assert telemetry["plain_fallbacks"] == 0

    runtime = mlp._gptqmodel_qvq_fused_mlp_runtime
    old_down_payload = runtime._h100_fp8_mlp_down_payload
    mlp.down_proj.SV.add_(0.125)
    with torch.inference_mode():
        mlp(static_input)
    assert runtime._h100_fp8_mlp_down_payload is not old_down_payload


def test_h100_fp8_mlp_prefill_is_independently_opt_in(monkeypatch):
    monkeypatch.delenv("QVQ_HOPPER_FP8_MLP_PREFILL", raising=False)
    assert not QVQHopperGroupedRuntime._h100_fp8_mlp_prefill_enabled()
    monkeypatch.setenv("QVQ_HOPPER_FP8_MLP_PREFILL", "1")
    assert QVQHopperGroupedRuntime._h100_fp8_mlp_prefill_enabled()


def test_h100_fused_silu_row_quant_matches_staged_reference_and_graph():
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    from gptqmodel.nn_modules.triton_utils.kernels import (
        fused_silu_mul,
        fused_silu_mul_quant_fp8,
    )
    from gptqmodel.utils.qvq_cuda import qvq_cuda_quantize_fp8_per_row

    source = torch.randn((3, 16384), device=device, dtype=torch.float16)
    gate, up = source[:, :8192], source[:, 8192:]
    with torch.inference_mode():
        intermediate = fused_silu_mul(gate, up)
        expected, expected_scale = qvq_cuda_quantize_fp8_per_row(intermediate)
        actual, actual_scale = fused_silu_mul_quant_fp8(gate, up)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_scale = fused_silu_mul_quant_fp8(gate, up)
        graph.replay()
        torch.cuda.synchronize(device)

    assert actual.is_contiguous()
    assert captured.is_contiguous()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_scale, expected_scale)
    assert torch.equal(captured.view(torch.uint8), actual.view(torch.uint8))
    assert torch.equal(captured_scale, actual_scale)


def test_h100_m16384_qkv_on_demand_fp8_is_bounded_and_replays_cuda_graph(
    monkeypatch,
):
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    from gptqmodel.nn_modules.qlinear.qvq import qvq_dense_oracle_forward

    monkeypatch.setenv("QVQ_HOPPER_FP8_PREFILL_ON_DEMAND", "1")
    shared = torch.ones(2048, device=device)
    widths = (2048, 512, 512)
    children = tuple(
        _child(
            name,
            in_features=2048,
            out_features=width,
            bits=3,
            su=shared,
            alt_id=index + 1,
            seed=320 + index,
            device=device,
        )
        for index, (name, width) in enumerate(
            zip(("q_proj", "k_proj", "v_proj"), widths, strict=True)
        )
    )
    children[2].output_hadamard = False
    for child in children:
        child.bias = None
        child.SV.fill_(0.002)
        child._dtype_cache_clear()
    attention = _Attention(children)
    static_input = (
        torch.randn((16384, 2048), device=device, dtype=torch.float16) * 0.02
    )
    with torch.inference_mode():
        expected = tuple(
            qvq_dense_oracle_forward(child, static_input, device=device)
            for child in children
        )
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}

    def execute_group():
        return tuple(
            getattr(attention, name)(static_input)
            for name in ("q_proj", "k_proj", "v_proj")
        )

    with torch.inference_mode():
        eager = execute_group()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = execute_group()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)

    for actual, reference in zip(eager, expected, strict=True):
        torch.testing.assert_close(actual.float(), reference, rtol=0, atol=2e-3)
    assert all(
        torch.equal(actual, reference)
        for actual, reference in zip(captured, eager, strict=True)
    )
    telemetry = qvq_grouped_runtime_telemetry(attention)[0]
    assert telemetry["h100_fp8_ondemand_launches"] == 2
    assert telemetry["h100_fp8_ondemand_retained_bytes"] == 8
    assert telemetry["h100_fp8_ondemand_scratch_bytes"] == 2048 * 3072 * 7
    assert telemetry["h100_fp8_prefill_launches"] == 0
    assert telemetry["h100_large_m_chunked_group_launches"] == 0
    assert telemetry["plain_fallbacks"] == 0


@pytest.mark.parametrize("native", (False, True))
def test_h100_grouped_fp16_prefill_is_bounded_and_replays_cuda_graph(
    monkeypatch, native
):
    device = _h100_device()
    if device is None:
        pytest.skip("requires the exclusive H100 validation device")
    from gptqmodel.nn_modules.qlinear.qvq import qvq_dense_oracle_forward

    monkeypatch.setenv("QVQ_HOPPER_FP8_PREFILL", "0")
    monkeypatch.setenv("QVQ_HOPPER_FP16_PREFILL", "1")
    monkeypatch.setenv("QVQ_HOPPER_FP16_PREFILL_NATIVE", str(int(native)))
    shared = torch.ones(2048, device=device)
    widths = (2048, 512, 512)
    children = tuple(
        _child(
            name,
            in_features=2048,
            out_features=width,
            bits=3,
            su=shared,
            alt_id=index + 1,
            seed=340 + index,
            device=device,
        )
        for index, (name, width) in enumerate(
            zip(("q_proj", "k_proj", "v_proj"), widths, strict=True)
        )
    )
    children[2].output_hadamard = False
    for child in children:
        child.SV.fill_(0.002)
        child.bias = None
        child._dtype_cache_clear()
    attention = _Attention(children)
    static_input = torch.randn(
        (512, 2048), device=device, dtype=torch.float16
    ) * 0.02
    with torch.inference_mode():
        expected = tuple(
            qvq_dense_oracle_forward(child, static_input, device=device)
            for child in children
        )
    assert install_qvq_hopper_groups(attention, gate_up=False) == {"qkv": 1}

    def execute_group():
        return tuple(
            getattr(attention, name)(static_input)
            for name in ("q_proj", "k_proj", "v_proj")
        )

    with torch.inference_mode():
        eager = execute_group()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = execute_group()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)

    for actual, reference in zip(eager, expected, strict=True):
        torch.testing.assert_close(actual.float(), reference, rtol=0, atol=2e-3)
    assert all(
        torch.equal(actual, reference)
        for actual, reference in zip(captured, eager, strict=True)
    )
    telemetry = qvq_grouped_runtime_telemetry(attention)[0]
    assert telemetry["h100_fp16_prefill_launches"] == 2
    assert telemetry["h100_fp16_prefill_temporary_bytes"] == 2048 * 3072 * 2
    assert telemetry["h100_fp16_prefill_native_launches"] == 2 * int(native)
    assert telemetry["h100_fp16_prefill_native_scratch_bytes"] == (
        2048 * 3072 * 12 * int(native)
    )
    assert telemetry["h100_fp8_prefill_launches"] == 0
    assert telemetry["h100_large_m_chunked_group_launches"] == 0
    assert telemetry["plain_fallbacks"] == 0


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
    assert all(entry["h100_direct_padded_input_launches"] >= 3 for entry in telemetry)
    assert all(
        entry["h100_multiblock_input_hadamard_launches"] >= 4 for entry in telemetry
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
