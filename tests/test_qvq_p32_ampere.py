# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Correctness coverage for the exact sm_80 continuous-window P32 kernel."""

import pytest
import torch

from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    reconstruct_p32_window_inner_weight,
    repack_p32_planar_to_window,
)
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    pgc16_levels_for_version,
)
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
from gptqmodel.utils import qvq_ampere_cuda
from gptqmodel.utils.qvq_ampere_cuda import (
    _auto_split_count,
    _autotune_candidates,
    _autotune_enabled,
    qvq_p32_window_ampere,
)


def test_p32_ampere_auto_split_uses_live_resource_inputs():
    assert (
        _auto_split_count(
            in_features=5120, out_features=1024, k_tiles=320, sm_count=108
        )
        == 8
    )
    assert (
        _auto_split_count(
            in_features=5120, out_features=12288, k_tiles=320, sm_count=108
        )
        == 5
    )
    assert (
        _auto_split_count(
            in_features=4096, out_features=17408, k_tiles=256, sm_count=108
        )
        == 1
    )
    assert (
        _auto_split_count(in_features=4096, out_features=1024, k_tiles=8, sm_count=124)
        == 8
    )


def test_p32_ampere_autotune_defaults_on_and_is_bounded(monkeypatch):
    monkeypatch.delenv("QVQ_AMPERE_AUTOTUNE", raising=False)
    assert _autotune_enabled()
    monkeypatch.setenv("QVQ_AMPERE_AUTOTUNE", "0")
    assert not _autotune_enabled()
    candidates = _autotune_candidates(fallback=32, k_tiles=320, max_candidates=6)
    assert candidates == [32, 16, 64, 8, 24, 40]
    assert len(candidates) == len(set(candidates))
    assert all(1 <= candidate <= 320 for candidate in candidates)
    full_candidates = _autotune_candidates(fallback=6, k_tiles=512)
    assert full_candidates == [6, 3, 12, 8, 16, 24, 32, 40, 48, 64, 96, 128]
    wide_candidates = _autotune_candidates(fallback=128, k_tiles=1088)
    assert wide_candidates == [128, 64, 8, 16, 24, 32, 40, 48, 96]
    assert max(wide_candidates) == 128


def test_p32_ampere_autotune_setting_refreshes_with_cache_clear(monkeypatch):
    monkeypatch.setenv("QVQ_AMPERE_AUTOTUNE", "0")
    qvq_ampere_cuda.clear_qvq_ampere_autotune_cache()
    assert not qvq_ampere_cuda._AUTOTUNE_ENABLED

    monkeypatch.delenv("QVQ_AMPERE_AUTOTUNE", raising=False)
    qvq_ampere_cuda.clear_qvq_ampere_autotune_cache()
    assert qvq_ampere_cuda._AUTOTUNE_ENABLED


def test_p32_ampere_autotune_skips_cold_cuda_graph_capture(monkeypatch):
    monkeypatch.setattr(qvq_ampere_cuda, "_AUTOTUNE_CACHE", {})
    monkeypatch.setattr(
        qvq_ampere_cuda, "_autotune_cache_key", lambda *args, **kwargs: "capture"
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    input = torch.empty((1, 256))

    selected = qvq_ampere_cuda._autotune_split_count(
        input,
        input,
        input,
        input,
        transition_bits=6,
        out_features=80,
        bank_alt_id=3,
        fallback=8,
    )
    assert selected == 8
    assert qvq_ampere_cuda._AUTOTUNE_CACHE == {}

    qvq_ampere_cuda._AUTOTUNE_CACHE["capture"] = 4
    assert (
        qvq_ampere_cuda._autotune_split_count(
            input,
            input,
            input,
            input,
            transition_bits=6,
            out_features=80,
            bank_alt_id=3,
            fallback=8,
        )
        == 4
    )


@pytest.mark.parametrize(("size_m", "expected_split"), ((8, 48), (16, 32)))
def test_p32_ampere_dispatches_measured_wmma_kv_plan_directly(
    monkeypatch, size_m, expected_split
):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((size_m, 5120))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        3,
        out_features=1024,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == expected_split


def test_p32_ampere_dispatches_measured_m8_gate_plan_directly(monkeypatch):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((8, 5120))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        3,
        out_features=17408,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == 10


def test_p32_ampere_dispatches_measured_m16_wide_gate_plan_directly(monkeypatch):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((16, 5120))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        3,
        out_features=17408,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == 10


@pytest.mark.parametrize("rate", (2, 2.5, 3, 3.5))
def test_p32_ampere_dispatches_measured_m16_packed_qkv_plan_directly(
    monkeypatch, rate
):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((16, 5120))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        rate,
        out_features=10240,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == 10


@pytest.mark.parametrize("rate", (2, 2.5, 3, 3.5))
def test_p32_ampere_dispatches_measured_m16_wide_fullq_plan_directly(
    monkeypatch, rate
):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((16, 5120))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        rate,
        out_features=12288,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == 10


def test_p32_ampere_dispatches_measured_m16_wide_long_k_plan_directly(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((16, 17408))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        3,
        out_features=5120,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == 24


def test_p32_ampere_dispatches_measured_m8_wide_long_k_plan_directly(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((8, 17408))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        3,
        out_features=5120,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == 40


@pytest.mark.parametrize(
    ("in_features", "out_features", "expected_split"),
    ((5120, 1024, 56), (5120, 12288, 40), (6144, 5120, 48)),
)
def test_p32_ampere_dispatches_measured_m1_plans_directly(
    monkeypatch, in_features, out_features, expected_split
):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((1, in_features))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        3,
        out_features=out_features,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == expected_split


@pytest.mark.parametrize("size_m", (2, 4))
def test_p32_ampere_dispatches_measured_small_m_kv_plans_directly(
    monkeypatch, size_m
):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((size_m, 5120))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        3,
        out_features=1024,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == 64


def test_p32_ampere_dispatches_measured_m4_linear_z_plan_directly(monkeypatch):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((4, 5120))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        3,
        out_features=6144,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == 40


@pytest.mark.parametrize(("bits", "expected_split"), ((3, 48), (3.5, 64)))
def test_p32_ampere_dispatches_measured_m2_attention_plan_directly(
    monkeypatch, bits, expected_split
):
    calls = []
    monkeypatch.setattr(
        qvq_ampere_cuda, "_P32_WINDOW_OP", lambda *args: calls.append(args)
    )
    input = torch.empty((2, 6144))

    qvq_ampere_cuda.qvq_p32_window_ampere(
        input,
        input,
        input,
        input,
        bits,
        out_features=5120,
        bank_alt_id=3,
    )
    assert len(calls) == 1
    assert calls[0][-1] == expected_split


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
@pytest.mark.parametrize("size_m", (1, 2, 4, 8, 16))
def test_p32_window_ampere_matches_exact_matrix(bits, size_m):
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (8, 0):
        pytest.skip("P32 Ampere WMMA requires SM80")

    in_features = 256
    out_features = 80
    tile_count = (in_features // 16) * (out_features // 16)
    generator = torch.Generator(device="cuda").manual_seed(
        20260907 + int(bits * 10) + size_m
    )
    words_per_tile = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    planar = torch.randint(
        0,
        1 << 32,
        (tile_count, words_per_tile),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)
    window = repack_p32_planar_to_window(planar, bits=bits)
    selectors = torch.randint(
        0,
        2,
        (tile_count * 8,),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    bank_ids = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor(3, dtype=torch.uint8, device="cuda")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    input = (
        torch.randn((size_m, in_features), generator=generator, device="cuda") * 0.1
    ).half()
    dense = reconstruct_p32_window_inner_weight(
        window,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
    )
    expected = input.float() @ dense

    actual = qvq_p32_window_ampere(
        input,
        window,
        levels,
        bank_ids,
        bits,
        out_features=out_features,
        bank_alt_id=3,
        split_count=2,
    )

    difference = actual - expected
    max_absolute = difference.abs().max().item()
    mean_absolute = difference.abs().mean().item()
    relative_l2 = (
        difference.float().norm().div(expected.float().norm().clamp_min(1e-12)).item()
    )
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert max_absolute <= 2e-3, (
        f"W{bits:g} M{size_m} max_abs={max_absolute:.7g} "
        f"mean_abs={mean_absolute:.7g} relative_l2={relative_l2:.7g}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(("size_m", "out_features"), ((1, 6144), (2, 1024)))
def test_p32_window_ampere_static_n_dispatch_matches_exact(size_m, out_features):
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (8, 0):
        pytest.skip("P32 Ampere WMMA requires SM80")

    bits = 3.0
    in_features = 256
    tile_count = (in_features // 16) * (out_features // 16)
    generator = torch.Generator(device="cuda").manual_seed(
        20261020 + size_m + out_features
    )
    planar = torch.randint(
        0,
        1 << 32,
        (tile_count, qvq_words_per_tile(bits, weight_count=256, vector_size=2)),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)
    window = repack_p32_planar_to_window(planar, bits=bits)
    bank_ids = pack_qvq_binary_bank_ids(
        torch.randint(
            0,
            2,
            (tile_count * 8,),
            generator=generator,
            device="cuda",
            dtype=torch.uint8,
        )
    )
    bank_alt_id = torch.tensor(3, dtype=torch.uint8, device="cuda")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    input = (
        torch.randn((size_m, in_features), generator=generator, device="cuda") * 0.1
    ).half()
    dense = reconstruct_p32_window_inner_weight(
        window,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
    )
    expected = input.float() @ dense

    actual = qvq_p32_window_ampere(
        input,
        window,
        levels,
        bank_ids,
        bits,
        out_features=out_features,
        bank_alt_id=3,
        split_count=2,
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_p32_window_ampere_cold_autotune_supports_cuda_graph_capture():
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (8, 0):
        pytest.skip("P32 Ampere WMMA requires SM80")

    bits = 3.0
    size_m, in_features, out_features = 1, 256, 80
    tile_count = (in_features // 16) * (out_features // 16)
    generator = torch.Generator(device="cuda").manual_seed(20261021)
    planar = torch.randint(
        0,
        1 << 32,
        (tile_count, qvq_words_per_tile(bits, weight_count=256, vector_size=2)),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)
    window = repack_p32_planar_to_window(planar, bits=bits)
    bank_ids = pack_qvq_binary_bank_ids(
        torch.randint(
            0,
            2,
            (tile_count * 8,),
            generator=generator,
            device="cuda",
            dtype=torch.uint8,
        )
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    input = (
        torch.randn((size_m, in_features), generator=generator, device="cuda") * 0.1
    ).half()
    dense = reconstruct_p32_window_inner_weight(
        window,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=bank_ids,
        bank_alt_id=torch.tensor(3, dtype=torch.uint8, device="cuda"),
    )
    expected = input.float() @ dense

    qvq_p32_window_ampere(
        input,
        window,
        levels,
        bank_ids,
        bits,
        out_features=out_features,
        split_count=2,
    )
    torch.cuda.synchronize()
    qvq_ampere_cuda.clear_qvq_ampere_autotune_cache()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_p32_window_ampere(
            input,
            window,
            levels,
            bank_ids,
            bits,
            out_features=out_features,
        )
    assert qvq_ampere_cuda._AUTOTUNE_CACHE == {}
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=2e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_p32_window_ampere_long_k_is_repeatable_on_non_default_stream():
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (8, 0):
        pytest.skip("P32 Ampere WMMA requires SM80")

    bits = 3.0
    in_features = 2048
    out_features = 80
    tile_count = (in_features // 16) * (out_features // 16)
    generator = torch.Generator(device="cuda").manual_seed(20260980)
    planar = torch.randint(
        0,
        1 << 32,
        (tile_count, qvq_words_per_tile(bits, weight_count=256, vector_size=2)),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)
    window = repack_p32_planar_to_window(planar, bits=bits)
    bank_ids = pack_qvq_binary_bank_ids(
        torch.randint(
            0,
            2,
            (tile_count * 8,),
            generator=generator,
            device="cuda",
            dtype=torch.uint8,
        )
    )
    bank_alt_id = torch.tensor(3, dtype=torch.uint8, device="cuda")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    input = (
        torch.randn((16, in_features), generator=generator, device="cuda") * 0.1
    ).half()
    dense = reconstruct_p32_window_inner_weight(
        window,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
    )
    expected = input.float() @ dense

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        first = qvq_p32_window_ampere(
            input, window, levels, bank_ids, bits, out_features=out_features
        )
        second = qvq_p32_window_ampere(
            input, window, levels, bank_ids, bits, out_features=out_features
        )
    stream.synchronize()

    torch.testing.assert_close(first, expected, atol=2e-3, rtol=0.0)
    assert torch.equal(first, second)
