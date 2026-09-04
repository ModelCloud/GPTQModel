# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Phase-3 coverage for grouped exact-P32 TMA/RS-WGMMA execution."""

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
from gptqmodel.utils import qvq_wgmma_cuda
from gptqmodel.utils.qvq_wgmma_cuda import (
    qvq_h100_large_m_ordered_split_count,
    qvq_p32_window_wgmma_group_plan,
    qvq_p32_window_wgmma_grouped,
    qvq_p32_window_wgmma_grouped_ordered_packed,
    qvq_p32_window_wgmma_grouped_ordered_partials_packed,
    qvq_p32_window_wgmma_grouped_packed,
    qvq_p32_window_wgmma_grouped_reuse2_packed,
    qvq_p32_window_wgmma_grouped_reuse4_packed,
    qvq_p32_window_wgmma_m16_tma,
    qvq_p32_window_wgmma_m16_tma_ordered_split,
    qvq_pack_p32_window_hopper_group,
)


@pytest.mark.parametrize(
    ("logical_rows", "expected"),
    (
        (17, 8),
        (32, 8),
        (64, 8),
        (65, 4),
        (128, 4),
        (129, 2),
        (256, 2),
        (257, 1),
        (4096, 1),
    ),
)
def test_h100_large_m_down_split_policy(logical_rows, expected):
    assert (
        qvq_h100_large_m_ordered_split_count(
            device_name="NVIDIA H100",
            compute_capability=(9, 0),
            logical_rows=logical_rows,
            in_features=8192,
            out_features=2048,
        )
        == expected
    )


def test_h100_large_m_down_split_policy_fails_closed_for_other_devices_and_shapes():
    common = {
        "compute_capability": (9, 0),
        "logical_rows": 64,
        "in_features": 8192,
        "out_features": 2048,
    }
    assert (
        qvq_h100_large_m_ordered_split_count(device_name="NVIDIA H200", **common) == 1
    )
    assert (
        qvq_h100_large_m_ordered_split_count(
            device_name="NVIDIA H100", **(common | {"out_features": 4096})
        )
        == 1
    )


def test_hopper_group_plan_retains_child_boundaries_and_policy():
    plan = qvq_p32_window_wgmma_group_plan(
        torch.empty((16, 512)),
        (torch.empty(1),) * 3,
        torch.empty(256),
        (torch.empty(1),) * 3,
        3,
        out_features=(512, 256, 768),
        bank_alt_ids=(3, 1, 2),
        split_counts=(1, 2, 1),
    )

    assert plan.in_features == 512
    assert plan.transition_bits == 6
    assert plan.out_features == 1536
    assert [segment.output_tile_start for segment in plan.segments] == [0, 32, 48]
    assert [segment.output_tile_count for segment in plan.segments] == [32, 16, 48]
    assert [segment.bank_alt_id for segment in plan.segments] == [3, 1, 2]
    assert [segment.split_count for segment in plan.segments] == [1, 2, 1]


def test_hopper_grouped_window_payload_is_lossless_and_storage_neutral():
    widths = (512, 256, 768)
    plan = qvq_p32_window_wgmma_group_plan(
        torch.empty((16, 512)),
        (torch.empty(1),) * 3,
        torch.empty(256),
        (torch.empty(1),) * 3,
        3.5,
        out_features=widths,
        bank_alt_ids=(3, 1, 2),
        split_counts=(1, 2, 1),
    )
    k_tiles = 32
    words_per_tile = 28
    trellises = tuple(
        torch.arange(
            k_tiles * (width // 16) * words_per_tile, dtype=torch.int32
        ).reshape(-1, words_per_tile)
        + index * 1_000_000
        for index, width in enumerate(widths)
    )
    selectors = tuple(
        torch.arange(k_tiles * (width // 16), dtype=torch.uint8) + index
        for index, width in enumerate(widths)
    )

    payload = qvq_pack_p32_window_hopper_group(trellises, selectors, plan)

    assert payload.trellis.numel() == sum(tensor.numel() for tensor in trellises)
    assert payload.bank_ids.numel() == sum(tensor.numel() for tensor in selectors)
    grouped_trellis = payload.trellis.reshape(k_tiles, -1, words_per_tile)
    grouped_selectors = payload.bank_ids.reshape(k_tiles, -1)
    for trellis, child_selectors, segment in zip(
        trellises, selectors, plan.segments, strict=True
    ):
        start = segment.output_tile_start
        stop = start + segment.output_tile_count
        assert torch.equal(grouped_trellis[:, start:stop].reshape_as(trellis), trellis)
        assert torch.equal(
            grouped_selectors[:, start:stop].reshape_as(child_selectors),
            child_selectors,
        )


def test_hopper_grouped_split_k_fails_closed_to_plain_children(monkeypatch):
    input = torch.empty((16, 512))
    widths = (256, 512)
    plan = qvq_p32_window_wgmma_group_plan(
        input,
        (torch.empty(1),) * 2,
        torch.empty(256),
        (torch.empty(1),) * 2,
        3,
        out_features=widths,
        bank_alt_ids=(1, 2),
        split_counts=(2, 1),
    )
    trellises = tuple(
        torch.empty((32 * (width // 16), 24), dtype=torch.int32) for width in widths
    )
    selectors = tuple(
        torch.empty(32 * (width // 16), dtype=torch.uint8) for width in widths
    )
    payload = qvq_pack_p32_window_hopper_group(trellises, selectors, plan)
    with pytest.raises(ValueError, match="split-1 children"):
        qvq_p32_window_wgmma_grouped_packed(
            input, payload, torch.empty(256, dtype=torch.float16)
        )

    calls = []

    def plain(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full((16, kwargs["out_features"]), len(calls))

    monkeypatch.setattr(qvq_wgmma_cuda, "qvq_p32_window_wgmma_m16_tma", plain)
    outputs = qvq_p32_window_wgmma_grouped(
        input,
        trellises,
        torch.empty(256, dtype=torch.float16),
        selectors,
        3,
        out_features=widths,
        bank_alt_ids=(1, 2),
        split_counts=(2, 1),
    )

    assert len(calls) == 2
    assert [call[1]["split_count"] for call in calls] == [2, 1]
    assert [tuple(output.shape) for output in outputs] == [(16, 256), (16, 512)]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("trellises", (), "one and three"),
        ("bank_ids", (), "metadata lengths"),
        ("out_features", (256,), "metadata lengths"),
        ("bank_alt_ids", (1,), "metadata lengths"),
        ("split_counts", (1,), "split_counts length"),
    ),
)
def test_hopper_group_plan_rejects_invalid_metadata(field, value, message):
    arguments = {
        "trellises": (torch.empty(1), torch.empty(1)),
        "levels": torch.empty(256),
        "bank_ids": (torch.empty(1), torch.empty(1)),
        "bits": 3,
        "out_features": (256, 512),
        "bank_alt_ids": (1, 2),
        "split_counts": (1, 1),
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        qvq_p32_window_wgmma_group_plan(torch.empty((16, 256)), **arguments)


def _hopper_device() -> torch.device | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return None
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) == (9, 0) and (
        "H100" in properties.name or "H200" in properties.name
    ):
        return torch.device("cuda", 0)
    return None


def _payloads(
    *,
    bits: float,
    in_features: int,
    widths: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    windows = []
    selectors = []
    words_per_tile = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    for width in widths:
        tile_count = (in_features // 16) * (width // 16)
        planar = torch.randint(
            0,
            1 << 32,
            (tile_count, words_per_tile),
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).to(torch.int32)
        windows.append(repack_p32_planar_to_window(planar, bits=bits))
        selectors.append(
            pack_qvq_binary_bank_ids(
                torch.randint(
                    0,
                    2,
                    (tile_count * 8,),
                    generator=generator,
                    device=device,
                    dtype=torch.uint8,
                )
            )
        )
    return tuple(windows), tuple(selectors)


@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
@pytest.mark.parametrize("logical_m", (1, 2, 4, 8, 16))
def test_grouped_hopper_is_bit_exact_to_plain_children_and_bounded_by_dense(
    bits, logical_m
):
    device = _hopper_device()
    if device is None:
        pytest.skip("requires an exclusive SM90 H100/H200 validation device")
    in_features = 256
    widths = (256, 512, 256)
    alt_ids = (3, 1, 2)
    generator = torch.Generator(device=device).manual_seed(
        20261201 + int(bits * 10) + logical_m
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = torch.zeros((16, in_features), dtype=torch.float16, device=device)
    input[:logical_m] = (
        torch.randn((logical_m, in_features), generator=generator, device=device).half()
        * 0.1
    )
    windows, selectors = _payloads(
        bits=bits,
        in_features=in_features,
        widths=widths,
        generator=generator,
        device=device,
    )
    plain = tuple(
        qvq_p32_window_wgmma_m16_tma(
            input,
            window,
            levels,
            child_selectors,
            bits,
            out_features=width,
            bank_alt_id=alt_id,
            split_count=1,
        )
        for window, child_selectors, width, alt_id in zip(
            windows, selectors, widths, alt_ids, strict=True
        )
    )
    plan = qvq_p32_window_wgmma_group_plan(
        input,
        windows,
        levels,
        selectors,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=(1, 1, 1),
    )
    payload = qvq_pack_p32_window_hopper_group(windows, selectors, plan)
    grouped = qvq_p32_window_wgmma_grouped_packed(input, payload, levels)

    for child, expected, window, child_selectors, width, alt_id in zip(
        grouped,
        plain,
        windows,
        selectors,
        widths,
        alt_ids,
        strict=True,
    ):
        assert torch.equal(child, expected)
        dense = reconstruct_p32_window_inner_weight(
            window,
            bits=bits,
            in_features=in_features,
            out_features=width,
            bank_ids=child_selectors,
            bank_alt_id=torch.tensor(alt_id, dtype=torch.uint8, device=device),
        )
        torch.testing.assert_close(
            child[:logical_m], input[:logical_m].float() @ dense, rtol=0, atol=2e-3
        )


@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
@pytest.mark.parametrize("logical_m", (32, 64, 128, 256))
def test_grouped_hopper_large_m_is_exact_to_m16_tiles_and_graph_safe(bits, logical_m):
    device = _hopper_device()
    if device is None:
        pytest.skip("requires an exclusive SM90 H100/H200 validation device")
    in_features = 256
    widths = (256, 512, 256)
    alt_ids = (3, 1, 2)
    generator = torch.Generator(device=device).manual_seed(20260904 + int(bits * 10))
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = (
        torch.randn((logical_m, in_features), generator=generator, device=device) * 0.1
    ).half()
    windows, selectors = _payloads(
        bits=bits,
        in_features=in_features,
        widths=widths,
        generator=generator,
        device=device,
    )
    plan = qvq_p32_window_wgmma_group_plan(
        input,
        windows,
        levels,
        selectors,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=(1, 1, 1),
    )
    payload = qvq_pack_p32_window_hopper_group(windows, selectors, plan)
    actual = qvq_p32_window_wgmma_grouped_packed(input, payload, levels)
    reused = qvq_p32_window_wgmma_grouped_reuse2_packed(input, payload, levels)
    reused4 = (
        qvq_p32_window_wgmma_grouped_reuse4_packed(input, payload, levels)
        if logical_m >= 64
        else None
    )
    tiled = tuple(
        torch.cat(
            tuple(
                qvq_p32_window_wgmma_grouped_packed(
                    input[row_start : row_start + 16], payload, levels
                )[child_index]
                for row_start in range(0, logical_m, 16)
            ),
            dim=0,
        )
        for child_index in range(len(widths))
    )

    for child, expected, window, child_selectors, width, alt_id in zip(
        reused,
        tiled,
        windows,
        selectors,
        widths,
        alt_ids,
        strict=True,
    ):
        assert torch.equal(child, expected)
        dense = reconstruct_p32_window_inner_weight(
            window,
            bits=bits,
            in_features=in_features,
            out_features=width,
            bank_ids=child_selectors,
            bank_alt_id=torch.tensor(alt_id, dtype=torch.uint8, device=device),
        )
        torch.testing.assert_close(child, input.float() @ dense, rtol=0, atol=2e-3)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = (
            qvq_p32_window_wgmma_grouped_reuse4_packed(input, payload, levels)
            if reused4 is not None
            else qvq_p32_window_wgmma_grouped_reuse2_packed(input, payload, levels)
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert all(
        torch.equal(child, expected)
        for child, expected in zip(captured, reused4 or reused, strict=True)
    )
    assert all(
        torch.equal(child, expected)
        for child, expected in zip(reused, actual, strict=True)
    )
    if reused4 is not None:
        assert all(
            torch.equal(child, expected)
            for child, expected in zip(reused4, actual, strict=True)
        )


@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
def test_h100_gate_up_n128_reuse_is_exact_and_graph_safe(bits):
    """Exercise the measured N128 x M64 CTA, not a reduced test geometry."""

    device = _hopper_device()
    if device is None:
        pytest.skip("requires an exclusive SM90 H100/H200 validation device")
    logical_m = 128
    in_features = 2048
    widths = (8192, 8192)
    alt_ids = (1, 3)
    generator = torch.Generator(device=device).manual_seed(20260940 + int(bits * 10))
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = (
        torch.randn((logical_m, in_features), generator=generator, device=device) * 0.1
    ).half()
    windows, selectors = _payloads(
        bits=bits,
        in_features=in_features,
        widths=widths,
        generator=generator,
        device=device,
    )
    plan = qvq_p32_window_wgmma_group_plan(
        input,
        windows,
        levels,
        selectors,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=(1, 1),
    )
    payload = qvq_pack_p32_window_hopper_group(windows, selectors, plan)
    expected = qvq_p32_window_wgmma_grouped_packed(input, payload, levels)
    actual = qvq_p32_window_wgmma_grouped_reuse4_packed(input, payload, levels)
    assert all(
        torch.equal(child, reference)
        for child, reference in zip(actual, expected, strict=True)
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_p32_window_wgmma_grouped_reuse4_packed(
            input, payload, levels
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert all(
        torch.equal(child, reference)
        for child, reference in zip(captured, expected, strict=True)
    )


@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
@pytest.mark.parametrize("logical_m", (1, 2, 4, 8, 16))
def test_grouped_ordered_split_is_exact_to_ordered_children(bits, logical_m):
    device = _hopper_device()
    if device is None:
        pytest.skip("requires an exclusive SM90 H100/H200 validation device")
    in_features = 512
    widths = (512, 256, 256)
    alt_ids = (3, 1, 2)
    split_counts = (2, 2, 1)
    generator = torch.Generator(device=device).manual_seed(
        20261210 + int(bits * 10) + logical_m
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = torch.zeros((16, in_features), dtype=torch.float16, device=device)
    input[:logical_m] = (
        torch.randn((logical_m, in_features), generator=generator, device=device).half()
        * 0.1
    )
    windows, selectors = _payloads(
        bits=bits,
        in_features=in_features,
        widths=widths,
        generator=generator,
        device=device,
    )
    expected = tuple(
        qvq_p32_window_wgmma_m16_tma_ordered_split(
            input,
            window,
            levels,
            child_selectors,
            bits,
            out_features=width,
            bank_alt_id=alt_id,
            split_count=split_count,
        )
        for window, child_selectors, width, alt_id, split_count in zip(
            windows,
            selectors,
            widths,
            alt_ids,
            split_counts,
            strict=True,
        )
    )
    plan = qvq_p32_window_wgmma_group_plan(
        input,
        windows,
        levels,
        selectors,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=split_counts,
    )
    payload = qvq_pack_p32_window_hopper_group(windows, selectors, plan)

    grouped = qvq_p32_window_wgmma_grouped_ordered_packed(input, payload, levels)
    repeated = qvq_p32_window_wgmma_grouped_ordered_packed(input, payload, levels)
    partials = qvq_p32_window_wgmma_grouped_ordered_partials_packed(
        input, payload, levels
    )

    assert all(
        torch.equal(child, plain)
        for child, plain in zip(grouped, expected, strict=True)
    )
    assert all(
        torch.equal(child, repeat)
        for child, repeat in zip(grouped, repeated, strict=True)
    )
    partial_offset = 0
    for child, width, split_count in zip(grouped, widths, split_counts, strict=True):
        partial_count = split_count * 16 * width
        child_partials = partials[
            partial_offset : partial_offset + partial_count
        ].reshape(split_count, 16, width)
        reduced = torch.zeros_like(child)
        for split in range(split_count):
            reduced = reduced + child_partials[split]
        assert torch.equal(child, reduced)
        partial_offset += partial_count
    assert partial_offset == partials.numel()
    for child, window, child_selectors, width, alt_id in zip(
        grouped,
        windows,
        selectors,
        widths,
        alt_ids,
        strict=True,
    ):
        dense = reconstruct_p32_window_inner_weight(
            window,
            bits=bits,
            in_features=in_features,
            out_features=width,
            bank_ids=child_selectors,
            bank_alt_id=torch.tensor(alt_id, dtype=torch.uint8, device=device),
        )
        torch.testing.assert_close(
            child[:logical_m], input[:logical_m].float() @ dense, rtol=0, atol=2e-3
        )


def test_grouped_ordered_split_is_cuda_graph_stable_at_llama_qkv_shape():
    device = _hopper_device()
    if device is None:
        pytest.skip("requires an exclusive SM90 H100/H200 validation device")
    bits = 3.0
    in_features = 2048
    widths = (2048, 512, 512)
    alt_ids = (3, 1, 2)
    split_counts = (8, 8, 8)
    generator = torch.Generator(device=device).manual_seed(20261219)
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = torch.randn(
        (16, in_features), generator=generator, device=device, dtype=torch.float16
    )
    windows, selectors = _payloads(
        bits=bits,
        in_features=in_features,
        widths=widths,
        generator=generator,
        device=device,
    )
    plan = qvq_p32_window_wgmma_group_plan(
        input,
        windows,
        levels,
        selectors,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=split_counts,
    )
    payload = qvq_pack_p32_window_hopper_group(windows, selectors, plan)
    expected = qvq_p32_window_wgmma_grouped_ordered_packed(input, payload, levels)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qvq_p32_window_wgmma_grouped_ordered_packed(input, payload, levels)
    graph.replay()
    torch.cuda.synchronize(device)

    assert all(
        torch.equal(child, plain)
        for child, plain in zip(captured, expected, strict=True)
    )


@pytest.mark.parametrize(
    ("group", "widths", "alt_ids"),
    (
        ("qkv", (2048, 512, 512), (3, 1, 2)),
        ("gate_up", (8192, 8192), (1, 3)),
    ),
)
@pytest.mark.parametrize("bits", (2.5, 3.0))
@pytest.mark.parametrize("logical_m", (1, 2, 4, 8, 16))
def test_grouped_hopper_is_exact_at_llama32_1b_shapes(
    group, widths, alt_ids, bits, logical_m
):
    device = _hopper_device()
    if device is None:
        pytest.skip("requires an exclusive SM90 H100/H200 validation device")
    in_features = 2048
    generator = torch.Generator(device=device).manual_seed(
        20261220 + logical_m + (0 if group == "qkv" else 100)
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = torch.zeros((16, in_features), dtype=torch.float16, device=device)
    input[:logical_m] = (
        torch.randn((logical_m, in_features), generator=generator, device=device).half()
        * 0.1
    )
    windows, selectors = _payloads(
        bits=bits,
        in_features=in_features,
        widths=widths,
        generator=generator,
        device=device,
    )
    plain = tuple(
        qvq_p32_window_wgmma_m16_tma(
            input,
            window,
            levels,
            child_selectors,
            bits,
            out_features=width,
            bank_alt_id=alt_id,
            split_count=1,
        )
        for window, child_selectors, width, alt_id in zip(
            windows, selectors, widths, alt_ids, strict=True
        )
    )
    plan = qvq_p32_window_wgmma_group_plan(
        input,
        windows,
        levels,
        selectors,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=(1,) * len(widths),
    )
    payload = qvq_pack_p32_window_hopper_group(windows, selectors, plan)
    grouped = qvq_p32_window_wgmma_grouped_packed(input, payload, levels)

    assert all(
        torch.equal(child, expected)
        for child, expected in zip(grouped, plain, strict=True)
    )
