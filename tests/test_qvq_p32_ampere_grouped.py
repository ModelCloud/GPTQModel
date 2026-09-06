# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Phase-2 coverage for segmented exact-P32 execution on SM80."""

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
    qvq_p32_window_ampere,
    qvq_p32_window_ampere_group_plan,
    qvq_p32_window_ampere_grouped,
    qvq_p32_window_ampere_grouped_packed,
    qvq_p32_window_ampere_grouped_kernel_candidates,
    qvq_pack_p32_window_ampere_group,
)


def test_group_plan_retains_ordered_child_metadata():
    input = torch.empty((4, 256))
    trellises = [torch.empty(1, dtype=torch.int32) for _ in range(3)]
    selectors = [torch.empty(1, dtype=torch.uint8) for _ in range(3)]
    levels = torch.empty(256, dtype=torch.float16)

    plan = qvq_p32_window_ampere_group_plan(
        input,
        trellises,
        levels,
        selectors,
        3,
        out_features=(64, 32, 48),
        bank_alt_ids=(3, 1, 2),
        split_counts=(2, 4, 3),
    )

    assert plan.in_features == 256
    assert plan.transition_bits == 6
    assert plan.out_features == 144
    assert [segment.output_tile_start for segment in plan.segments] == [0, 4, 6]
    assert [segment.output_tile_count for segment in plan.segments] == [4, 2, 3]
    assert [segment.split_count for segment in plan.segments] == [2, 4, 3]
    assert [segment.bank_alt_id for segment in plan.segments] == [3, 1, 2]


def test_group_plan_resolves_each_child_width_independently(monkeypatch):
    calls = []

    def resolve(*args, **kwargs):
        del args
        calls.append(kwargs["out_features"])
        return kwargs["out_features"] // 16

    monkeypatch.setattr(qvq_ampere_cuda, "_resolve_split_count", resolve)
    input = torch.empty((1, 256))
    payloads = [torch.empty(1), torch.empty(1), torch.empty(1)]

    plan = qvq_p32_window_ampere_group_plan(
        input,
        payloads,
        torch.empty(256),
        payloads,
        2.5,
        out_features=(64, 32, 48),
        bank_alt_ids=(1, 2, 3),
    )

    assert calls == [64, 32, 48]
    assert [segment.split_count for segment in plan.segments] == [4, 2, 3]
    assert 144 not in calls


def test_grouped_ampere_candidates_tune_children_independently_without_cuda_work():
    candidates = qvq_p32_window_ampere_grouped_kernel_candidates(
        (1, 5120),
        out_features=(1024, 12288),
        bits=3,
        sm_count=108,
        max_candidates=7,
    )

    # The first tuple is the shape-policy baseline for each child.  Subsequent
    # tuples vary one child at a time, so a tuner never has to pretend that the
    # synthetic N=13312 projection has one shared optimum.
    assert candidates == (
        (56, 40),
        (28, 40),
        (56, 20),
        (112, 40),
        (56, 80),
        (8, 40),
        (56, 8),
    )
    assert len(candidates) == len(set(candidates))


def test_ampere_graph_capture_rejects_cold_sm_cache_and_group_pack(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    qvq_ampere_cuda._SM_COUNT_CACHE.clear()
    with pytest.raises(RuntimeError, match="SM-count cache.*before CUDA Graph capture"):
        qvq_ampere_cuda._device_sm_count(torch.device("cuda", 0))

    plan = qvq_p32_window_ampere_group_plan(
        torch.empty((1, 256)),
        (torch.empty(1),),
        torch.empty(256),
        (torch.empty(1),),
        3,
        out_features=(64,),
        bank_alt_ids=(1,),
        split_counts=(2,),
    )
    with pytest.raises(RuntimeError, match="grouped payload.*before CUDA Graph capture"):
        qvq_pack_p32_window_ampere_group(
            (torch.empty((64, 24), dtype=torch.int32),),
            (torch.empty(64, dtype=torch.uint8),),
            plan,
        )


@pytest.mark.parametrize(
    ("shape", "widths", "bits", "message"),
    (
        ((1, 5119), (1024, 512), 3, "input width"),
        ((1, 5120), (), 3, "one through three"),
        ((1, 5120), (1024, 512, 256, 128), 3, "one through three"),
        ((1, 5120), (1025, 512), 3, "output widths"),
        ((1, 5120), (1024, 512), 1, "supports W2"),
    ),
)
def test_grouped_ampere_candidates_reject_invalid_shapes(shape, widths, bits, message):
    with pytest.raises(ValueError, match=message):
        qvq_p32_window_ampere_grouped_kernel_candidates(
            shape, out_features=widths, bits=bits
        )


def test_grouped_window_payload_is_lossless_and_storage_neutral():
    input = torch.empty((2, 256))
    widths = (64, 32, 48)
    plan = qvq_p32_window_ampere_group_plan(
        input,
        (torch.empty(1),) * 3,
        torch.empty(256),
        (torch.empty(1),) * 3,
        3,
        out_features=widths,
        bank_alt_ids=(3, 1, 2),
        split_counts=(2, 4, 3),
    )
    k_tiles = 16
    words_per_tile = 24
    trellises = tuple(
        torch.arange(
            k_tiles * (width // 16) * words_per_tile, dtype=torch.int32
        ).reshape(-1, words_per_tile)
        + index * 100_000
        for index, width in enumerate(widths)
    )
    selectors = tuple(
        torch.arange(k_tiles * (width // 16), dtype=torch.uint8) + index
        for index, width in enumerate(widths)
    )

    payload = qvq_pack_p32_window_ampere_group(trellises, selectors, plan)

    assert payload.trellis.numel() == sum(tensor.numel() for tensor in trellises)
    assert payload.bank_ids.numel() == sum(tensor.numel() for tensor in selectors)
    grouped_trellis = payload.trellis.reshape(k_tiles, -1, words_per_tile)
    grouped_selectors = payload.bank_ids.reshape(k_tiles, -1)
    for source_trellis, source_selectors, segment in zip(
        trellises, selectors, plan.segments, strict=True
    ):
        start = segment.output_tile_start
        stop = start + segment.output_tile_count
        assert torch.equal(
            grouped_trellis[:, start:stop].reshape_as(source_trellis), source_trellis
        )
        assert torch.equal(
            grouped_selectors[:, start:stop].reshape_as(source_selectors),
            source_selectors,
        )


def test_grouped_dispatch_is_one_native_call_with_resolved_plans(monkeypatch):
    calls = []

    def grouped_op(*args):
        calls.append(args)
        return torch.cat(
            [
                torch.full((width,), index, dtype=torch.float32)
                for index, width in enumerate(args[5])
            ],
            dim=0,
        )

    monkeypatch.setattr(qvq_ampere_cuda, "_P32_WINDOW_GROUPED_FUSED_OP", grouped_op)
    input = torch.empty((1, 256))
    trellises = [
        torch.empty((64, 28), dtype=torch.int32),
        torch.empty((32, 28), dtype=torch.int32),
    ]
    selectors = [
        torch.empty(64, dtype=torch.uint8),
        torch.empty(32, dtype=torch.uint8),
    ]

    outputs = qvq_p32_window_ampere_grouped(
        input,
        trellises,
        torch.empty(256, dtype=torch.float16),
        selectors,
        3.5,
        out_features=(64, 32),
        bank_alt_ids=(3, 1),
        split_counts=(2, 4),
    )

    assert len(calls) == 1
    assert calls[0][4:] == (7, [64, 32], [3, 1], [2, 4])
    assert [tuple(output.shape) for output in outputs] == [(1, 64), (1, 32)]


def test_grouped_warp_reducer_case_fails_closed_to_child_dispatcher(monkeypatch):
    input = torch.empty((1, 256))
    widths = (1024, 64)
    plan = qvq_p32_window_ampere_group_plan(
        input,
        (torch.empty(1), torch.empty(1)),
        torch.empty(256),
        (torch.empty(1), torch.empty(1)),
        3,
        out_features=widths,
        bank_alt_ids=(3, 1),
        split_counts=(64, 2),
    )
    trellises = tuple(
        torch.empty((16 * (width // 16), 24), dtype=torch.int32)
        for width in widths
    )
    selectors = tuple(
        torch.empty(16 * (width // 16), dtype=torch.uint8) for width in widths
    )
    payload = qvq_pack_p32_window_ampere_group(trellises, selectors, plan)
    calls = []

    def dispatcher(*args):
        calls.append(args)
        return [torch.empty((1, width), dtype=torch.float32) for width in args[5]]

    monkeypatch.setattr(qvq_ampere_cuda, "_P32_WINDOW_GROUPED_OP", dispatcher)
    monkeypatch.setattr(
        qvq_ampere_cuda,
        "_P32_WINDOW_GROUPED_FUSED_OP",
        lambda *args: pytest.fail("warp-reducer plan must not use fused reduction"),
    )

    outputs = qvq_p32_window_ampere_grouped_packed(
        input, payload, torch.empty(256, dtype=torch.float16)
    )

    assert len(calls) == 1
    assert [tuple(output.shape) for output in outputs] == [(1, 1024), (1, 64)]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("bank_ids", (), "metadata lengths"),
        ("out_features", (64,), "metadata lengths"),
        ("bank_alt_ids", (1,), "metadata lengths"),
        ("split_counts", (1,), "split_counts length"),
    ),
)
def test_grouped_plan_rejects_misaligned_metadata(field, value, message):
    arguments = {
        "trellises": (torch.empty(1), torch.empty(1)),
        "levels": torch.empty(256),
        "bank_ids": (torch.empty(1), torch.empty(1)),
        "bits": 3,
        "out_features": (64, 32),
        "bank_alt_ids": (1, 2),
        "split_counts": (1, 1),
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        qvq_p32_window_ampere_group_plan(torch.empty((1, 256)), **arguments)


def _native_validation_device() -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(0)
    capability = (properties.major, properties.minor)
    if capability == (8, 0):
        return torch.device("cuda", 0)
    if capability == (9, 0) and "H100" in properties.name:
        return torch.device("cuda", 0)
    return None


@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
@pytest.mark.parametrize("size_m", (1, 2, 4, 8, 16))
def test_grouped_ampere_is_bit_exact_to_plain_children_on_h100_or_sm80(
    bits, size_m, monkeypatch
):
    device = _native_validation_device()
    if device is None:
        pytest.skip("requires SM80 or the dedicated H100 validation device")
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) == (9, 0):
        monkeypatch.setenv("QVQ_AMPERE_ALLOW_SM90_VALIDATION", "1")

    in_features = 256
    widths = (64, 80, 96)
    alt_ids = (1, 2, 3)
    split_counts = (1, 2, 4)
    generator = torch.Generator(device=device).manual_seed(
        20261102 + int(bits * 10) + size_m
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = (
        torch.randn((size_m, in_features), generator=generator, device=device) * 0.1
    ).half()
    windows = []
    selectors = []
    expected = []
    for width, alt_id in zip(widths, alt_ids, strict=True):
        tile_count = (in_features // 16) * (width // 16)
        planar = torch.randint(
            0,
            1 << 32,
            (tile_count, qvq_words_per_tile(bits, weight_count=256, vector_size=2)),
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).to(torch.int32)
        window = repack_p32_planar_to_window(planar, bits=bits)
        bank_ids = pack_qvq_binary_bank_ids(
            torch.randint(
                0,
                2,
                (tile_count * 8,),
                generator=generator,
                device=device,
                dtype=torch.uint8,
            )
        )
        dense = reconstruct_p32_window_inner_weight(
            window,
            bits=bits,
            in_features=in_features,
            out_features=width,
            bank_ids=bank_ids,
            bank_alt_id=torch.tensor(alt_id, dtype=torch.uint8, device=device),
        )
        windows.append(window)
        selectors.append(bank_ids)
        expected.append(input.float() @ dense)

    plain = tuple(
        qvq_p32_window_ampere(
            input,
            window,
            levels,
            bank_id,
            bits,
            out_features=width,
            bank_alt_id=alt_id,
            split_count=split_count,
        )
        for window, bank_id, width, alt_id, split_count in zip(
            windows, selectors, widths, alt_ids, split_counts, strict=True
        )
    )
    grouped = qvq_p32_window_ampere_grouped(
        input,
        windows,
        levels,
        selectors,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=split_counts,
    )

    assert len(grouped) == len(plain)
    for grouped_child, plain_child, dense_child in zip(
        grouped, plain, expected, strict=True
    ):
        assert grouped_child.is_contiguous()
        assert torch.equal(grouped_child, plain_child)
        torch.testing.assert_close(grouped_child, dense_child, atol=2e-3, rtol=0.0)


@pytest.mark.parametrize("size_m", (1, 2, 4, 8, 16))
@pytest.mark.parametrize(
    ("widths", "alt_ids", "split_counts"),
    (
        ((2048, 512, 512), (3, 1, 2), (8, 16, 16)),
        ((8192, 8192), (1, 3), (8, 12)),
    ),
    ids=("llama32-1b-qkv", "llama32-1b-gate-up"),
)
def test_grouped_ampere_matches_plain_llama32_1b_shapes(
    size_m, widths, alt_ids, split_counts, monkeypatch
):
    device = _native_validation_device()
    if device is None:
        pytest.skip("requires SM80 or the dedicated H100 validation device")
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) == (9, 0):
        monkeypatch.setenv("QVQ_AMPERE_ALLOW_SM90_VALIDATION", "1")

    bits = 3.0
    in_features = 2048
    generator = torch.Generator(device=device).manual_seed(
        20261120 + size_m + sum(widths)
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = (
        torch.randn((size_m, in_features), generator=generator, device=device) * 0.1
    ).half()
    windows = []
    selectors = []
    for width in widths:
        tile_count = (in_features // 16) * (width // 16)
        planar = torch.randint(
            0,
            1 << 32,
            (tile_count, qvq_words_per_tile(bits, weight_count=256, vector_size=2)),
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

    plain = tuple(
        qvq_p32_window_ampere(
            input,
            window,
            levels,
            bank_id,
            bits,
            out_features=width,
            bank_alt_id=alt_id,
            split_count=split_count,
        )
        for window, bank_id, width, alt_id, split_count in zip(
            windows, selectors, widths, alt_ids, split_counts, strict=True
        )
    )
    grouped = qvq_p32_window_ampere_grouped(
        input,
        windows,
        levels,
        selectors,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=split_counts,
    )

    assert all(
        torch.equal(grouped_child, plain_child)
        for grouped_child, plain_child in zip(grouped, plain, strict=True)
    )
    assert all(grouped_child.is_contiguous() for grouped_child in grouped)


@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
@pytest.mark.parametrize("size_m", (1, 2, 4, 8, 16))
def test_grouped_dynamic_segments_match_plain_static_n_routes(
    bits, size_m, monkeypatch
):
    device = _native_validation_device()
    if device is None:
        pytest.skip("requires SM80 or the dedicated H100 validation device")
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) == (9, 0):
        monkeypatch.setenv("QVQ_AMPERE_ALLOW_SM90_VALIDATION", "1")

    in_features = 256
    widths = (5120, 1024)
    alt_ids = (2, 3)
    split_counts = (2, 3)
    generator = torch.Generator(device=device).manual_seed(
        20261201 + int(bits * 10) + size_m
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = (
        torch.randn((size_m, in_features), generator=generator, device=device) * 0.1
    ).half()
    windows = []
    selectors = []
    for width in widths:
        tile_count = (in_features // 16) * (width // 16)
        planar = torch.randint(
            0,
            1 << 32,
            (tile_count, qvq_words_per_tile(bits, weight_count=256, vector_size=2)),
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

    plain = tuple(
        qvq_p32_window_ampere(
            input,
            window,
            levels,
            bank_id,
            bits,
            out_features=width,
            bank_alt_id=alt_id,
            split_count=split_count,
        )
        for window, bank_id, width, alt_id, split_count in zip(
            windows, selectors, widths, alt_ids, split_counts, strict=True
        )
    )
    grouped = qvq_p32_window_ampere_grouped(
        input,
        windows,
        levels,
        selectors,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=split_counts,
    )

    assert all(
        grouped_child.is_contiguous() and torch.equal(grouped_child, plain_child)
        for grouped_child, plain_child in zip(grouped, plain, strict=True)
    )
