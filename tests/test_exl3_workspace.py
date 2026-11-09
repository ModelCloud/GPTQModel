# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib import quantize as implementation


def test_workspace_is_private_per_stream(monkeypatch):
    """Each stream owns its scratch so concurrent launches cannot overwrite it."""
    calls = []
    allocated = []

    monkeypatch.delenv("GPTQMODEL_EXL3_TILE_BATCH", raising=False)

    def allocate(device, K, batch):
        allocated.append((device, K, batch))
        return (f"costs-{len(allocated)}", f"edges-{len(allocated)}")

    monkeypatch.setattr(implementation, "_workspace_buffers", {})
    monkeypatch.setattr(implementation, "_allocate_temp_buffers", allocate)
    monkeypatch.setattr(
        implementation,
        "ext",
        SimpleNamespace(quantize_tiles=lambda *args: calls.append(args)),
    )
    tiles = SimpleNamespace(device="cuda:0", shape=(4, 256))
    stream_a = object()
    stream_b = object()
    current = {"stream": stream_a}
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current["stream"])

    implementation._quantize_tiles_with_workspace(tiles, None, None, 4, False, False)
    implementation._quantize_tiles_with_workspace(tiles, None, None, 4, False, False)
    current["stream"] = stream_b
    implementation._quantize_tiles_with_workspace(tiles, None, None, 4, False, False)

    # One buffer per stream, reused by every launch on that stream.
    assert allocated == [(torch.device("cuda:0"), 4, 256)] * 2
    assert calls[0][3] is calls[1][3]
    assert calls[0][4] is calls[1][4]
    assert calls[2][3] is not calls[0][3]
    assert calls[2][4] is not calls[0][4]


def test_tile_batch_defaults_and_clamps(monkeypatch):
    """The scratch slot count is opt-in: unchanged by default, bounded when set."""
    monkeypatch.delenv("GPTQMODEL_EXL3_TILE_BATCH", raising=False)
    assert implementation._tile_batch_size() == 256

    for value, expected in (("8", 8), ("1", 1), ("0", 1), ("-5", 1), ("4096", 256), ("junk", 256)):
        monkeypatch.setenv("GPTQMODEL_EXL3_TILE_BATCH", value)
        assert implementation._tile_batch_size() == expected, value


def test_oversized_tile_sets_are_split_into_launches(monkeypatch):
    """More tiles than scratch slots must split the launch, not overflow it."""
    calls = []

    monkeypatch.setattr(implementation, "_workspace_buffers", {})
    monkeypatch.setattr(
        implementation,
        "_allocate_temp_buffers",
        lambda device, K, batch: (object(), object()),
    )
    monkeypatch.setattr(
        implementation,
        "ext",
        SimpleNamespace(quantize_tiles=lambda tiles, out, idx, *rest: calls.append(tiles.shape[0])),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    monkeypatch.setenv("GPTQMODEL_EXL3_TILE_BATCH", "2")

    tiles = torch.zeros((5, 256))
    implementation._quantize_tiles_with_workspace(
        tiles, torch.zeros_like(tiles), torch.zeros_like(tiles), 4, False, False
    )
    assert calls == [2, 2, 1]

    # At or below the slot count the launch is unchanged.
    calls.clear()
    small = torch.zeros((2, 256))
    implementation._quantize_tiles_with_workspace(
        small, torch.zeros_like(small), torch.zeros_like(small), 4, False, False
    )
    assert calls == [2]


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_concurrent_streams_do_not_overwrite_tile_workspace(device):
    if torch.cuda.device_count() <= torch.device(device).index:
        pytest.skip(f"requires {device}")
    generator = torch.Generator().manual_seed(152)
    tiles = [torch.randn((16, 256), generator=generator).to(device) for _ in range(4)]
    config = {"K": 4, "mcg": True}
    expected = [
        tuple(value.cpu() for value in implementation.quantize_tiles(tile, config))
        for tile in tiles
    ]
    barrier = threading.Barrier(4)

    def quantize(index):
        stream = torch.cuda.Stream(device=device)
        barrier.wait()
        with torch.cuda.stream(stream):
            for _ in range(3):
                actual = implementation.quantize_tiles(tiles[index], config)
                stream.synchronize()
                assert all(
                    torch.equal(value.cpu(), reference)
                    for value, reference in zip(actual, expected[index])
                )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(quantize, range(4)))
