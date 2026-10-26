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

    def allocate(device, K):
        allocated.append((device, K))
        return (f"costs-{len(allocated)}", f"edges-{len(allocated)}")

    monkeypatch.setattr(implementation, "_workspace_buffers", {})
    monkeypatch.setattr(implementation, "_allocate_temp_buffers", allocate)
    monkeypatch.setattr(
        implementation,
        "ext",
        SimpleNamespace(quantize_tiles=lambda *args: calls.append(args)),
    )
    tiles = SimpleNamespace(device="cuda:0")
    stream_a = object()
    stream_b = object()
    current = {"stream": stream_a}
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current["stream"])

    implementation._quantize_tiles_with_workspace(tiles, None, None, 4, False, False)
    implementation._quantize_tiles_with_workspace(tiles, None, None, 4, False, False)
    current["stream"] = stream_b
    implementation._quantize_tiles_with_workspace(tiles, None, None, 4, False, False)

    # One buffer per stream, reused by every launch on that stream.
    assert allocated == [(torch.device("cuda:0"), 4)] * 2
    assert calls[0][3] is calls[1][3]
    assert calls[0][4] is calls[1][4]
    assert calls[2][3] is not calls[0][3]
    assert calls[2][4] is not calls[0][4]


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
