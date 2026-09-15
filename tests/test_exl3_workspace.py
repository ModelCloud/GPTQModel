# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib import quantize as implementation


def test_workspace_orders_kernel_launches_with_events(monkeypatch):
    calls = []
    stream = SimpleNamespace(wait_event=lambda event: calls.append("wait"))
    monkeypatch.setattr(implementation, "_workspace_gates", {})
    monkeypatch.setattr(implementation, "get_temp_buffers", lambda *args: (None, None))
    monkeypatch.setattr(
        implementation,
        "ext",
        SimpleNamespace(quantize_tiles=lambda *args: calls.append("kernel")),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda: SimpleNamespace(record=lambda stream: calls.append("record")),
    )
    tiles = SimpleNamespace(device="cuda:0")
    for _ in range(2):
        implementation._quantize_tiles_with_workspace(
            tiles, None, None, 4, False, False
        )
    assert calls == ["kernel", "record", "wait", "kernel", "record"]


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
