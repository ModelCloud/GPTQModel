# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exercise native EXL3 launches in a child so a runtime abort is a test failure."""

import struct
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pytest
import torch


def _codebook(codebook):
    states = torch.arange(65536, dtype=torch.int64)
    if codebook == "mul1":
        raw = (states * 0x83DCD12D) & 0xFFFFFFFF
        byte_sum = sum((raw >> shift) & 255 for shift in (0, 8, 16, 24))
        accumulator = (byte_sum + 0x6400).to(torch.int16).view(torch.float16).double()
        scale = struct.unpack("<e", struct.pack("<H", 0x1EEE))[0]
        bias = struct.unpack("<e", struct.pack("<H", 0xC931))[0]
        return (accumulator * scale + bias).half().float()
    raw = states * (89226354 if codebook == "3inst" else 0xCBAC1FED)
    if codebook == "3inst":
        raw += 64248484
    raw = (raw & 0x8FFF8FFF) ^ 0x3B603B60
    halves = torch.stack((raw & 65535, (raw >> 16) & 65535), dim=-1)
    return halves.to(torch.int16).view(torch.float16).double().sum(-1).half().float()


def _torch_oracle(source, bits, codebook):
    """Independent arithmetic for the native FP32 search with FP16 cost storage.

    Preserve native tail-state selection from the penultimate cost buffer and
    the strict-less-than shuffle tree's tie order; neither is changed by the
    compact LDS layout. FP64 multiply/add followed by FP32 models its FMA.
    """
    weights = source.cpu().half().float()
    count = len(weights)
    edges = 65536 >> bits
    states = torch.arange(65536).reshape(1 << bits, edges)
    incoming = states >> bits
    lut = _codebook(codebook)
    decoded = lut[states]
    history = torch.empty((count, 256, edges), dtype=torch.int16)

    def forward(roll, fixed=None):
        costs = None
        previous = None
        for step in range(256):
            ri = (step + roll) % 256
            delta = (decoded[None] - weights[:, ri, None, None]).double()
            candidates = delta * delta
            if costs is not None:
                candidates += costs[:, incoming].double()
            elif fixed is not None:
                candidates.masked_fill_(incoming[None] != fixed[:, None, None], torch.inf)
            best, symbols = candidates.float().min(dim=1)
            history[:, ri] = incoming.gather(0, symbols).to(torch.int16)
            previous, costs = costs, best.half()
        return previous

    previous = forward(128)
    # Each thread first scans its strided edges, then two descending-offset
    # shuffle trees reduce lanes and warps. Ties retain the left operand.
    local_costs, groups = previous.reshape(count, -1, 1024).min(dim=1)
    order = torch.tensor([int(f"{i:010b}"[::-1], 2) for i in range(1024)])
    thread = order[local_costs[:, order].argmin(dim=1)]
    edge = groups.gather(1, thread[:, None]).squeeze(1) * 1024 + thread
    for step in range(255, -1, -1):
        ri = (step + 128) % 256
        edge = history[:, ri].gather(1, edge[:, None]).squeeze(1).long()
        if ri == 0:
            break
    initial = edge
    forward(0, initial)
    indices = torch.empty_like(weights, dtype=torch.int64)
    for ri in range(255, -1, -1):
        previous = history[:, ri].gather(1, edge[:, None]).squeeze(1).long()
        indices[:, ri] = (previous << bits) | edge
        edge = previous
    return lut[indices], indices.to(torch.int16)


def _native_checks(codebook):
    from gptqmodel.exllamav3.ext import exllamav3_ext

    torch.set_num_threads(2)
    props = torch.cuda.get_device_properties(0)
    limit = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    lut = _codebook(codebook)
    # FP16 conversion midpoint and its FP32 neighbors, signed zero, codebook
    # endpoints, and an exact codebook-distance tie with its FP16 neighbors.
    midpoint = torch.tensor(1.0 + 2**-11)
    candidates = lut.reshape(4, -1)
    for left, right in combinations(range(4), 2):
        mid = (candidates[left] + candidates[right]) / 2
        distance = (candidates - mid).abs()
        tied = (
            (mid.half().float() == mid)
            & (candidates[left] != candidates[right])
            & (distance[left] == distance.min(dim=0).values)
        )
        if tied.any():
            tie = mid[tied][0].half()
            break
    else:
        raise AssertionError(f"no representable codebook tie for {codebook}")
    values = torch.stack(
        [
            torch.nextafter(midpoint, torch.tensor(-torch.inf)),
            midpoint,
            torch.nextafter(midpoint, torch.tensor(torch.inf)),
            torch.nextafter(tie, torch.tensor(-torch.inf, dtype=torch.float16)).float(),
            tie.float(),
            torch.nextafter(tie, torch.tensor(torch.inf, dtype=torch.float16)).float(),
            torch.tensor(-0.0),
            torch.tensor(0.0),
            lut.min(),
            lut.max(),
        ]
    )
    source = torch.randn((2, 256), generator=torch.Generator().manual_seed(3528))
    source[1] = values.repeat(26)[:256]
    source = source.cuda()

    def launch(bits, slots):
        edges = 65536 >> bits
        output = torch.full_like(source, 123.0)
        indices = torch.full_like(source, -123, dtype=torch.int16)
        costs = torch.empty((slots, 2, edges), device="cuda", dtype=torch.float16)
        history = torch.empty((slots, 256, edges), device="cuda", dtype=torch.int16)
        required = 4 * edges + 704
        if bits == 2 and 65536 <= limit < required:
            required = 65536
        if required > limit:
            with pytest.raises(RuntimeError, match=f"needs {required} B.*K={bits}.*{limit} B"):
                exllamav3_ext.quantize_tiles(
                    source,
                    output,
                    indices,
                    costs,
                    history,
                    bits,
                    codebook == "mcg",
                    codebook == "mul1",
                )
            torch.cuda.synchronize()
            assert torch.all(output == 123)
            assert torch.all(indices == -123)
            return None
        exllamav3_ext.quantize_tiles(
            source,
            output,
            indices,
            costs,
            history,
            bits,
            codebook == "mcg",
            codebook == "mul1",
        )
        torch.cuda.synchronize()
        return output.cpu(), indices.cpu()

    # Exercise rejection on RDNA4/Ada before a supported call in the same process.
    launch(1, 1)
    for bits in (2, 3):
        expected = _torch_oracle(source, bits, codebook)
        for slots in (1, 2, 1):
            actual = launch(bits, slots)
            if actual is not None:
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("codebook", ["3inst", "mcg", "mul1"])
def test_native_quantize_tiles_memory_layout(codebook):
    if not torch.cuda.is_available():
        pytest.skip("EXL3 native quantization requires CUDA/HIP")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), codebook],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _native_checks(sys.argv[1])
