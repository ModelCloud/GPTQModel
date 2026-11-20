# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Pin the EXL3 Q4NX encoder against a device-independent Viterbi reference.

``quantize_tiles_q4nx`` plus ``pack_trellis_q4nx`` must reproduce the bytes of a
straightforward implementation of the documented beam-Viterbi recurrence. The
reference below is written with plain torch and runs on CPU or CUDA, so the
golden case still runs where the CUDA extension cannot be built; the kernel
cases skip with a reason.

The fixture targets are exact multiples of 1/8. Every error, square and
four-term sum in a step is then exactly representable in FP32, so the
reference's arithmetic does not depend on whether the compiler contracts the
multiply-adds, and the comparison can demand bit equality. The fixture asserts
that property instead of assuming it, so a platform whose FP32 division rounds
differently fails loudly rather than comparing the wrong values.

Recurrence, per 256-value tile and per step of four weights:

* a state is the ``32 - kv`` bit window over the suffix stream; a candidate
  appends its ``kv`` bit suffix and keeps the low ``32 - 2 * kv`` bits as the
  successor class, which becomes the high half of the next state;
* a candidate costs ``prior + sum((byte_sum - 15 - target) ** 2)``;
* candidates are ordered by the 64-bit key ``(cost_bits, state, predecessor)``;
* each step keeps the ``beam`` smallest keys, or every class when
  ``beam == class_count``.
"""

import math
import struct

import pytest
import torch


Q4NX_ALPHA = 1 / math.sqrt(42.5)
Q4NX_MULT = 0x83DCD12D
STEPS = 64
TILE = 256
# (bits per weight, step bits, default beam) for the three Q4NX rates.
RATES = ((2.5, 10, 128), (3.0, 12, 128), (3.5, 14, 16))

# Packed bytes captured from the pre-optimization encoder on the same fixture.
_GOLDEN = {
    2.5: (
        "edba15c0d8d3b666b52cfbe32a4e7dc9f3e44a1a01423c1808106500c842b62cbea45bc25e2510aaf2ae806e1196c3f8391484e82d0b02449630b29084aa2fe9a09b570930bebc2b217a84e500c00b0502441940b2902d0b2fe99630570984aabc2ba09b84e530be0e05217acb820051258c2c64a1ea4bbae8e655028c2fef0a881e613900f0424166edba35e3d8d3b6c9b52cfb1a2a4e7d54c3e64a4afc347cbf573d20b220400009230bd9a8fa926eba799540e3cbbb02a247580e10e5501042b62c08a45bc2c82510aabeae806e5e96c3f8f21484e8115b03002f6d6bd6aeb23f8e3dd4975ccbaea4a1e2c347356c"
    ),
    3.0: (
        "d65b05906af8bb6eaf05b6a22ab1b1343c4c58c3645ba00178c52110098000ae66bd5500aa86bfebf35a602bac121b4bc0c3843541b6051a8a571c02900008e06ed65b05a26af8bb34af05b6c32ab1b1013c4c5810645ba0ae78c52100098000900008106ed65b05a26af8bb34af05b6c32ab1b1013c4c5810645ba0ae78c52100098000eb66bd552baa86bf4bf35a6035ac121b1ac0c3840241b605e08a571c05900008bb6ed65bb6a26af8b134af0558c32ab1a0013c4c2110645b00ae78c5f07f539b05900008bb6ed65bb6a26af8b134af0558c32ab1a0013c4c2110645b00ae78c555000980bfeb66bd602baa861b4bf35a8435ac12051ac0c31c0241b608e08a575b059000f8bb6ed605b6a26ab1b134af4c58c32a5ba0013cc5211064"
    ),
    3.5: (
        "b6750124ef0b83adaa38496f78550628e83e88b050b95e5eb08f40264ca7912ddc4a01a1d60590002f0cb6dae224bdbd5519a0a8fb20c2e2e57a79a13e0299409d46b6c02b0584321740027030d86a5b93f4f6be6580a28a83088b57ebe585ee086402951ad902fb1410ca740009c0ad028000046a5b1740f6be30d8a28a93f48b57658085ee83080295ebe502fb0864ca741ad9c0ad14106d5d0009fbc260ab2a4ed2db5e95018aba0f222c54ae9717ec239009d369640bb7524028750124000b83adb638496fef550628aa3e88b078b95e5ee88f402650a7912db000adde4f402317872400b752adb675016fef0b8328aa3849b07855065ee83e882650b95e2db08f40a14ca79100dc4a01dad60590bd2f0cb6a8e224bde25519a0a1fb20c240e57a79c03e0299329d46b6702b05845b174002be30d86a8a93f4f6576580a2ee83088b95ebe585fb0864027a3fd902"
    ),
}


def _nibble_sums(states):
    """``q4nx_nibble_sums`` over uint32 states held in int64."""

    product = (states.to(torch.int64) * Q4NX_MULT) & 0xFFFFFFFF
    return ((product >> 4) & 0x0F0F0F0F) + (product & 0x0F0F0F0F)


def _step_cost(states, target):
    """FP32 squared error of the four decoded weights, in the kernel's order."""

    sums = _nibble_sums(states)
    errors = [
        ((sums >> (8 * byte)) & 0xFF).to(torch.float32) - 15.0 - target[..., byte]
        for byte in range(4)
    ]
    e0, e1, e2, e3 = errors
    return e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3


def _keys(cost, state, previous):
    bits = cost.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    path = ((state << 7) | previous) & 0xFFFFFFFF
    return (bits << 32) | path


def encode_tiles(tiles, kv, beam):
    """Return ``(suffixes, packed)`` exactly as ``quantize``/``pack_trellis`` do."""

    tiles = tiles.to(torch.float32)
    device = tiles.device
    count = tiles.shape[0]
    suffix_count = 1 << kv
    class_count = 1 << (32 - 2 * kv)
    exact = beam == class_count
    old_mask = (1 << max(0, 32 - 3 * kv)) - 1
    suffix = torch.arange(suffix_count, device=device, dtype=torch.int64)
    class_of_suffix = suffix & (class_count - 1)
    targets = (tiles / Q4NX_ALPHA).reshape(count, STEPS, 4)

    empty = torch.iinfo(torch.int64).max
    costs = torch.zeros((count, 1), dtype=torch.float32, device=device)
    classes = torch.zeros((count, 1), dtype=torch.int64, device=device)
    history = []
    for step in range(STEPS):
        live = classes.shape[1]
        target = targets[:, step]
        class_p = classes[:, :, None]
        state = (class_p << kv) | suffix
        cost = costs[:, :, None] + _step_cost(state, target[:, None, None, :])
        previous = torch.arange(live, device=device)[None, :, None].expand_as(state)
        key = _keys(cost, state, previous)
        index = (((class_p & old_mask) << kv) | class_of_suffix).expand(count, live, suffix_count)
        candidates = torch.full((count, class_count), empty, dtype=torch.int64, device=device)
        candidates.scatter_reduce_(1, index.reshape(count, -1), key.reshape(count, -1), reduce="amin")
        if exact:
            keep = torch.arange(class_count, device=device).expand(count, class_count)
        else:
            keep = torch.sort(candidates, dim=1, stable=True).indices[:, :beam]
        selected = torch.gather(candidates, 1, keep)
        history.append(selected)
        costs = (selected >> 32).to(torch.int32).view(torch.float32)
        classes = keep

    suffixes = torch.zeros((count, STEPS), dtype=torch.int64, device=device)
    slot = torch.argmin(history[-1], dim=1)
    for step in range(STEPS - 1, -1, -1):
        path = history[step].gather(1, slot[:, None]).squeeze(1) & 0xFFFFFFFF
        suffixes[:, step] = (path >> 7) & (suffix_count - 1)
        slot = path & 127

    codes = torch.zeros((count, TILE), dtype=torch.int16, device=device)
    codes[:, :STEPS] = suffixes.to(torch.int16)
    return codes, pack_trellis(codes, kv)


def pack_trellis(codes, kv):
    """Bit-exact reimplementation of ``pack_kernel``."""

    count = codes.shape[0]
    words = kv * 2
    per_window = (32 + 2 * kv - 2) // kv
    stream = codes.to(torch.int64) & ((1 << kv) - 1)
    zeros = torch.zeros(count, dtype=torch.int64, device=codes.device)
    packed = torch.zeros((count, words), dtype=torch.int64, device=codes.device)
    for index in range(words):
        start = index * 32
        step = start // kv
        window = torch.zeros(count, dtype=torch.int64, device=codes.device)
        for offset in range(per_window):
            column = step + offset
            window = (window << kv) | (stream[:, column] if column < STEPS else zeros)
        packed[:, index] = window >> (per_window * kv - 32 - start % kv)
    halves = torch.zeros((count, 2 * words), dtype=torch.int64, device=codes.device)
    halves[:, 0::2] = packed & 0xFFFF
    halves[:, 1::2] = (packed >> 16) & 0xFFFF
    return halves.to(torch.int16)


def _fixture_targets(count=3):
    """Return exact multiples of 1/8 that survive the kernel's ``input / alpha``."""

    pool = torch.tensor(
        [-20.0, -16.0, -15.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.125, 0.0,
         0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 16.0, 20.0],
        dtype=torch.float32,
    )
    tiles = (pool * Q4NX_ALPHA).float()
    pool = pool[tiles / Q4NX_ALPHA == pool]
    index = (torch.arange(count * TILE).reshape(count, TILE) * 5 + torch.arange(count)[:, None] * 3) % pool.numel()
    return pool[index]


def _fixture_tiles(count=3):
    targets = _fixture_targets(count)
    tiles = (targets * Q4NX_ALPHA).float().contiguous()
    # Every step's error, square and four-term sum must be exact in FP32 so the
    # reference cannot diverge from a differently contracted kernel.
    assert torch.equal(tiles / Q4NX_ALPHA, targets), "fixture targets are not exactly representable"
    return tiles


def _golden_bytes(bits):
    raw = bytes.fromhex(_GOLDEN[bits])
    values = struct.unpack(f"<{len(raw) // 2}h", raw)
    return torch.tensor(values, dtype=torch.int16).reshape(-1, int(16 * bits))


def _extension():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("Q4NX encoding requires an NVIDIA CUDA device")
    from gptqmodel.exllamav3.ext import exllamav3_ext, exllamav3_runtime_available, exllamav3_runtime_error

    if not exllamav3_runtime_available():
        pytest.skip(f"EXL3 CUDA extension unavailable: {exllamav3_runtime_error()}")
    return exllamav3_ext


def _run_kernel(ext, tiles, kv, scale_search):
    count = tiles.shape[0]
    output = torch.empty_like(tiles)
    suffixes = torch.empty_like(tiles, dtype=torch.int16)
    ext.quantize_tiles_q4nx(tiles, output, suffixes, kv, scale_search)
    packed = torch.empty((count, 4 * kv), dtype=torch.int16, device=tiles.device)
    ext.pack_trellis_q4nx(packed.view(count, 1, 4 * kv), suffixes.view(count, 1, TILE), kv)
    return suffixes[:, :STEPS], packed


def test_q4nx_reference_matches_golden_on_cpu():
    """The reference itself is pinned, so this case runs without the extension."""

    tiles = _fixture_tiles()
    for bits, kv, beam in RATES:
        _, packed = encode_tiles(tiles, kv, beam)
        assert torch.equal(packed, _golden_bytes(bits)), f"{bits} bpw reference drifted"


@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "kv", "beam"), RATES)
@pytest.mark.parametrize("scale_search", [False, True])
def test_q4nx_encode_is_bit_exact_against_reference(bits, kv, beam, scale_search):
    ext = _extension()
    device = torch.device("cuda", torch.cuda.current_device())
    tiles = _fixture_tiles().to(device)
    # The scale search narrows the beam to 16 states for the two beam rates.
    expected_codes, expected_packed = encode_tiles(tiles, kv, 16 if scale_search else beam)
    codes, packed = _run_kernel(ext, tiles, kv, scale_search)
    assert torch.equal(codes, expected_codes[:, :STEPS])
    assert torch.equal(packed, expected_packed)
    if not scale_search:
        assert torch.equal(packed, _golden_bytes(bits).to(device))


@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "kv", "beam"), RATES)
@pytest.mark.parametrize("count", [1, 2, 7])
def test_q4nx_encode_reference_holds_for_several_shapes(bits, kv, beam, count):
    ext = _extension()
    device = torch.device("cuda", torch.cuda.current_device())
    tiles = _fixture_tiles(count).to(device)
    expected_codes, expected_packed = encode_tiles(tiles, kv, beam)
    codes, packed = _run_kernel(ext, tiles, kv, False)
    assert torch.equal(codes, expected_codes[:, :STEPS])
    assert torch.equal(packed, expected_packed)
