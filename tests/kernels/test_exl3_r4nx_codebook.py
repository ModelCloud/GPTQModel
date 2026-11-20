# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Pin the EXL3 R4NX codebook, encoder, and native decode paths.

R4NX reuses the Q4NX tile layout, kv-bit states and 64 zero-prefixed states,
and changes only the state -> weight map::

    product = state * R4NX_MULTIPLIER
    w[c]    = int8((product >> (8 * c)) & 0xFF) * R4NX_ALPHA

``R4NX_MULTIPLIER`` is odd, so ``state -> product`` is a bijection on the
32-bit state space and the four-byte extraction is a bijection onto the
codebook the quantizer searches. The reference below is written with plain
torch and runs on CPU or CUDA, so the golden case still runs where the CUDA
extension cannot be built; the kernel cases skip with a reason.

The fixture targets are exact multiples of 1/8 in ``[-90, 90]``. Every error,
square and four-term sum in a step is then exactly representable in FP32, so
the reference's arithmetic does not depend on whether the compiler contracts
the multiply-adds, and the comparison can demand bit equality. The fixture
asserts that property instead of assuming it.

The last case pins the measured 4.0 bpw codebook quality. R4NX's 256-level
byte grid is coarser in reconstruction error than Q4NX's 31-level nibble-sum
grid at the same state count; the issue's "not worse than Q4NX" target is not
met by this codebook family and the gap is documented in the PR and docs.
"""

import math
import struct

import pytest
import torch

from gptqmodel.exllamav3.trellis import (
    EXL3_CODEBOOK_MARKERS,
    R4NX_RATES,
    trellis_rate,
    validate_trellis_k,
)
from tests.accelerator_gates import requires_nvidia_cuda


R4NX_ALPHA = 1 / math.sqrt(5461.25)
R4NX_MULT = 0x8095576D
Q4NX_ALPHA = 1 / math.sqrt(42.5)
Q4NX_MULT = 0x83DCD12D
STEPS = 64
TILE = 256
# (bits per weight, step bits, default beam) for the four R4NX rates.
RATES = ((2.5, 10, 128), (3.0, 12, 128), (3.5, 14, 16), (4.0, 16, 1))

# Packed bytes captured from the reference encoder on the fixture below.
_GOLDEN = {
    2.5: (
        "d38fa4f5549bfdc17d2631cb66f774ca8f6841291e00b7351ac27b6fb1da7526c5ac5e29f4b20ae5b40f6c41b999bd32"
        "6c41cc30bd32b40fcc30b999eb597d440ea52c2b7fd2590beca5fad6e6adb0aee7d394868d1269f7ba09e6a151bab341"
        "c5ed658dd8082c4d65d8c511ffa267b59939b86a7ab616f71ed03f1ea8a6518466f02a676a908c249d125cfb508ad93d"
        "16f7235acbca7ab600825f25ed11ac91908b1490edaa5b39de9b07c09d8986f0574aac76427931ab5b10bdacaf0ced03"
        "334c6e66ed035b106e66af0c1f11334ccbca7a56d6824329bef59f74ac2b7ba9c8b0762b99a7002d334cbe1c7a561f11"
    ),
    3.0: (
        "5c933a669816f737fb0f1994a83addf2c9822dfa60f5cb28082c59b5c9822d8260f5cb28082c59b5c9822d8260f5cb28"
        "99918e8c11a98441602458f999918e8c11a98441602458f999918e8c11a98441602458f970918e8c6189b377674b57ca"
        "f098e4741f33407a6024b80c70918e8c1f33407a6024b80c70918e8c6a89b377c5964d6a29336b42ee9697b360741c78"
        "48918e8c2e201d60f4e8bc862933ff21e39697b3fbb655572933ffa1eb9697b388c072eb092cd822bfff90411cadd32d"
        "922cd8820b56bf8c88c09255922cd8220b56bf8c88c09255922cd8220856bf8c9419e9c81f914a18084682959419e9c8"
        "1f914a18084682959419e9c81f914a18084682950717e9c8a596387b084682e50717e9c8f03103a4084682cb0717e9c8"
    ),
    3.5: (
        "38339fe65247b6f92fa3dc0c800ed3cab6f9388edc0c5247d3ca2fa3388e800ef0a59cf933fa327b17a4be47e277f4e9"
        "f82afaba9652a2616f8a6672fabae237a261f82ac79d5278b0689686afa3d711526358249686c79dd711b0685824afa3"
        "c79d526396689686ab846f6ff47cec5717a93f573ba53091ec57ab84c79d526330259286ab843ba55263ec579286c79d"
        "3ba53025ec57ab84c79d526317259286f0312f031cea9dc292860727730317251081a76a8d5952631725b0f5fab99503"
        "e6e3380213591ed94897974d92bba30b8d39f43f094ab2f5d192ba9e15c3823fec66f6bd1ecde8cba75f90fa5729bec6"
        "e8cbecc190fa1ecddfd1a75fabe8eb8a4a8986e1299ac959eb8adfbc86e1abe8774ae189a2591a1e8e5e47c08d6191bc"
        "1a1e774a47c0a25991bc8e5e774a8d61a2591a1e12bebd598db15fad1a1e774abd59a2595fad12be774a8db194481a1e"
    ),
    4.0: (
        "0bfea7b15c24e6e357d05bdfcccbcd100bfedd264e13e6e357d05bdfcccbcd100bfedd264e13e6e357d05bdfcccb6191"
        "0bfedd264e13d8d228b45bdfcccb61910bfefea94e13d8d228b4e87567e66191dce1fea9e293d8d2b54ae875f47c6191"
        "f67afea9e293445eb54ae875f47c5380697871a7e293445eb54a465cf47c90dc697871a7e293445eb54a465cf47c90dc"
        "69788b40e293445eb54ada64c560896a69785c24cd10364db54ada64c560896a8afb5c24cd10364da739384bc560e308"
        "8afb5c24cd10364da739dadce6e3e3088afb5c24cd1057d0a739cccbe6e3e3085bdf5c24cd1057d0a7b1cccbe6e30bfe"
        "5bdf5c24cd1057d0dd26cccbe6e30bfe5bdf4e13cd1057d0dd26cccbe6e30bfe5bdf4e13619157d0dd26cccbd8d20bfe"
        "5bdf4e13619128b4fea9cccbd8d20bfee8754e13619128b4fea967e6d8d2dce1e875e2936191b54afea9f47c445e6978"
        "e875e2935380b54a71a7f47c445e6978465ce29390dcb54a71a7f47c445e6978465ce29390dcb54a8b40f47c445e6978"
    ),
}


def _r4nx_values(states):
    """Signed bytes of ``state * R4NX_MULT``, last axis = fragment order."""

    product = (states.to(torch.int64) * R4NX_MULT) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 8, dtype=torch.int64, device=states.device)
    octets = (product.unsqueeze(-1) >> shifts) & 0xFF
    return torch.where(octets >= 128, octets - 256, octets)


def _q4nx_values(states):
    """Q4NX nibble sums for the same state set, used only for the quality pin."""

    product = (states.to(torch.int64) * Q4NX_MULT) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 8, dtype=torch.int64, device=states.device)
    octets = (product.unsqueeze(-1) >> shifts) & 0xFF
    return ((octets >> 4) & 15) + (octets & 15) - 15


def _step_cost(states, target):
    """FP32 squared error of the four decoded weights, in the kernel's order."""

    errors = _r4nx_values(states).to(torch.float32) - target[..., None, :]
    e0, e1, e2, e3 = errors.unbind(-1)
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
    targets = (tiles / tiles.new_tensor(R4NX_ALPHA)).reshape(count, STEPS, 4)

    empty = torch.iinfo(torch.int64).max
    costs = torch.zeros((count, 1), dtype=torch.float32, device=device)
    classes = torch.zeros((count, 1), dtype=torch.int64, device=device)
    history = []
    for step in range(STEPS):
        live = classes.shape[1]
        target = targets[:, step]
        class_p = classes[:, :, None]
        state = (class_p << kv) | suffix
        cost = costs[:, :, None] + _step_cost(state, target[:, None, :])
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


def unpack_states(packed):
    """Read the 64 zero-prefixed trellis states from each packed tile."""

    kv = packed.shape[-1] // 4
    halves = packed.to(torch.int64) & 0xFFFF
    words = halves[..., 0::2] | (halves[..., 1::2] << 16)
    end = kv * torch.arange(1, 65, device=packed.device)
    index = (end - 1) // 32
    shift = 32 * (index + 1) - end
    previous = torch.where(index > 0, words[..., (index - 1).clamp_min(0)], 0)
    window = (previous << 32) | words[..., index]
    return (window >> shift) & ((1 << (32 - kv)) - 1)


def _fixture_targets(count=3):
    """Return exact multiples of 1/8 that survive the kernel's ``input / alpha``."""

    pool = torch.arange(-720, 721, dtype=torch.float32) / 8
    tiles = (pool * R4NX_ALPHA).float()
    pool = pool[tiles / R4NX_ALPHA == pool]
    index = (torch.arange(count * TILE).reshape(count, TILE) * 1103515245 + 12345) % pool.numel()
    return pool[index]


def _fixture_tiles(count=3):
    targets = _fixture_targets(count)
    tiles = (targets * R4NX_ALPHA).float().contiguous()
    # Every step's error, square and four-term sum must be exact in FP32 so the
    # reference cannot diverge from a differently contracted kernel.
    assert torch.equal(tiles / tiles.new_tensor(R4NX_ALPHA), targets), "fixture targets are not exactly representable"
    return tiles


def _golden_bytes(bits):
    raw = bytes.fromhex(_GOLDEN[bits])
    values = struct.unpack(f"<{len(raw) // 2}h", raw)
    return torch.tensor(values, dtype=torch.int16).reshape(-1, int(16 * bits))


def _extension():
    if not requires_nvidia_cuda():
        pytest.skip("R4NX encoding requires an NVIDIA CUDA device")
    from gptqmodel.exllamav3.ext import exllamav3_ext, exllamav3_runtime_available, exllamav3_runtime_error

    if not exllamav3_runtime_available():
        pytest.skip(f"EXL3 CUDA extension unavailable: {exllamav3_runtime_error()}")
    return exllamav3_ext


def _run_kernel(ext, tiles, kv, scale_search):
    count = tiles.shape[0]
    output = torch.empty_like(tiles)
    suffixes = torch.empty_like(tiles, dtype=torch.int16)
    ext.quantize_tiles_r4nx(tiles, output, suffixes, kv, scale_search)
    packed = torch.empty((count, 4 * kv), dtype=torch.int16, device=tiles.device)
    ext.pack_trellis_r4nx(packed.view(count, 1, 4 * kv), suffixes.view(count, 1, TILE), kv)
    return suffixes[:, :STEPS], packed


def test_r4nx_reference_matches_golden_on_cpu():
    """The reference itself is pinned, so this case runs without the extension."""

    tiles = _fixture_tiles()
    for bits, kv, beam in RATES:
        _, packed = encode_tiles(tiles, kv, beam)
        assert torch.equal(packed, _golden_bytes(bits)), f"{bits} bpw reference drifted"


def test_r4nx_marker_and_rate_contract():
    assert "r4nx" in EXL3_CODEBOOK_MARKERS
    assert R4NX_RATES == (2.5, 3, 3.5, 4.0)
    assert trellis_rate(64, r4nx=True) == 4.0
    for bits in R4NX_RATES:
        validate_trellis_k(bits, r4nx=True)
    with pytest.raises(ValueError, match="cannot set both q4nx and r4nx"):
        validate_trellis_k(3.0, q4nx=True, r4nx=True)
    with pytest.raises(ValueError, match="supports 2.5, 3.0, 3.5, or 4.0"):
        validate_trellis_k(2.0, r4nx=True)


def test_r4nx_byte_extraction_is_a_bijection():
    """An odd multiplier makes state -> product injective over all 2^32 states.

    At 4.0 bpw the encoder only ever visits 16-bit states, so this also checks
    the 65536 states it does use map to 65536 distinct level vectors.
    """

    assert R4NX_MULT & 1 == 1
    values = _r4nx_values(torch.arange(1 << 16, dtype=torch.int64))
    assert values.min().item() == -128
    assert values.max().item() == 127
    unique = torch.unique(values, dim=0)
    assert unique.shape[0] == 1 << 16


@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "kv", "beam"), RATES)
@pytest.mark.parametrize("scale_search", [False, True])
def test_r4nx_encode_is_bit_exact_against_reference(bits, kv, beam, scale_search):
    ext = _extension()
    device = torch.device("cuda", torch.cuda.current_device())
    tiles = _fixture_tiles().to(device)
    # The scale search narrows the beam to 16 states for the beam rates.
    expected_codes, expected_packed = encode_tiles(tiles, kv, 16 if scale_search else beam)
    codes, packed = _run_kernel(ext, tiles, kv, scale_search)
    assert torch.equal(codes, expected_codes[:, :STEPS])
    assert torch.equal(packed, expected_packed)
    if not scale_search:
        assert torch.equal(packed, _golden_bytes(bits).to(device))


@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "kv", "beam"), RATES)
@pytest.mark.parametrize("count", [1, 2, 7])
def test_r4nx_encode_reference_holds_for_several_shapes(bits, kv, beam, count):
    ext = _extension()
    device = torch.device("cuda", torch.cuda.current_device())
    tiles = _fixture_tiles(count).to(device)
    expected_codes, expected_packed = encode_tiles(tiles, kv, beam)
    codes, packed = _run_kernel(ext, tiles, kv, False)
    assert torch.equal(codes, expected_codes[:, :STEPS])
    assert torch.equal(packed, expected_packed)


def _reconstruct_reference(packed):
    """Unnormalized R4NX levels in the row-major tile layout ``reconstruct`` writes."""

    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import tensor_core_perm_i

    values = _r4nx_values(unpack_states(packed)).flatten(-2).to(torch.float16)
    tiles_k, tiles_n, _ = packed.shape
    tiles = values[..., tensor_core_perm_i(packed.device)]
    return tiles.reshape(tiles_k, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(tiles_k * 16, tiles_n * 16)


@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "kv", "beam"), RATES)
@pytest.mark.parametrize("tiles_n", [8, 16])
def test_r4nx_reconstruct_matches_reference(bits, kv, beam, tiles_n):
    _extension()
    device = torch.device("cuda", torch.cuda.current_device())
    ops = torch.ops.gptqmodel_exllamav3
    tiles_k = 2
    generator = torch.Generator(device="cpu").manual_seed(20261120 + kv + tiles_n)
    trellis = torch.randint(-32768, 32768, (tiles_k, tiles_n, 4 * kv), generator=generator,
                            dtype=torch.int16).to(device)
    half_k = kv % 4 != 0
    K = kv // 4
    weight = torch.empty((tiles_k * 16, tiles_n * 16), dtype=torch.float16, device=device)
    ops.reconstruct_r4nx(weight, trellis, K, half_k)
    assert torch.equal(weight, _reconstruct_reference(trellis))


def _packed_projection(k, n, bits, seed):
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32768, (k // 16, n // 16, int(16 * bits)),
                            generator=generator, dtype=torch.int16)
    suh = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half()
    svh = (torch.randn(n, generator=generator) * 0.03).half()
    return trellis, suh, svh


@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "kv", "beam"), RATES)
def test_r4nx_native_forward_matches_torch_oracle(bits, kv, beam):
    _extension()
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear

    device = torch.device("cuda", torch.cuda.current_device())
    k, n = 512, 256
    trellis, suh, svh = _packed_projection(k, n, bits, seed=5709 + kv)
    tensors = {
        "trellis": trellis.to(device),
        "suh": suh.to(device),
        "svh": svh.to(device),
        "r4nx": torch.ones((), dtype=torch.int32, device=device),
    }
    native = ExllamaV3Linear.from_tensors(in_features=k, out_features=n, name="r4nx", tensors=tensors).eval()
    oracle = ExllamaV3TorchLinear.from_tensors(in_features=k, out_features=n, name="r4nx_torch",
                                              tensors=tensors).eval()
    x = (torch.randn((3, k), generator=torch.Generator().manual_seed(5709)) * 0.1).half().to(device)
    y = native(x)
    y_ref = oracle(x.float())
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), y_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.cuda
def test_r4nx_marker_round_trips_through_safetensors(tmp_path):
    _extension()
    from safetensors.torch import load_file, save_file

    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear

    device = torch.device("cuda", torch.cuda.current_device())
    k, n = 256, 128
    trellis, suh, svh = _packed_projection(k, n, 4.0, seed=4)
    tensors = {
        "trellis": trellis,
        "suh": suh,
        "svh": svh,
        "r4nx": torch.tensor(0x8095576D - (1 << 32), dtype=torch.int32),
    }
    path = tmp_path / "r4nx.safetensors"
    save_file(tensors, path)
    reloaded = {name: value.to(device) for name, value in load_file(path).items()}
    assert reloaded["r4nx"].item() == 0x8095576D - (1 << 32)

    native = ExllamaV3Linear.from_tensors(in_features=k, out_features=n, name="r4nx", tensors=reloaded).eval()
    oracle = ExllamaV3TorchLinear.from_tensors(in_features=k, out_features=n, name="r4nx_torch",
                                              tensors=reloaded).eval()
    x = (torch.randn((2, k), generator=torch.Generator().manual_seed(4)) * 0.1).half().to(device)
    torch.testing.assert_close(native(x).float(), oracle(x.float()), atol=2e-3, rtol=2e-3)


def _best_loading_mse(levels, samples, loadings):
    """Mean squared error per component of the nearest scaled codebook vector."""

    samples = samples.to(torch.float32)
    levels = levels.to(torch.float32)
    squared_norm = (levels * levels).sum(-1)
    best = None
    for loading in loadings:
        distance = (
            (samples * samples).sum(-1, keepdim=True)
            - 2 * loading * samples @ levels.T
            + loading * loading * squared_norm
        )
        mse = distance.min(dim=1).values.mean().item() / samples.shape[1]
        best = mse if best is None else min(best, mse)
    return best


def test_r4nx_codebook_quality_is_pinned():
    """Pin the measured 4.0 bpw nearest-neighbor error against Q4NX.

    The issue asks for R4NX to be no worse than Q4NX at 4.0 bpw. With 2^16
    states the four bytes of ``state * R4NX_MULTIPLIER`` trace a sparse
    lattice, so the 256-level grid reconstructs worse than Q4NX's 31-level
    nibble-sum grid even though both use the same state count. The pinned
    values make that gap explicit; see ``docs/exl3-native-decode.md``.
    """

    states = torch.arange(1 << 16, dtype=torch.int64)
    samples = torch.randn((512, 4), generator=torch.Generator().manual_seed(5709))
    loadings = (1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85)
    r4nx_mse = _best_loading_mse(_r4nx_values(states) * R4NX_ALPHA, samples, loadings)
    q4nx_mse = _best_loading_mse(_q4nx_values(states) * Q4NX_ALPHA, samples, loadings)
    assert r4nx_mse == pytest.approx(0.00963, rel=0.03)
    assert q4nx_mse == pytest.approx(0.00797, rel=0.03)
    # Documented gap: about 21% worse on this fixture. Fail if the chosen
    # constant regresses far past it; the issue's "not worse than Q4NX" target
    # remains unmet.
    assert r4nx_mse <= 1.35 * q4nx_mse
