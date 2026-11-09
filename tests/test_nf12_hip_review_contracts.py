# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native vendored HIP routing bounds and encoder scheduling regressions."""

import ctypes as C
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def native():
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from gptqmodel.utils import nf12_ad_hip as hip

    if not hip.nf12_ad_hip_supported():
        pytest.skip("RDNA4 GPU required")
    assert hip.prewarm_nf12_ad_hip_extension(), hip.last_error()
    paths = [p for p in torch.ops.loaded_libraries if "gptqmodel_nf12_ad_hip_ops" in Path(p).name]
    assert len(paths) == 1, paths
    lib = C.CDLL(paths[0])
    p, i = C.c_void_p, C.c_int64
    lib.nf12_ad_moe_linear.argtypes = [p, p, p, p, C.c_int32, p, i, p, i, p, p, i, p, p, i, i, i, p]
    lib.nf12_ad_moe_linear.restype = C.c_int
    lib.nf12_ad_moe_align_workspace_bytes.argtypes = [i, i]
    lib.nf12_ad_moe_align_workspace_bytes.restype = C.c_size_t
    lib.nf12_ad_moe_align.argtypes = [p, i, i, i, p, p, p, C.c_size_t, p]
    lib.nf12_ad_moe_align.restype = C.c_int
    lib.nf12_ad_moe_decode.argtypes = [p, p, p, p, C.c_int32, i, i, p, i, i, i, p, i, p]
    lib.nf12_ad_moe_decode.restype = C.c_int
    lib.nf12_ad_moe_grouped_operands.argtypes = [p, i, p, i, p, i, i, i, p, i, p, p]
    lib.nf12_ad_moe_grouped_operands.restype = C.c_int
    return hip, lib


@pytest.fixture(params=[False, True], ids=["default-producer", "nondefault-producer"])
def producer_stream(native, request):
    stream = torch.cuda.Stream() if request.param else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        yield stream
    stream.synchronize()


@pytest.mark.parametrize("raw", [[-1, 1], [0, 3, 1, 4], [3, 1, 2, 4], [-9, -4, 2, 99]])
def test_native_moe_offset_bounds(native, producer_stream, raw):
    hip, lib = native
    count, n, k = 4, 16, 64
    experts = len(raw) - 1
    weights = [torch.full((n, k), j + 1., device="cuda", dtype=torch.bfloat16) for j in range(experts)]
    encoded = [hip.encode(w) for w in weights]
    packed = torch.stack([v[0] for v in encoded])
    escapes = torch.cat([v[1] for v in encoded])
    ends = [0]
    for v in encoded:
        ends.append(ends[-1] + v[1].shape[0])
    escape_offsets = torch.tensor(ends, device="cuda", dtype=torch.int64)
    bases = torch.tensor([v[2] for v in encoded], device="cuda", dtype=torch.uint8)
    offsets = torch.tensor(raw, device="cuda", dtype=torch.int64)
    normalized = offsets.clamp(0, count).cummax(0).values.cpu().tolist()
    # Analytic BF16 reference: row sums of constant integral BF16 weights.
    expected = torch.full((count + 2, n), -7., device="cuda", dtype=torch.bfloat16)
    for expert in range(experts):
        expected[1 + normalized[expert]:1 + normalized[expert + 1]] = k * (expert + 1)
    producer = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(producer)
    with torch.cuda.stream(stream):
        x = torch.ones((count + 2, k), device="cuda", dtype=torch.bfloat16)
        out = torch.full_like(expected, -7.)
        assert lib.nf12_ad_moe_linear(
            packed.data_ptr(), escapes.data_ptr(), escape_offsets.data_ptr(), bases.data_ptr(), experts,
            x[1:].data_ptr(), k, None, 1, None, offsets.data_ptr(), count, None, out[1:].data_ptr(),
            n, n, k, stream.cuda_stream,
        ) == 0
    stream.synchronize()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("raw", [[-1, 1], [0, 3, 1, 4], [3, 1, 2, 4], [-9, -4, 2, 99]])
def test_native_decode_and_grouped_operands_share_offset_bounds(native, raw):
    hip, lib = native
    count, n, k = 4, 16, 64
    experts = len(raw) - 1
    weights = torch.stack([
        torch.full((n, k), j + 1., device="cuda", dtype=torch.bfloat16)
        for j in range(experts)
    ])
    encoded = [hip.encode(w) for w in weights]
    packed = torch.stack([v[0] for v in encoded])
    escapes = torch.cat([v[1] for v in encoded])
    ends = [0]
    for v in encoded:
        ends.append(ends[-1] + v[1].shape[0])
    escape_offsets = torch.tensor(ends, device="cuda", dtype=torch.int64)
    bases = torch.tensor([v[2] for v in encoded], device="cuda", dtype=torch.uint8)
    offsets = torch.tensor(raw, device="cuda", dtype=torch.int64)
    normalized = torch.tensor(raw).clamp(0, count).cummax(0).values.tolist()
    stream = torch.cuda.current_stream().cuda_stream

    decoded = torch.full_like(weights, -7.)
    assert lib.nf12_ad_moe_decode(
        packed.data_ptr(), escapes.data_ptr(), escape_offsets.data_ptr(), bases.data_ptr(),
        experts, n, k, offsets.data_ptr(), count, 0, n, decoded.data_ptr(), k, stream,
    ) == 0
    expected = torch.full_like(weights, -7.)
    for expert in range(experts):
        if normalized[expert] < normalized[expert + 1]:
            expected[expert] = weights[expert]
    assert torch.equal(decoded.view(torch.int16), expected.view(torch.int16))

    x = torch.arange(count * k, device="cuda").reshape(count, k).to(torch.bfloat16)
    gather = torch.arange(count - 1, -1, -1, device="cuda", dtype=torch.int32)
    gathered = torch.full_like(x, -7.)
    group_ends = torch.full((experts,), -1, device="cuda", dtype=torch.int32)
    assert lib.nf12_ad_moe_grouped_operands(
        x.data_ptr(), k, gather.data_ptr(), 1, offsets.data_ptr(), count, experts, k,
        gathered.data_ptr(), k, group_ends.data_ptr(), stream,
    ) == 0
    expected = torch.full_like(x, -7.)
    expected[:normalized[-1]] = x[gather[:normalized[-1]].long()]
    assert group_ends.tolist() == normalized[1:]
    assert torch.equal(gathered.view(torch.int16), expected.view(torch.int16))


def test_native_decode_grid_x_above_32767(native):
    _, lib = native
    n, k = 524288, 64  # 32768 blocks; output is 64 MiB.
    # Every zero metadata word selects escape zero: eight exact BF16 ones.
    # Construct the format directly, independently of the native encoder.
    packed = torch.zeros((n // 16, k // 64, 1536), device="cuda", dtype=torch.uint8)
    escapes = torch.ones((8,), device="cuda", dtype=torch.bfloat16)
    escape_offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    bases = torch.zeros((1,), device="cuda", dtype=torch.uint8)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    decoded = torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
    assert lib.nf12_ad_moe_decode(
        packed.data_ptr(), escapes.data_ptr(), escape_offsets.data_ptr(), bases.data_ptr(),
        1, n, k, offsets.data_ptr(), 1, 0, n, decoded.data_ptr(), k,
        torch.cuda.current_stream().cuda_stream,
    ) == 0
    assert bool((decoded == 1).all())


def test_native_empty_align_with_no_workspace(native):
    _, lib = native
    assert lib.nf12_ad_moe_align_workspace_bytes(0, 4) == 0
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        offsets = torch.full((5,), -7, device="cuda", dtype=torch.int64)
        assert lib.nf12_ad_moe_align(None, 0, 2, 4, None, offsets.data_ptr(), None, 0,
                                     stream.cuda_stream) == 0
    stream.synchronize()
    assert bool((offsets == 0).all())


def test_encode_preserves_every_bf16_bit_pattern(native):
    from gptqmodel.utils import nf12_codec

    hip, _ = native
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(256, 256)
    w = bits.view(torch.bfloat16)
    expected_packed, expected_escapes, expected_base = nf12_codec.encode(w)
    packed, escapes, base = hip.encode(w.cuda())
    assert base == expected_base
    assert torch.equal(packed.cpu(), expected_packed)
    assert torch.equal(escapes.cpu(), expected_escapes)
    assert torch.equal(hip.decode(packed, escapes, base).cpu().view(torch.int16), bits)


@pytest.mark.parametrize("super_atoms", [3, 4095, 4097])
def test_encode_scheduling_preserves_all_bytes(native, super_atoms):
    from gptqmodel.utils import nf12_codec

    hip, _ = native
    gen = torch.Generator().manual_seed(super_atoms)
    bits = torch.randint(-32768, 32768, (16, 64 * super_atoms), dtype=torch.int16, generator=gen)
    w = bits.view(torch.bfloat16)
    expected_packed, expected_escapes, expected_base = nf12_codec.encode(w)
    packed, escapes, base = hip.encode(w.cuda())
    assert base == expected_base
    assert torch.equal(packed.cpu(), expected_packed)
    assert torch.equal(escapes.cpu(), expected_escapes)
    assert torch.equal(hip.decode(packed, escapes, base).cpu().view(torch.int16), bits)
