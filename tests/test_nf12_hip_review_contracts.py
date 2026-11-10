# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native vendored HIP routing bounds and encoder scheduling regressions."""

import ctypes as C
from pathlib import Path

import pytest
import torch

from gptqmodel.utils import nf12_codec


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
    lib.nf12_ad_moe_silu_mul.argtypes = [p, i, i, i, p, i, p]
    lib.nf12_ad_moe_silu_mul.restype = C.c_int
    lib.nf12_ad_moe_combine.argtypes = [p, i, i, i, i, p, i, p]
    lib.nf12_ad_moe_combine.restype = C.c_int
    lib.nf12_ad_wmma_workspace_bytes.argtypes = [i, i, i]
    lib.nf12_ad_wmma_workspace_bytes.restype = C.c_size_t
    lib.nf12_ad_linear_wmma_bf16t.argtypes = [p, p, p, i, i, i, p, p, C.c_size_t, p]
    lib.nf12_ad_linear_wmma_bf16t.restype = C.c_int
    return hip, lib


def test_bf16t_launcher_uses_shared_split_dispatch():
    source = (
        Path(__file__).parents[1]
        / "gptqmodel_ext"
        / "nf12_hip"
        / "nf12_ad_wmma.hip"
    ).read_text(encoding="utf-8")
    assert "NF12_AD_SWITCH(WeightSource::kBf16t)" in source
    assert "kBf16t, false, kMaxTiles, 1" not in source


@pytest.mark.parametrize(
    "n,k,m",
    [
        (4096, 4096, 16),  # four-way split
        (4096, 4096, 64),  # two-way split
        (2048, 4096, 16),  # eight-way split
    ],
)
def test_bf16t_wmma_matches_nf12_split_k_plans(native, n, k, m):
    hip, lib = native
    if lib.nf12_ad_wmma_supported(m) == 0:
        pytest.skip(f"WMMA path unavailable for m={m}")
    workspace_bytes = lib.nf12_ad_wmma_workspace_bytes(n, k, m)
    assert workspace_bytes > 0

    torch.manual_seed(m)
    weight = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    activations = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = hip.encode(weight)
    twin = nf12_codec.pack_bf16t(weight.cpu()).cuda()
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    twin_out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    expected = hip.linear(activations, packed, escapes, int(exp_base))
    status = lib.nf12_ad_linear_wmma_bf16t(
        twin.data_ptr(),
        activations.data_ptr(),
        twin_out.data_ptr(),
        n,
        k,
        m,
        None,
        workspace.data_ptr(),
        workspace_bytes,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert status == 0
    assert torch.equal(
        twin_out.view(torch.int16),
        expected.view(torch.int16),
    )


@pytest.mark.parametrize("magnitude,bias_value", [(2.**25, 1.), (1., 2.**-25), (2.**127, 2.**127)])
def test_split_k_bias_follows_complete_reduction(native, magnitude, bias_value):
    hip, _ = native
    m, n, k = 16, 16, 512
    w = torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
    w[:, 0], w[:, 64] = magnitude, -magnitude
    x = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
    bias = torch.full((n,), bias_value, device="cuda", dtype=torch.bfloat16)
    packed, escapes, base = hip.encode(w)
    # The two exactly representable products cancel, leaving bias unchanged.
    for _ in range(2):
        out = hip.linear(x, packed, escapes, int(base), bias)
        assert bool(out.isfinite().all())
        assert torch.equal(out.view(torch.int16), bias.expand(m, n).contiguous().view(torch.int16))


def test_native_decode_grid_overflow_is_invalid_value(native):
    _, lib = native
    experts, n, k = 65536, 16, 64
    packed = torch.empty((experts, n, k * 3 // 2), device="cuda", dtype=torch.uint8)
    offsets = torch.zeros(experts + 1, device="cuda", dtype=torch.int64)
    bases = torch.zeros(experts, device="cuda", dtype=torch.uint8)
    out = torch.full((experts, n, k), -7., device="cuda", dtype=torch.bfloat16)
    assert lib.nf12_ad_moe_decode(packed.data_ptr(), None, offsets.data_ptr(), bases.data_ptr(),
                                  experts, n, k, None, 0, 0, n, out.data_ptr(), k,
                                  torch.cuda.current_stream().cuda_stream) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all())


@pytest.mark.parametrize("row0,rows,ldo", [
    (16, 16, 64), (0, 32, 64), (32, 0, 64),
    (0, 16, 0), (0, 16, 56), (0, 16, 65),
    (2**63 - 16, 16, 64), (0, 2**63 - 16, 64),
])
def test_native_decode_rejects_invalid_slice_or_stride(native, row0, rows, ldo):
    hip, lib = native
    # The second encoded matrix is a readable canary, outside the declared N.
    w = torch.cat([torch.ones(16, 64), torch.full((16, 64), 2.)]).bfloat16().cuda()
    packed, escapes, exp_base = hip.encode(w)
    ends = torch.tensor([0, escapes.shape[0]], device="cuda", dtype=torch.int64)
    base = torch.tensor([exp_base], device="cuda", dtype=torch.uint8)
    out = torch.full((34, 80), -7., device="cuda", dtype=torch.bfloat16)
    assert lib.nf12_ad_moe_decode(
        packed.data_ptr(), escapes.data_ptr(), ends.data_ptr(), base.data_ptr(),
        1, 16, 64, None, 0, row0, rows, out[1:].data_ptr(), ldo,
        torch.cuda.current_stream().cuda_stream,
    ) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all())


@pytest.mark.parametrize("op", ["silu_mul", "combine"])
@pytest.mark.parametrize("bad", ["input_stride", "output_stride", "stride_alignment",
                                "pointer_alignment", "row_overflow", "stride_overflow"])
def test_native_elementwise_rejects_invalid_buffers(native, op, bad):
    _, lib = native
    x = torch.ones((4, 32), device="cuda", dtype=torch.bfloat16)
    out = torch.full((4, 32), -7., device="cuda", dtype=torch.bfloat16)
    ldi, ldo, rows, offset = 32, 32, 2, 0
    if bad == "input_stride":
        ldi = 0
    elif bad == "output_stride":
        ldo = 0
    elif bad == "stride_alignment":
        ldo = 9
    elif bad == "pointer_alignment":
        offset = 2
    elif bad == "row_overflow":
        rows = 2**62
    else:
        ldi = 2**62
    shape = (rows, 8) if op == "silu_mul" else (rows, 2, 8)
    args = (x.data_ptr(), ldi, *shape, out.data_ptr() + offset, ldo,
            torch.cuda.current_stream().cuda_stream)
    assert getattr(lib, "nf12_ad_moe_" + op)(*args) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all())


@pytest.mark.parametrize("ldx,ldo,count", [(0, 8, 4), (8, 0, 4), (8, 8, -1), (2**62, 8, 4)])
def test_native_grouped_rejects_invalid_buffers(native, ldx, ldo, count):
    _, lib = native
    x = torch.ones((4, 8), device="cuda", dtype=torch.bfloat16)
    out = torch.full_like(x, -7.)
    gather = torch.arange(4, device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int64)
    ends = torch.full((2,), -7, device="cuda", dtype=torch.int32)
    assert lib.nf12_ad_moe_grouped_operands(
        x.data_ptr(), ldx, gather.data_ptr(), 1, offsets.data_ptr(), count, 2, 8,
        out.data_ptr(), ldo, ends.data_ptr(), torch.cuda.current_stream().cuda_stream,
    ) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all()) and bool((ends == -7).all())


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


@pytest.mark.parametrize("count", [524256, 524272, 1048544, 1048545, 1048560, 2097120])
def test_native_wmma_grid_y_boundary(native, count):
    _, lib = native
    n, k = 16, 64
    # Independently construct all-one weights: zero metadata selects escape 0.
    packed = torch.zeros((1536,), device="cuda", dtype=torch.uint8)
    escapes = torch.ones((8,), device="cuda", dtype=torch.bfloat16)
    escape_offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    bases = torch.zeros((1,), device="cuda", dtype=torch.uint8)
    offsets = torch.tensor([0, count], device="cuda", dtype=torch.int64)
    x = torch.ones((count, k), device="cuda", dtype=torch.bfloat16)
    guarded = torch.full((count + 2, n), -7., device="cuda", dtype=torch.bfloat16)
    out = guarded[1:-1]
    rc = lib.nf12_ad_moe_linear(
        packed.data_ptr(), escapes.data_ptr(), escape_offsets.data_ptr(), bases.data_ptr(),
        1, x.data_ptr(), k, None, 1, None, offsets.data_ptr(), count, None,
        out.data_ptr(), n, n, k, torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    assert rc == 0
    assert bool((out == k).all())  # Exact BF16 oracle: sum of 64 ones.
    assert bool((guarded[[0, -1]] == -7).all())


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


@pytest.mark.parametrize("twin", [False, True])
@pytest.mark.parametrize("bad", ["k_zero", "k_tail", "n_zero", "ldx", "ldy", "gather_div",
                                "negative_count", "large_count", "packed", "x", "out", "offsets",
                                "alignment", "empty"])
def test_native_moe_linear_matches_shared_validation(native, twin, bad):
    hip, abi = native
    p, i = C.c_void_p, C.c_int64
    fn = abi.nf12_ad_moe_linear_bf16t if twin else abi.nf12_ad_moe_linear
    fn.restype = C.c_int
    fn.argtypes = ([p] if twin else [p] * 4) + [C.c_int32, p, i, p, i, p, p, i, p, p, i, i, i, p]
    w = torch.ones((1, 16, 64), device="cuda", dtype=torch.bfloat16)
    from types import SimpleNamespace

    if twin:
        weight = w.view(torch.uint8)  # Uniform BF16 values are layout invariant.
    else:
        packed_data, escapes, base = hip.encode(w[0])
        weight = SimpleNamespace(
            packed=packed_data, escapes=escapes,
            escape_offsets=torch.tensor([0, escapes.shape[0]], device="cuda", dtype=torch.int64),
            exp_base=torch.tensor([base], device="cuda", dtype=torch.uint8),
        )
    packed = weight.data_ptr() if twin else weight.packed.data_ptr()
    x = torch.ones((1, 72), device="cuda", dtype=torch.bfloat16)
    out = torch.full((1, 16), -7., device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    n, k, ldx, ldy, div, count = 16, 64, 72, 16, 1, 1
    xp, yp, off = x.data_ptr(), out.data_ptr(), offsets.data_ptr()
    if bad == "k_zero": k = 0
    elif bad == "k_tail": k = 65
    elif bad == "n_zero": n = 0
    elif bad == "ldx": ldx = 0
    elif bad == "ldy": ldy = 0
    elif bad == "gather_div": div = 0
    elif bad == "negative_count": count = -1
    elif bad == "large_count": count = 2**31
    elif bad == "packed": packed = None
    elif bad == "x": xp = None
    elif bad == "out": yp = None
    elif bad == "offsets": off = None
    elif bad == "alignment": xp += 2
    else:
        count, xp, yp, off = 0, None, None, None
    prefix = [packed] if twin else [packed, weight.escapes.data_ptr(),
                                    weight.escape_offsets.data_ptr(), weight.exp_base.data_ptr()]
    expected = 0 if bad == "empty" else 1
    assert fn(*prefix, 1, xp, ldx, None, div, None, off, count, None, yp, ldy, n, k, torch.cuda.current_stream().cuda_stream) == expected
    torch.cuda.synchronize()
    assert bool((out == -7).all())
