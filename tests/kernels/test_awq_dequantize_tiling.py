# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ GEMM dequantization: tiled broadcast, exact codes, bounded scratch.

``dequantize_gemm`` used to expand scales and zero points with
``repeat_interleave`` and keep several full-size INT64/INT32 copies alive at
once. These tests pin the replacement against an oracle that decodes the packed
words itself, against the removed formulation, and against a bound on the
temporary storage of one call.
"""

import pytest
import torch

import gptqmodel.quantization.awq.utils.packing_utils as packing_utils


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    # AwqTorchLinear is portable; the cuda runtime covers both CUDA and ROCm.
    return torch.device(request.param)


def _pack_hand(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack codes without using the module under test.

    Four-bit words interleave the codes with ``AWQ_ORDER``; every other width is
    a continuous little-endian bit stream.
    """

    rows, columns = codes.shape
    values = codes.to(torch.int64) & ((1 << bits) - 1)
    if bits == 4:
        words = columns // 8
        grouped = values.view(rows, words, 8)[:, :, [0, 2, 4, 6, 1, 3, 5, 7]]
        packed = torch.zeros((rows, words), dtype=torch.int64)
        for lane in range(8):
            packed |= grouped[:, :, lane] << (bits * lane)
        return packed.to(torch.int32)

    words = (columns * bits + 31) // 32
    packed = torch.zeros((rows, words), dtype=torch.int64)
    for index in range(columns):
        bit = index * bits
        word, shift = divmod(bit, 32)
        packed[:, word] |= values[:, index] << shift
        if shift + bits > 32:
            packed[:, word + 1] |= values[:, index] >> (32 - shift)
    return packed.to(torch.int32)


def _unpack_hand(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """Decode packed words back to natural column order without the module."""

    rows, words = packed.shape
    raw = packed.cpu().to(torch.int64) & 0xFFFFFFFF
    if bits == 4:
        lanes = 8
        codes = torch.empty((rows, words * lanes), dtype=torch.int64)
        for word in range(words):
            for lane in range(lanes):
                codes[:, word * lanes + lane] = (raw[:, word] >> (bits * lane)) & 15
        # Packed lane `i` holds natural column `AWQ_ORDER[i]`.
        reordered = codes.view(rows, words, lanes)[:, :, [0, 4, 1, 5, 2, 6, 3, 7]]
        return reordered.reshape(rows, words * lanes)[:, :count]

    codes = torch.empty((rows, count), dtype=torch.int64)
    mask = (1 << bits) - 1
    for index in range(count):
        bit = index * bits
        word, shift = divmod(bit, 32)
        value = raw[:, word] >> shift
        if shift + bits > 32 and word + 1 < words:
            value |= raw[:, word + 1] << (32 - shift)
        codes[:, index] = value & mask
    return codes


def _previous_dequantize(qweight, qzeros, scales, bits, group_size):
    """The removed formulation: expand both operands, then combine."""

    iweight, izeros = packing_utils.unpack_awq(qweight, qzeros, bits, scales.shape[-1])
    iweight, izeros = packing_utils.reverse_awq_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    return (iweight - izeros) * scales


def _reference_weight(codes, zeros, scales, group_size, dtype):
    """FP64 oracle: integer difference times the group scale, rounded once."""

    in_features = codes.shape[0]
    group = in_features if group_size is None or group_size <= 0 else group_size
    groups = torch.clamp(torch.arange(in_features) // group, max=scales.shape[0] - 1)
    exact = (codes.to(torch.float64) - zeros.to(torch.float64)[groups]) * scales.to(
        torch.float64
    )[groups]
    return exact.to(dtype)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [16, 32, 128])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matches_previous_repeat_interleave_formulation(bits, group_size, scale_dtype, device):
    torch.manual_seed(0)
    in_features, out_features = 256, 64
    groups = in_features // group_size
    int_weight = torch.randint(0, 1 << bits, (in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 1 << bits, (groups, out_features), dtype=torch.int32)
    qweight, qzeros = packing_utils.pack_awq(int_weight, zero_points, bits)
    scales = ((torch.rand(groups, out_features) * 0.5) + 0.75).to(scale_dtype)

    expected = _previous_dequantize(qweight, qzeros, scales, bits, group_size)
    actual = packing_utils.dequantize_gemm(
        qweight.to(device), qzeros.to(device), scales.to(device), bits, group_size
    )

    assert actual.dtype is scale_dtype
    assert actual.shape == (in_features, out_features)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matches_hand_decoded_oracle(bits, scale_dtype, device):
    """The oracle packs and decodes the words itself, in natural column order."""

    torch.manual_seed(1)
    in_features, out_features, group_size = 64, 32, 16
    groups = in_features // group_size
    codes = torch.randint(0, 1 << bits, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 1 << bits, (groups, out_features), dtype=torch.int32)
    qweight, qzeros = _pack_hand(codes, bits), _pack_hand(zeros, bits)
    scales = ((torch.rand(groups, out_features) * 0.5) + 0.75).to(scale_dtype)

    actual = packing_utils.dequantize_gemm(
        qweight.to(device), qzeros.to(device), scales.to(device), bits, group_size
    )
    expected = _reference_weight(codes, zeros, scales, group_size, scale_dtype)

    assert actual.dtype == scale_dtype
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(_unpack_hand(qweight, bits, out_features), codes.to(torch.int64))
    assert torch.equal(_unpack_hand(qzeros, bits, out_features), zeros.to(torch.int64))


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_endpoint_codes_zeros_and_scales(scale_dtype, device):
    """Clamp endpoints, zero/negative scales, and group-boundary rows."""

    bits, in_features, out_features, group_size = 4, 64, 32, 16
    groups = in_features // group_size
    codes = torch.zeros((in_features, out_features), dtype=torch.int32)
    codes[::2] = (1 << bits) - 1
    codes[:, 1] = (1 << bits) - 1
    zeros = torch.zeros((groups, out_features), dtype=torch.int32)
    zeros[:, ::2] = (1 << bits) - 1
    scales = torch.zeros((groups, out_features), dtype=scale_dtype)
    scales[:, 0] = 1.0
    scales[:, 1] = -1.0
    scales[:, 2] = 1e-4
    qweight, qzeros = _pack_hand(codes, bits), _pack_hand(zeros, bits)

    actual = packing_utils.dequantize_gemm(
        qweight.to(device), qzeros.to(device), scales.to(device), bits, group_size
    )
    expected = _reference_weight(codes, zeros, scales, group_size, scale_dtype)

    assert torch.equal(actual.cpu(), expected)
    assert torch.isfinite(actual).all()
    for group in range(groups):
        start = group * group_size
        assert torch.equal(
            actual[start : start + group_size].cpu(), expected[start : start + group_size]
        )


@pytest.mark.parametrize("group_size", [None, -1, 0, 64])
def test_per_channel_group_spans_all_rows(group_size, device):
    torch.manual_seed(2)
    bits, in_features, out_features = 4, 64, 32
    codes = torch.randint(0, 1 << bits, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 1 << bits, (1, out_features), dtype=torch.int32)
    qweight, qzeros = _pack_hand(codes, bits), _pack_hand(zeros, bits)
    scales = torch.rand((1, out_features), dtype=torch.float16) + 0.5

    actual = packing_utils.dequantize_gemm(
        qweight.to(device), qzeros.to(device), scales.to(device), bits, group_size
    )
    expected = _reference_weight(codes, zeros, scales, group_size, torch.float16)

    assert actual.shape == (in_features, out_features)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("groups", [2, 3])
def test_uneven_tail_uses_the_last_group(groups, device):
    """``in_features`` is not a multiple of ``group_size``.

    The removed formulation expanded the scale rows to ``groups * group_size``
    and could not combine them with the shorter weight. The replacement
    returns exactly ``in_features`` rows and keeps the last group's scale on the
    rows past the final full group.
    """

    torch.manual_seed(3)
    bits, in_features, out_features, group_size = 4, 40, 32, 16
    codes = torch.randint(0, 1 << bits, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 1 << bits, (groups, out_features), dtype=torch.int32)
    qweight, qzeros = _pack_hand(codes, bits), _pack_hand(zeros, bits)
    scales = torch.rand((groups, out_features), dtype=torch.float16) + 0.5

    actual = packing_utils.dequantize_gemm(
        qweight.to(device), qzeros.to(device), scales.to(device), bits, group_size
    )
    expected = _reference_weight(codes, zeros, scales, group_size, torch.float16)

    assert actual.shape == (in_features, out_features)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("tile_bytes", [1, 384, 512, 1024, 4096, 64 * 1024])
@pytest.mark.parametrize("in_features,group_size", [(128, 16), (136, 16), (128, -1), (136, 64)])
def test_tile_size_does_not_change_the_result(monkeypatch, tile_bytes, in_features, group_size, device):
    """Splitting groups and ragged tails across tiles must preserve every value."""

    torch.manual_seed(4)
    bits, out_features = 4, 32
    groups = 1 if group_size <= 0 else (in_features + group_size - 1) // group_size
    codes = torch.randint(0, 1 << bits, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 1 << bits, (groups, out_features), dtype=torch.int32)
    qweight, qzeros = _pack_hand(codes, bits), _pack_hand(zeros, bits)
    scales = torch.rand((groups, out_features), dtype=torch.float16) + 0.5

    expected = packing_utils.dequantize_gemm(qweight, qzeros, scales, bits, group_size)
    monkeypatch.setattr(packing_utils, "_AWQ_DEQUANT_TILE_BYTES", tile_bytes)
    actual = packing_utils.dequantize_gemm(
        qweight.to(device), qzeros.to(device), scales.to(device), bits, group_size
    )

    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(
        actual.cpu(), _reference_weight(codes, zeros, scales, group_size, torch.float16)
    )


def test_missing_zero_points_is_rejected():
    scales = torch.ones((1, 32), dtype=torch.float16)
    qweight = _pack_hand(torch.zeros((64, 32), dtype=torch.int32), 4)

    with pytest.raises(ValueError, match="zero points"):
        packing_utils.dequantize_gemm(qweight, None, scales, 4, 64)


@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_forward_keeps_weights_quantized_between_calls(activation_dtype, device):
    """Two forwards must not leave a dequantized weight on the module."""

    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    torch.manual_seed(5)
    bits, in_features, out_features, group_size = 4, 128, 32, 16
    groups = in_features // group_size
    codes = torch.randint(0, 1 << bits, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 1 << bits, (groups, out_features), dtype=torch.int32)
    qweight, qzeros = _pack_hand(codes, bits), _pack_hand(zeros, bits)
    scales = (torch.randint(1, 5, (groups, out_features)).float() / 4).to(activation_dtype)

    module = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=activation_dtype,
    )
    module.register_buffer("qweight", qweight.to(device))
    module.register_buffer("qzeros", qzeros.to(device))
    module.register_buffer("scales", scales.to(device))
    module.bias = None
    module = module.to(device).eval()
    module.post_init()
    stored = {name: tensor.clone() for name, tensor in module.state_dict().items()}

    x = torch.randint(-2, 3, (2, in_features), device=device).to(activation_dtype)
    with torch.inference_mode():
        first = module(x)
        second = module(x)

    expected_weight = _reference_weight(codes, zeros, scales, group_size, activation_dtype)
    expected = (x.cpu().double() @ expected_weight.double()).to(activation_dtype)
    assert first.dtype == activation_dtype
    assert torch.equal(first.cpu(), expected)
    assert torch.equal(first, second)
    for name, tensor in module.state_dict().items():
        assert torch.equal(tensor, stored[name]), f"{name} was rewritten by forward"
    retained = list(vars(module).values()) + list(module.buffers()) + list(module.parameters())
    assert not any(
        isinstance(value, torch.Tensor) and value.shape == (in_features, out_features)
        for value in retained
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA/ROCm device")
@pytest.mark.parametrize("group_size", [128, 1024, 4096, -1])
def test_temporary_storage_is_bounded_by_the_weight_size(group_size):
    """One call must not allocate several copies of the whole weight.

    The removed formulation peaked at 448 MiB for this shape, fourteen times the
    32 MiB FP16 weight. The bound is 2.5x the weight, which the tiled broadcast
    stays well under and the removed formulation cannot.
    """

    torch.manual_seed(6)
    device = torch.device("cuda:0")
    bits, in_features, out_features = 4, 4096, 4096
    groups = 1 if group_size <= 0 else in_features // group_size
    qweight, qzeros = packing_utils.pack_awq(
        torch.randint(0, 1 << bits, (in_features, out_features), dtype=torch.int32, device=device),
        torch.randint(0, 1 << bits, (groups, out_features), dtype=torch.int32, device=device),
        bits,
    )
    scales = torch.rand((groups, out_features), dtype=torch.float16, device=device) + 0.5
    weight_bytes = in_features * out_features * scales.element_size()

    with torch.inference_mode():
        packing_utils.dequantize_gemm(qweight, qzeros, scales, bits, group_size)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        weight = packing_utils.dequantize_gemm(qweight, qzeros, scales, bits, group_size)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)

    extra = peak - baseline
    assert weight.shape == (in_features, out_features)
    assert extra > 0
    assert extra <= 2.5 * weight_bytes, (
        f"dequantize peak {extra / 2**20:.1f} MiB exceeds 2.5x "
        f"the {weight_bytes / 2**20:.1f} MiB weight"
    )


@pytest.mark.parametrize(
    ("bits", "out_features"), [(2, 8), (2, 24), (3, 8), (3, 40), (8, 8)]
)
def test_narrow_out_features_slices_the_packed_row(bits, out_features, device):
    """A packed row can hold more codes than ``out_features`` asks for."""

    torch.manual_seed(7)
    in_features, group_size = 32, 16
    groups = in_features // group_size
    codes = torch.randint(0, 1 << bits, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 1 << bits, (groups, out_features), dtype=torch.int32)
    qweight, qzeros = _pack_hand(codes, bits), _pack_hand(zeros, bits)
    scales = torch.rand((groups, out_features), dtype=torch.float16) + 0.5

    actual = packing_utils.dequantize_gemm(
        qweight.to(device), qzeros.to(device), scales.to(device), bits, group_size
    )
    expected = _reference_weight(codes, zeros, scales, group_size, torch.float16)

    assert actual.shape == (in_features, out_features)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("tile_bytes", [384, 1024, 64 * 1024])
@pytest.mark.parametrize("in_features", [40, 64])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_trainable_scales_backward_across_group_slices(
    monkeypatch, tile_bytes, in_features, scale_dtype, device
):
    """Ragged group slices must not invalidate tensors saved for backward."""

    torch.manual_seed(8)
    group_size, out_features = 16, 32
    groups = (in_features + group_size - 1) // group_size
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    scales = torch.rand(groups, out_features, dtype=scale_dtype).requires_grad_()
    qweight, qzeros = _pack_hand(codes, 4), _pack_hand(zeros, 4)
    monkeypatch.setattr(packing_utils, "_AWQ_DEQUANT_TILE_BYTES", tile_bytes)

    actual = packing_utils.dequantize_gemm(
        qweight.to(device), qzeros.to(device), scales.to(device), 4, group_size
    )
    expected = _reference_weight(codes, zeros, scales.detach(), group_size, scale_dtype)
    assert torch.equal(actual.cpu(), expected)
    with torch.no_grad():
        inference_weight = packing_utils.dequantize_gemm(
            qweight.to(device), qzeros.to(device), scales.to(device), 4, group_size
        )
    assert not inference_weight.requires_grad
    assert torch.equal(inference_weight.cpu(), expected)
    actual.sum().backward()
    group_ids = torch.arange(in_features) // group_size
    expected_grad = torch.zeros(groups, out_features, dtype=torch.float64)
    expected_grad.index_add_(0, group_ids, codes.double() - zeros.double()[group_ids])
    assert torch.equal(scales.grad, expected_grad.to(scale_dtype))
