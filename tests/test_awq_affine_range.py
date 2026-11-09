# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import FORMAT, AWQConfig


def _stub(bits=4, dynamic=None):
    class Stub:
        qcfg = AWQConfig(bits=bits, group_size=-1, sym=True, dynamic=dynamic)
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor
        _pseudo_quantize_tensor_into = AWQProcessor._pseudo_quantize_tensor_into

    return Stub()


@pytest.mark.parametrize("value", [1.0, -1.0])
def test_asymmetric_one_sided_group_preserves_constant(value):
    stub = _stub()
    stub.qcfg.group_size = 16
    stub.qcfg.sym = False

    actual, _, _ = stub.pseudo_quantize_tensor(
        torch.full((8, 16), value, dtype=torch.float16)
    )

    torch.testing.assert_close(
        actual,
        torch.full_like(actual, value),
        rtol=0,
        atol=torch.finfo(torch.float16).eps,
    )

def test_asymmetric_finite_fp16_extrema_do_not_overflow():
    stub = _stub()
    stub.qcfg.group_size = 16
    stub.qcfg.sym = False
    weight = torch.tensor(
        [-65504.0, 65504.0] * 8,
        dtype=torch.float16,
    ).repeat(8, 1)

    actual, scales, _ = stub.pseudo_quantize_tensor(weight)

    assert actual.isfinite().all()
    assert scales.isfinite().all()
    assert actual.min().item() >= torch.finfo(torch.float16).min
    assert actual.max().item() <= torch.finfo(torch.float16).max


def _boundary_row(kind: str, scale: float, width: int) -> torch.Tensor:
    half_step = torch.tensor(scale / 2, dtype=torch.float16)
    lower = torch.nextafter(half_step, torch.tensor(-torch.inf, dtype=half_step.dtype))
    upper = torch.nextafter(half_step, torch.tensor(torch.inf, dtype=half_step.dtype))
    positive = [lower.item(), half_step.item(), upper.item()]
    negative = [-value for value in reversed(positive)]
    if kind == "negative":
        values = [-65504.0, -65504.0, 0.0, -0.0, *negative]
    elif kind == "positive":
        values = [65504.0, 65504.0, 0.0, -0.0, *positive]
    elif kind == "mixed":
        values = [-65504.0, 65504.0, 0.0, -0.0, *negative, *positive]
    else:
        values = [-7.0, 8.0, 0.0, -0.0, *negative, *positive]
    return torch.tensor((values * ((width + len(values) - 1) // len(values)))[:width], dtype=torch.float16)


def _finite_code_oracle(weight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
    """Choose the nearest representable reconstruction by enumerating all INT4 codes."""
    codes = torch.empty(weight.shape, dtype=torch.int32)
    decoded = torch.empty_like(weight)
    for row in range(weight.shape[0]):
        scale, zero = scales[row].item(), int(zeros[row].item())
        candidates = []
        for code in range(16):
            value = torch.tensor((code - zero) * scale, dtype=torch.float16).item()
            if math.isfinite(value):
                candidates.append((code, value))
        for col in range(weight.shape[1]):
            source = weight[row, col].item()
            # The affine tie rule rounds the zero-centered integer to even.
            code, value = min(
                candidates,
                key=lambda pair: (abs(source - pair[1]), (pair[0] - zero) & 1),
            )
            codes[row, col] = code
            decoded[row, col] = value
    return codes, decoded


def _pack_awq_oracle(codes: torch.Tensor) -> torch.Tensor:
    """Pack complete INT4 words from Python integers in the AWQ wire order."""
    rows = []
    for row in codes.tolist():
        words = []
        for start in range(0, len(row), 8):
            block = row[start:start + 8]
            words.append(sum((block[index] & 15) << (4 * slot)
                             for slot, index in enumerate((0, 2, 4, 6, 1, 3, 5, 7))))
        rows.append(words)
    return torch.tensor(rows, dtype=torch.int64).to(torch.int32)


def _unpack_awq_oracle(packed: torch.Tensor, output_width: int) -> torch.Tensor:
    codes = torch.empty((packed.shape[0], output_width), dtype=torch.int32)
    for row, words in enumerate(packed.tolist()):
        for word_index, signed_word in enumerate(words):
            word = signed_word & 0xFFFFFFFF
            for slot, index in enumerate((0, 2, 4, 6, 1, 3, 5, 7)):
                codes[row, word_index * 8 + index] = (word >> (slot * 4)) & 15
    return codes


def _staged_boundary_weights(format: FORMAT, width: int, outputs: int):
    # Expected metadata comes from the FP16 format bounds, independent of the
    # production qparam helper. One-sided 4368 would overflow at code 15;
    # mixed GEMM needs 8736, while GEMV-fast must fit its FP16 scaled offset.
    mixed_scale = 8180.0 if format in (FORMAT.GEMV_FAST, FORMAT.LLM_AWQ) else 8736.0
    negative_scale = 4352.0 if format in (FORMAT.GEMV_FAST, FORMAT.LLM_AWQ) else 4364.0
    cases = [("negative", negative_scale, 15.0), ("positive", 4364.0, 0.0),
             ("mixed", mixed_scale, 8.0), ("ordinary", 1.0, 7.0)]
    cases = (cases * ((outputs + len(cases) - 1) // len(cases)))[:outputs]
    weight = torch.stack([_boundary_row(kind, scale, width) for kind, scale, _ in cases])
    expected_scales = torch.tensor([[scale] for _, scale, _ in cases], dtype=torch.float16)
    expected_zeros = torch.tensor([[zero] for _, _, zero in cases], dtype=torch.float16)
    return weight, expected_scales, expected_zeros


def test_fp16_affine_boundary_staging_and_gemm_packed_decode_match_oracle():
    stub = _stub()
    stub.qcfg.group_size = 16
    stub.qcfg.sym = False
    stub.qcfg.format = FORMAT.GEMM
    weight, scales_ref, zeros_ref = _staged_boundary_weights(FORMAT.GEMM, 16, 8)
    codes_ref, decoded_ref = _finite_code_oracle(weight, scales_ref, zeros_ref)

    staged, scales, zeros = stub.pseudo_quantize_tensor(weight)
    in_place = torch.empty_like(weight)
    stub._pseudo_quantize_tensor_into(weight, in_place)
    assert torch.equal(scales, scales_ref)
    assert torch.equal(zeros, zeros_ref)
    torch.testing.assert_close(staged, decoded_ref, rtol=0, atol=0)
    torch.testing.assert_close(in_place, decoded_ref, rtol=0, atol=0)
    assert staged.isfinite().all()
    assert (staged.float() - weight.float()).abs().amax() <= 1.1 * scales_ref.amax()

    source = torch.nn.Linear(16, 8, bias=False, dtype=torch.float16)
    source.weight.data.copy_(staged)
    packed = AwqTorchLinear(bits=4, group_size=16, sym=False, desc_act=False,
                            in_features=16, out_features=8, dtype=torch.float16)
    packed.pack(source, scales, zeros)
    assert torch.equal(packed.qweight, _pack_awq_oracle(codes_ref.T))
    assert torch.equal(packed.qzeros, _pack_awq_oracle(zeros_ref.to(torch.int32).T))
    assert torch.equal(packed.scales, scales_ref.T)

    decoded_codes = _unpack_awq_oracle(packed.qweight, 8)
    decoded_zeros = _unpack_awq_oracle(packed.qzeros, 8)
    assert torch.equal(decoded_codes, codes_ref.T)
    assert torch.equal(decoded_zeros, zeros_ref.to(torch.int32).T)
    decoded_packed = ((decoded_codes - decoded_zeros).to(torch.float16)
                      * packed.scales.repeat_interleave(16, dim=0)).T
    torch.testing.assert_close(decoded_packed, decoded_ref, rtol=0, atol=0)
    runtime_weight = packed._dequantize_weight_for_inference(
        torch.empty(1, 16, dtype=torch.float16), dtype=torch.float16,
    )
    torch.testing.assert_close(runtime_weight.T, decoded_ref, rtol=0, atol=0)


def _gemv_fast_positions(input_width: int) -> list[int]:
    inner = [a * 8 + b * 2 + c for b in range(4) for a in range(4) for c in range(2)]
    ordered = [inner[start + offset] for start in range(0, 32, 8)
               for offset in (0, 2, 4, 6, 1, 3, 5, 7)]
    return [base + position for base in range(0, input_width, 32) for position in ordered]


def _pack_gemv_fast_oracle(codes: torch.Tensor) -> torch.Tensor:
    """Reference the GEMV-fast 4-row/32-column permutation with scalar words."""
    output_width, input_width = codes.shape
    positions = _gemv_fast_positions(input_width)
    words = []
    for row in range(0, output_width, 4):
        packed_row = []
        for block in range(0, input_width, 64):
            for position in range(64):
                word = 0
                for lane in range(4):
                    # The layout reshapes four 64-value row tiles into
                    # 64 halfwords, rather than interleaving whole rows.
                    flat_index = position * 4 + lane
                    source_row = row + flat_index // 64
                    source_col = positions[block + flat_index % 64]
                    word |= int(codes[source_row, source_col]) << (lane * 4)
                packed_row.append(word)
        words.append(packed_row)
    return torch.tensor(words, dtype=torch.int64).to(torch.int16)


def _unpack_gemv_fast_oracle(packed: torch.Tensor) -> torch.Tensor:
    input_width = packed.shape[1]
    positions = _gemv_fast_positions(input_width)
    codes = torch.empty((packed.shape[0] * 4, input_width), dtype=torch.int32)
    for group, packed_row in enumerate(packed.tolist()):
        for block in range(0, input_width, 64):
            for position in range(64):
                word = packed_row[block + position] & 0xFFFF
                for lane in range(4):
                    flat_index = position * 4 + lane
                    source_row = group * 4 + flat_index // 64
                    source_col = positions[block + flat_index % 64]
                    codes[source_row, source_col] = (word >> (lane * 4)) & 15
    return codes


@pytest.mark.parametrize("format,kernel", [
    (FORMAT.GEMV_FAST, AwqGEMVFastLinear),
    (FORMAT.LLM_AWQ, LLMAwqLinear),
])
def test_fp16_affine_boundary_gemv_fast_packed_decode_matches_oracle(format, kernel):
    stub = _stub()
    stub.qcfg.group_size = 128
    stub.qcfg.sym = False
    stub.qcfg.format = format
    weight, scales_ref, zeros_ref = _staged_boundary_weights(format, 128, 128)
    codes_ref, decoded_ref = _finite_code_oracle(weight, scales_ref, zeros_ref)
    staged, scales, zeros = stub.pseudo_quantize_tensor(weight)
    assert torch.equal(scales, scales_ref)
    assert torch.equal(zeros, zeros_ref)
    torch.testing.assert_close(staged, decoded_ref, rtol=0, atol=0)

    source = torch.nn.Linear(128, 128, bias=False, dtype=torch.float16)
    source.weight.data.copy_(staged)
    packed = kernel(bits=4, group_size=128, sym=False, desc_act=False,
                    in_features=128, out_features=128)
    packed.pack(source, scales, zeros)
    assert torch.equal(packed.qweight, _pack_gemv_fast_oracle(codes_ref))
    unpacked_codes = _unpack_gemv_fast_oracle(packed.qweight)
    assert torch.equal(unpacked_codes, codes_ref)
    stored_zeros = getattr(packed, packed.zeros_name)[0]
    expected_offsets = (-scales_ref.float() * zeros_ref.float()).half().flatten()
    assert torch.equal(stored_zeros, expected_offsets)
    assert torch.isfinite(packed.scales).all()
    assert torch.isfinite(stored_zeros).all()
    # Native GEMV-fast uses half FMA: code * scale + stored scaled-zero.
    decoded_packed = (unpacked_codes.float() * packed.scales[0].float()[:, None]
                      + stored_zeros.float()[:, None]).half()
    assert torch.isfinite(decoded_packed).all()
    torch.testing.assert_close(decoded_packed, decoded_ref, rtol=0, atol=0)
    assert (decoded_packed.float() - weight.float()).abs().amax() <= 1.1 * scales_ref.amax()
