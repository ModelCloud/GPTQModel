# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.nn_modules.qlinear import fp8


FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)
OUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
BITS_DTYPE = {
    torch.float16: torch.uint16,
    torch.bfloat16: torch.uint16,
    torch.float32: torch.uint32,
}


def _reference(fp8_dtype, out_dtype):
    codes = torch.arange(256, dtype=torch.uint8).view(fp8_dtype)
    return codes.to(out_dtype)


def _edit_bits(tensor, mask, xor):
    bits_dtype = BITS_DTYPE[tensor.dtype]
    raw = tensor.view(bits_dtype).to(torch.int64)
    raw = torch.where(mask, raw ^ xor, raw)
    return raw.to(bits_dtype).view(tensor.dtype)


def _other_nan_payload(tensor):
    """Same values, but every NaN carries a different payload (bit 1 flipped)."""

    return _edit_bits(tensor, torch.isnan(tensor), 2)


def test_nan_payload_differences_are_tolerated():
    checked = 0
    for fp8_dtype in FP8_DTYPES:
        for out_dtype in OUT_DTYPES:
            expected = _reference(fp8_dtype, out_dtype)
            if not bool(torch.isnan(expected).any()):
                continue
            actual = _other_nan_payload(expected)
            assert not torch.equal(
                actual.view(torch.uint8), expected.view(torch.uint8)
            )
            assert torch.equal(torch.isnan(actual), torch.isnan(expected))
            verdict = fp8._compare_fp8_casts(expected, actual)
            assert verdict.dtype is torch.bool
            assert verdict.numel() == 1
            assert bool(verdict), (fp8_dtype, out_dtype)
            checked += 1
    assert checked == len(FP8_DTYPES) * len(OUT_DTYPES)


def test_non_nan_bit_difference_is_rejected():
    for fp8_dtype in FP8_DTYPES:
        for out_dtype in OUT_DTYPES:
            expected = _reference(fp8_dtype, out_dtype)
            finite = ~torch.isnan(expected)
            mask = torch.zeros_like(finite)
            mask[finite.nonzero()[0]] = True
            actual = _edit_bits(expected, mask, 1)
            assert not bool(fp8._compare_fp8_casts(expected, actual))


def test_nan_turned_into_a_number_is_rejected():
    for fp8_dtype in FP8_DTYPES:
        for out_dtype in OUT_DTYPES:
            expected = _reference(fp8_dtype, out_dtype)
            nan = torch.isnan(expected)
            if not bool(nan.any()):
                continue
            actual = expected.clone()
            actual[nan] = 0.0
            assert not bool(fp8._compare_fp8_casts(expected, actual))


def test_number_turned_into_nan_is_rejected():
    for fp8_dtype in FP8_DTYPES:
        for out_dtype in OUT_DTYPES:
            expected = _reference(fp8_dtype, out_dtype)
            finite = ~torch.isnan(expected)
            mask = torch.zeros_like(finite)
            mask[finite.nonzero()[0]] = True
            actual = expected.clone()
            actual[mask] = float("nan")
            assert not bool(fp8._compare_fp8_casts(expected, actual))


def test_legacy_raw_bit_comparison_rejected_the_rocm_cast():
    # #5570: the pre-fix probe compared every bit and returned False for a
    # ROCm cast that only differs in NaN payloads.
    expected = _reference(torch.float8_e5m2, torch.bfloat16)
    actual = _other_nan_payload(expected)
    assert not bool((actual.view(torch.uint8) == expected.view(torch.uint8)).all())
    assert bool(fp8._compare_fp8_casts(expected, actual))


def test_probe_uses_the_nan_aware_comparison(monkeypatch):
    seen = []

    def _compare(expected, actual):
        seen.append((expected.dtype, actual.dtype))
        return torch.tensor(True)

    monkeypatch.setattr(fp8, "_compare_fp8_casts", _compare)
    verdict = fp8._probe_native_fp8_cast(
        torch.device("cpu"), torch.float8_e4m3fn, torch.bfloat16
    )
    assert bool(verdict)
    assert seen == [(torch.bfloat16, torch.bfloat16)]
