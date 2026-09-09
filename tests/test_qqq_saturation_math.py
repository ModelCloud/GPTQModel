"""Exhaustive arithmetic audit of the proposed QQQ magic-conversion clamp.

No production CUDA change is implied. Float64 exactly represents these
integer-times-half products and sums, so a final half cast models the FMA.
"""

import torch


def test_post_fma_clamp_matches_int8_over_all_positive_finite_half_scales():
    # Positive finite FP16 bit patterns, including subnormals and max finite.
    scales = torch.arange(1, 0x7c00, dtype=torch.int16).view(torch.float16).double()
    signed_codes = torch.arange(-8, 8, dtype=torch.float64).reshape(-1, 1)
    exact_product = signed_codes * scales
    rounded_magic = (exact_product + 1152.0).half()
    corrected_magic = rounded_magic.clamp(1024.0, 1279.0)
    bits = corrected_magic.view(torch.int16).to(torch.int32)
    unsigned_byte = (bits & 255) ^ 128
    decoded = torch.where(unsigned_byte >= 128, unsigned_byte - 256, unsigned_byte)
    expected = exact_product.round().clamp(-128, 127).to(torch.int32)
    assert torch.equal(decoded, expected)
    assert decoded.numel() == 16 * (0x7c00 - 1)
    # At products below -128, the magic value crosses a half binade and
    # rounding spacing changes. Clamp fixes that boundary too.
    interior = (exact_product >= -128) & (exact_product <= 127)
    assert torch.equal(corrected_magic[interior], rounded_magic[interior])


def test_existing_magic_conversion_reproduces_native_endpoint_failures():
    product = torch.tensor([7.0 * 22.0, -8.0 * 20.0])
    bits = (product + 1152).half().view(torch.int16).to(torch.int32)
    unsigned_byte = (bits & 255) ^ 128
    existing = torch.where(unsigned_byte >= 128, unsigned_byte - 256, unsigned_byte)
    assert torch.equal(existing, torch.tensor([-102, 64]))
    assert torch.equal(product.round().clamp(-128, 127), torch.tensor([127.0, -128.0]))
