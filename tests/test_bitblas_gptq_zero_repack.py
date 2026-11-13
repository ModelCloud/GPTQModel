# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exercise GPTQ zero-point conversion through the compressed BitBLAS buffers."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.bitblas import BitblasLinear, BitblasQuantizationConfig
from gptqmodel.nn_modules.qlinear.bitblas_awq import AWQBitBlasKernel


def _compress_without_mask(values, source_bits=4, storage_dtype=np.int8):
    """Model BitBLAS's shift/OR contract, including sign propagation.

    Do not mask fields here: that would hide invalid negative input codes.
    Expected bytes below come from an independent Torch construction.
    """
    per_byte = 8 // source_bits
    packed = np.zeros((values.shape[0], values.shape[1] // per_byte), dtype=np.int8)
    for col in range(packed.shape[1]):
        for lane in range(per_byte):
            packed[:, col] |= values[:, col * per_byte + lane] << (source_bits * lane)
    return packed.view(storage_dtype)


@pytest.fixture(params=["standin", "bitblas"])
def compressor(request, monkeypatch):
    if request.param == "bitblas":
        try:
            from bitblas.quantization.utils import general_compress
        except (ImportError, OSError) as exc:
            pytest.skip(f"Optional BitBLAS compressor unavailable: {exc}")
        return general_compress

    # The repacker and buffer installation run unchanged; only the optional
    # dependency is replaced. No CUDA code generation is needed for this test.
    for name in ("bitblas", "bitblas.quantization", "bitblas.quantization.utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setattr(
        sys.modules["bitblas.quantization.utils"], "general_compress", _compress_without_mask, raising=False
    )
    return _compress_without_mask


def _pack_words(codes, bits):
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=codes.device, dtype=torch.int64) * bits
    return (codes.long().reshape(codes.shape[0], -1, per_word) << shifts).sum(-1).int()


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("requires_v2,format_id", [(True, 1), (True, 2), (False, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gptq_repack_preserves_every_zero_byte(compressor, bits, requires_v2, format_id, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm device required")

    maximum = (1 << bits) - 1
    # Every possible byte exercises max->zero neighbors in every lane, mixed
    # codes, and sign bits in the enclosing int32 GPTQ words. Two group rows
    # also guard against inadvertently flattening the group/output layout.
    expected_bytes = torch.arange(256, dtype=torch.int64, device=device)
    expected_bytes = torch.stack((expected_bytes, expected_bytes.flip(0))).to(torch.uint8)
    shifts = torch.arange(8 // bits, device=device) * bits
    zeros = ((expected_bytes.long()[..., None] >> shifts) & maximum).reshape(2, -1)
    stored = (zeros - 1) & maximum if requires_v2 and format_id == 1 else zeros
    k, n = 64, zeros.shape[1]
    weights = torch.arange(k * n, device=device).reshape(k, n).remainder(maximum + 1)
    scales = torch.arange(1, 2 * n + 1, device=device).reshape(2, n).to(dtype) / 1024

    source = type("GPTQSource", (), {"REQUIRES_FORMAT_V2": requires_v2})()
    source.qzeros = _pack_words(stored, bits)
    source.qweight = _pack_words(weights.T.contiguous(), bits).T.contiguous()
    source.scales = scales
    source.bias = None
    source.qzero_format = lambda: format_id

    # Bypass the native operator constructor, but use the real repack,
    # _load_bitblas_quant_state, and _compress_bitblas_zeros methods.
    module = object.__new__(BitblasLinear)
    torch.nn.Module.__init__(module)
    module.bits = bits
    module.TORCH_DTYPE = dtype
    module.quant_config = BitblasQuantizationConfig(bits, 32, False, False, torch_dtype=dtype)
    module.bitblas_matmul = SimpleNamespace(weight_transform=None)
    module.bias = None
    module.register_buffer("qweight", torch.empty(0, device=device, dtype=torch.int8))
    module.register_buffer("scales", torch.empty(0, device=device, dtype=dtype))
    module.register_buffer("qzeros", torch.empty(0, device=device, dtype=torch.int8))

    for _ in range(2):
        module.repack_from_gptq(source)
        if device == "cuda":
            torch.cuda.synchronize()
        assert module.qzeros.dtype == torch.int8
        assert module.qzeros.device == source.qzeros.device
        assert torch.equal(module.qzeros.view(torch.uint8), expected_bytes)
        assert module.zeros is module.qzeros
        assert torch.equal(module.scales, scales.T.contiguous())
        grouped_weights = weights.T.contiguous().reshape(n, -1, 8 // bits)
        expected_weights = (grouped_weights << shifts).sum(-1).to(torch.uint8)
        assert torch.equal(module.qweight.view(torch.uint8), expected_weights)
        assert set(module.state_dict()) == {"qweight", "scales", "qzeros"}
        assert module.qweight.numel() == k * n * bits // 8


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_pack_affine_overflow_and_endpoints(compressor, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm device required")
    module = object.__new__(AWQBitBlasKernel)
    torch.nn.Module.__init__(module)
    module.bits, module.group_size, module.in_features = 4, 128, 128
    module.TORCH_DTYPE = dtype
    module.quant_config = BitblasQuantizationConfig(4, 128, False, False, torch_dtype=dtype)
    module.bitblas_matmul = SimpleNamespace(weight_transform=None)
    module.bias = None
    module.register_buffer("qweight", torch.empty(0, dtype=torch.int8, device=device))
    module.register_buffer("scales", torch.empty(0, dtype=dtype, device=device))
    module.register_buffer("qzeros", torch.empty(0, dtype=torch.int8, device=device))
    linear = torch.nn.Linear(128, 64, bias=False, dtype=dtype, device=device)
    scales = torch.full((64, 1), 5000., dtype=dtype, device=device)
    zeros = torch.full((64, 1), 8, dtype=torch.int32, device=device)
    samples = torch.tensor([-65504., -40000., 0., 30000., 65504.], dtype=dtype, device=device)
    with torch.no_grad():
        linear.weight.copy_(samples.repeat(1639)[:8192].reshape(64, 128))
    expected = ((linear.weight.detach().double() + scales.double() * zeros.double()) / scales.double())
    codes = expected.round().clamp(0, 15).long()
    expected_bytes = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    for _ in range(2):
        module.pack(linear, scales, zeros)
        if device == "cuda":
            torch.cuda.synchronize()
        assert torch.equal(module.qweight.view(torch.uint8), expected_bytes)
        assert torch.equal(module.qzeros.view(torch.uint8), torch.full((1, 32), 0x88, dtype=torch.uint8, device=device))
