import numpy as np
import pytest
import torch

import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.bitblas_awq import AWQBitBlasKernel
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import get_kernel_for_backend, select_quant_linear


def _compress_ints(lowprecision_weight: torch.Tensor, bits: int) -> torch.Tensor:
    values = lowprecision_weight.detach().cpu().numpy().astype(np.int8, copy=False)
    elems_per_byte = 8 // bits
    packed = np.zeros(
        (*values.shape[:-1], values.shape[-1] // elems_per_byte),
        dtype=np.int8,
    )
    for col in range(packed.shape[-1]):
        for lane in range(elems_per_byte):
            packed[:, col] |= values[:, col * elems_per_byte + lane] << (bits * lane)
    return torch.from_numpy(packed)


def _pack_awq(iweights_in_out: torch.Tensor, izeros_group_out: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    pack_num = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    qweight = torch.zeros(
        (iweights_in_out.shape[0], iweights_in_out.shape[1] // pack_num),
        dtype=torch.int32,
    )
    for col in range(qweight.shape[1]):
        for lane, mapped_col in enumerate(order_map):
            qweight[:, col] |= iweights_in_out[:, col * pack_num + mapped_col].to(torch.int32) << (lane * bits)

    qzeros = torch.zeros(
        (izeros_group_out.shape[0], izeros_group_out.shape[1] // pack_num),
        dtype=torch.int32,
    )
    for col in range(qzeros.shape[1]):
        for lane, mapped_col in enumerate(order_map):
            qzeros[:, col] |= izeros_group_out[:, col * pack_num + mapped_col].to(torch.int32) << (lane * bits)

    return qweight, qzeros


def _install_dummy_bitblas(monkeypatch):
    captured = {}

    class _DummyMatmul:
        def __init__(self, config):
            self.config = config
            self.lib = object()
            self.weight_transform = None

        @staticmethod
        def retrieve_weight_shape():
            return (1, 1)

    def _fake_get_or_create(self, config, enable_tuning):
        captured["config"] = config
        captured["enable_tuning"] = enable_tuning
        return _DummyMatmul(config)

    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(bitblas_module, "import_bitblas", lambda: None)
    monkeypatch.setattr(AWQBitBlasKernel, "_get_or_create_bitblas_operator", _fake_get_or_create)
    AWQBitBlasKernel.cached_validate_once.cache_clear()

    return captured


@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_awq_bitblas_selects_bitblas_awq_for_awq_gemm(monkeypatch):
    _install_dummy_bitblas(monkeypatch)

    selected = select_quant_linear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        backend=BACKEND.AWQ_BITBLAS,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        device=DEVICE.CUDA,
        pack=False,
        pack_dtype=torch.int32,
    )

    assert selected is AWQBitBlasKernel


@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_awq_bitblas_kernel_mapping_uses_awq_backend():
    assert get_kernel_for_backend(BACKEND.AWQ_BITBLAS, METHOD.AWQ, FORMAT.BITBLAS) is AWQBitBlasKernel


@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_awq_bitblas_kernel_mapping_rejects_gptq_bitblas_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        get_kernel_for_backend(BACKEND.GPTQ_BITBLAS, METHOD.AWQ, FORMAT.BITBLAS)


@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_awq_bitblas_uses_unsigned_weights_and_qzeros(monkeypatch):
    captured = _install_dummy_bitblas(monkeypatch)

    AWQBitBlasKernel(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=32,
        out_features=32,
        bias=False,
    )

    assert captured["config"].W_dtype == "uint4"
    assert captured["config"].with_zeros is True
    assert captured["config"].zeros_mode == "quantized"


@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_awq_bitblas_repack_from_awq_preserves_codes(monkeypatch):
    _install_dummy_bitblas(monkeypatch)

    bits = 4
    group_size = 32
    in_features = 64
    out_features = 32
    groups = in_features // group_size

    intweight = torch.randint(0, 2**bits, (out_features, in_features), dtype=torch.int32)
    intzeros = torch.randint(0, 2**bits, (groups, out_features), dtype=torch.int32)
    scales = torch.rand(groups, out_features, dtype=torch.float32) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)
    qweight, qzeros = _pack_awq(intweight.t().contiguous(), intzeros, bits)

    awq_module = AwqGEMMLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    )
    awq_module.qweight.copy_(qweight)
    awq_module.qzeros.copy_(qzeros)
    awq_module.scales.copy_(scales.to(torch.float16))
    awq_module.bias.copy_(bias)

    bitblas_module_instance = AWQBitBlasKernel(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
    )
    bitblas_module_instance.repack_from_awq(awq_module)

    torch.testing.assert_close(bitblas_module_instance.qweight.cpu(), _compress_ints(intweight, bits))
    torch.testing.assert_close(bitblas_module_instance.qzeros.cpu(), _compress_ints(intzeros, bits))
    torch.testing.assert_close(bitblas_module_instance.scales.cpu(), scales.t().to(torch.float16))
    torch.testing.assert_close(bitblas_module_instance.bias.cpu(), bias)


@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_quantize_config_allows_awq_bitblas():
    cfg = QuantizeConfig(
        bits=4,
        group_size=128,
        quant_method=METHOD.AWQ,
        format=FORMAT.BITBLAS,
    )

    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.BITBLAS
