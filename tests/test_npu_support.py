import pytest
import torch

from gptqmodel.models._const import DEVICE, normalize_device
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch import _right_shift_unpack
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.awq.utils.packing_utils import unpack_awq
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import auto_select_device, select_quant_linear


def test_npu_device_normalization():
    assert normalize_device("npu") is DEVICE.NPU
    assert normalize_device("npu:3") is DEVICE.NPU
    assert DEVICE.NPU.type == "npu"
    try:
        expected = torch.device("npu:0")
    except (RuntimeError, ValueError):
        pytest.skip("This PyTorch build does not register the npu device type")
    assert DEVICE.NPU.to_torch_device() == expected


def test_auto_select_device_uses_npu_when_available(monkeypatch):
    monkeypatch.setattr(importer, "HAS_CUDA", False)
    monkeypatch.setattr(importer, "HAS_XPU", False)
    monkeypatch.setattr(importer, "HAS_NPU", True)
    monkeypatch.setattr(importer, "HAS_MPS", False)

    assert auto_select_device(None, BACKEND.AUTO) is DEVICE.NPU


@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
def test_npu_auto_selects_torch_gptq(fmt):
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is TorchLinear


def test_npu_auto_selects_torch_awq_for_gemm():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is AwqTorchLinear


@pytest.mark.skipif(not (hasattr(torch, "npu") and torch.npu.is_available()), reason="NPU is not available")
def test_npu_awq_unpack_preserves_pack_dimension():
    qweight = torch.randint(0, 2**30, (2, 3), dtype=torch.int32, device="npu:0")
    qzeros = torch.randint(0, 2**30, (2, 3), dtype=torch.int32, device="npu:0")

    iweight, izeros = unpack_awq(qweight, qzeros, bits=4)

    assert iweight.shape == (2, 24)
    assert izeros.shape == (2, 24)
    assert iweight.device.type == "npu"
    assert izeros.device.type == "npu"


@pytest.mark.skipif(not (hasattr(torch, "npu") and torch.npu.is_available()), reason="NPU is not available")
def test_npu_torch_gptq_unpack_preserves_pack_dimension():
    qweight = torch.randint(0, 2**30, (4, 3), dtype=torch.int32, device="npu:0")
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device="npu:0").view(1, 8, 1)

    unpacked = _right_shift_unpack(
        qweight.unsqueeze(1).expand(-1, 8, -1),
        shifts,
        torch.int8,
    )

    assert unpacked.shape == (4, 8, 3)
    assert unpacked.device.type == "npu"
