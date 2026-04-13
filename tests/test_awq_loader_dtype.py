import torch

from gptqmodel import BACKEND
from gptqmodel.models import loader
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig


def test_explicit_awq_backend_coerces_unsupported_bfloat16(monkeypatch):
    class FakeAwqKernel:
        SUPPORTS_DTYPES = [torch.float16]
        __name__ = "FakeAwqKernel"

    monkeypatch.setattr(loader, "get_kernel_for_backend", lambda *_args, **_kwargs: FakeAwqKernel)

    qcfg = QuantizeConfig(bits=4, group_size=128, quant_method=METHOD.AWQ, format=FORMAT.GEMM)

    dtype = loader._coerce_quantized_awq_dtype(
        backend=BACKEND.GEMM,
        qcfg=qcfg,
        dtype=torch.bfloat16,
    )

    assert dtype == torch.float16


def test_auto_awq_backend_keeps_requested_dtype():
    qcfg = QuantizeConfig(bits=4, group_size=128, quant_method=METHOD.AWQ, format=FORMAT.GEMM)

    dtype = loader._coerce_quantized_awq_dtype(
        backend=BACKEND.AUTO,
        qcfg=qcfg,
        dtype=torch.bfloat16,
    )

    assert dtype == torch.bfloat16


def test_explicit_awq_backend_keeps_supported_bfloat16(monkeypatch):
    class FakeAwqKernel:
        SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]
        __name__ = "FakeAwqKernel"

    monkeypatch.setattr(loader, "get_kernel_for_backend", lambda *_args, **_kwargs: FakeAwqKernel)

    qcfg = QuantizeConfig(bits=4, group_size=128, quant_method=METHOD.AWQ, format=FORMAT.GEMM)

    dtype = loader._coerce_quantized_awq_dtype(
        backend=BACKEND.MARLIN,
        qcfg=qcfg,
        dtype=torch.bfloat16,
    )

    assert dtype == torch.bfloat16
