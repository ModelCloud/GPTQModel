import pytest
import torch

from gptqmodel import BACKEND
from gptqmodel.models import loader
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig


def test_explicit_awq_backend_coerces_unsupported_bfloat16(monkeypatch):
    class FakeAwqKernel:
        SUPPORTS_DTYPES = [torch.float16]
        __name__ = "FakeAwqKernel"

    monkeypatch.setattr(
        loader, "get_kernel_for_backend", lambda *_args, **_kwargs: FakeAwqKernel
    )

    qcfg = QuantizeConfig(
        bits=4, group_size=128, quant_method=METHOD.AWQ, format=FORMAT.GEMM
    )

    dtype = loader._coerce_quantized_awq_dtype(
        backend=BACKEND.GEMM,
        qcfg=qcfg,
        dtype=torch.bfloat16,
    )

    assert dtype == torch.float16


def test_auto_awq_backend_keeps_requested_dtype():
    qcfg = QuantizeConfig(
        bits=4, group_size=128, quant_method=METHOD.AWQ, format=FORMAT.GEMM
    )

    dtype = loader._coerce_quantized_awq_dtype(
        backend=BACKEND.AUTO,
        qcfg=qcfg,
        dtype=torch.bfloat16,
    )

    assert dtype == torch.bfloat16


def test_sglang_awq_backend_keeps_requested_bfloat16():
    qcfg = QuantizeConfig(
        bits=4, group_size=128, quant_method=METHOD.AWQ, format=FORMAT.GEMM
    )

    dtype = loader._coerce_quantized_awq_dtype(
        backend=BACKEND.SGLANG,
        qcfg=qcfg,
        dtype=torch.bfloat16,
    )

    assert dtype == torch.bfloat16


def test_explicit_awq_backend_keeps_supported_bfloat16(monkeypatch):
    class FakeAwqKernel:
        SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]
        __name__ = "FakeAwqKernel"

    monkeypatch.setattr(
        loader, "get_kernel_for_backend", lambda *_args, **_kwargs: FakeAwqKernel
    )

    qcfg = QuantizeConfig(
        bits=4, group_size=128, quant_method=METHOD.AWQ, format=FORMAT.GEMM
    )

    dtype = loader._coerce_quantized_awq_dtype(
        backend=BACKEND.MARLIN,
        qcfg=qcfg,
        dtype=torch.bfloat16,
    )

    assert dtype == torch.bfloat16


def _gemv_qcfg(format_code=FORMAT.GEMV):
    return QuantizeConfig(
        bits=4, group_size=128, quant_method=METHOD.AWQ, format=format_code
    )


@pytest.mark.parametrize(
    "format_code", [FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ]
)
def test_auto_awq_gemv_layout_coerces_bfloat16(format_code):
    # AUTO has not resolved a kernel at load time, but every kernel it can pick
    # for a GEMV layout is fp16-only, so the format decides.
    dtype = loader._coerce_quantized_awq_dtype(
        backend=BACKEND.AUTO,
        qcfg=_gemv_qcfg(format_code),
        dtype=torch.bfloat16,
    )

    assert dtype == torch.float16


def test_auto_awq_gemm_layout_keeps_bfloat16():
    dtype = loader._coerce_quantized_awq_dtype(
        backend=BACKEND.AUTO,
        qcfg=_gemv_qcfg(FORMAT.GEMM),
        dtype=torch.bfloat16,
    )

    assert dtype == torch.bfloat16


def test_explicit_bfloat16_with_awq_gemv_raises_before_load():
    # Respect an explicit dtype instead of silently overriding it, and name the
    # conflict plus both fixes so the caller can act without reading the kernel.
    with pytest.raises(ValueError, match="float16") as excinfo:
        loader._coerce_quantized_awq_dtype(
            backend=BACKEND.AUTO,
            qcfg=_gemv_qcfg(),
            dtype=torch.bfloat16,
            explicit_dtype=True,
        )

    message = str(excinfo.value)
    assert "FORMAT.GEMV" in message
    assert "FORMAT.GEMM" in message


def test_explicit_awq_gemv_backend_rejects_bfloat16():
    with pytest.raises(ValueError, match="AwqGEMVLinear"):
        loader._coerce_quantized_awq_dtype(
            backend=BACKEND.AWQ_GEMV,
            qcfg=_gemv_qcfg(),
            dtype=torch.bfloat16,
            explicit_dtype=True,
        )
