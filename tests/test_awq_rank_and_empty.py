from types import SimpleNamespace

import pytest
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as gemm_awq
import gptqmodel.nn_modules.qlinear.gemm_awq_triton as gemm_awq_triton
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from gptqmodel.nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
from gptqmodel.nn_modules.qlinear import empty_linear_output, input_rows
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.nn_modules.qlinear.swordfish import AwqSwordfishLinear, SwordfishLinear


def _fake_quant_tensors(in_features=8, out_features=8):
    return (
        torch.ones((in_features, out_features // 8), dtype=torch.int32),
        torch.ones((in_features, out_features), dtype=torch.float16),
        torch.zeros((in_features, out_features // 8), dtype=torch.int32),
    )


def _patch_backend(monkeypatch, backend, calls):
    if backend == "triton":
        triton_state = getattr(
            gemm_awq_triton, "tritonv2", SimpleNamespace(TRITON_AVAILABLE=False)
        )
        monkeypatch.setattr(gemm_awq_triton, "tritonv2", triton_state, raising=False)
        monkeypatch.setattr(triton_state, "TRITON_AVAILABLE", True)

        def fake_dequant(qweight, scales, qzeros):
            calls["dequant"] += 1
            return torch.ones(
                (qweight.shape[0], qweight.shape[1] * 8), dtype=torch.float16
            )

        def fake_gemm(x, qweight, scales, qzeros, **kwargs):
            calls["gemm"] += 1
            return torch.ones(
                (x.shape[0], qweight.shape[1] * 8), dtype=x.dtype, device=x.device
            )

        monkeypatch.setattr(
            gemm_awq_triton, "awq_dequantize_triton", fake_dequant, raising=False
        )
        monkeypatch.setattr(
            gemm_awq_triton, "awq_gemm_triton", fake_gemm, raising=False
        )
        monkeypatch.setattr(
            "gptqmodel.quantization.awq.modules.triton.gemm.awq_dequantize_triton",
            fake_dequant,
            raising=False,
        )
        monkeypatch.setattr(
            "gptqmodel.quantization.awq.modules.triton.gemm.awq_gemm_triton",
            fake_gemm,
            raising=False,
        )
        return gemm_awq_triton.AwqGemmTritonFn

    def fake_dequant(qweight, scales, qzeros, *_args):
        calls["dequant"] += 1
        return torch.ones((qweight.shape[0], qweight.shape[1] * 8), dtype=torch.float16)

    def fake_gemm(x, qweight, scales, qzeros, *_args, **_kwargs):
        calls["gemm"] += 1
        return torch.ones(
            (x.shape[0], qweight.shape[1] * 8), dtype=x.dtype, device=x.device
        )

    monkeypatch.setattr(gemm_awq, "awq_dequantize_weights", fake_dequant)
    monkeypatch.setattr(gemm_awq, "_awq_cuda_gemm_forward", fake_gemm)
    return gemm_awq.AwqGemmFn


@pytest.mark.parametrize("backend", ["jit", "triton"])
@pytest.mark.parametrize("shape", [(2, 8), (2, 3, 8), (2, 3, 4, 8)])
def test_awq_gemm_autograd_supports_arbitrary_rank(monkeypatch, backend, shape):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)
    qweight, scales, qzeros = _fake_quant_tensors()
    x = torch.ones(shape, dtype=torch.float16, requires_grad=True)

    out = fn.apply(x, qweight, qzeros, scales, 4, 8, None, 8)
    assert out.shape == shape[:-1] + (8,)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.all(x.grad == 8)
    assert calls["gemm"] == 1
    assert calls["dequant"] == 1


@pytest.mark.parametrize("backend", ["jit", "triton"])
@pytest.mark.parametrize("shape", [(3, 0, 8), (0, 3, 8)])
def test_awq_gemm_empty_input_skips_forward_and_backward_backends(
    monkeypatch, backend, shape
):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)
    qweight, scales, qzeros = _fake_quant_tensors()
    x = torch.empty(shape, dtype=torch.float16, requires_grad=True)

    out = fn.apply(x, qweight, qzeros, scales, 4, 8, None, 8)
    assert out.shape == shape[:-1] + (8,)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert calls == {"dequant": 0, "gemm": 0}


@pytest.mark.parametrize("backend, rows", [("jit", 1025), ("triton", 129)])
def test_awq_gemm_heuristic_uses_logical_rows_for_2d_input(monkeypatch, backend, rows):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)
    qweight, scales, qzeros = _fake_quant_tensors()
    x = torch.ones((rows, 8), dtype=torch.float16)

    out = fn.apply(x, qweight, qzeros, scales, 4, 8, None, 8)
    assert out.shape == (rows, 8)
    assert calls == {"dequant": 1, "gemm": 0}


@pytest.mark.parametrize("shape", [(3, 0, 8), (0, 3, 8), (2, 3, 4, 8)])
def test_linear_shape_helpers_preserve_rank_dtype_and_device(shape):
    x = torch.empty(shape, dtype=torch.bfloat16)
    assert input_rows(x) == x.numel() // x.shape[-1]
    out = empty_linear_output(x, 11)
    assert out.shape == shape[:-1] + (11,)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_empty_linear_output_keeps_autograd_connection():
    x = torch.empty((2, 0, 8), dtype=torch.float32, requires_grad=True)
    out = empty_linear_output(x, 11)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.count_nonzero(x.grad) == 0


@pytest.mark.parametrize(
    "linear_cls",
    [
        AwqGEMMLinear,
        AwqGEMMTritonLinear,
        MarlinLinear,
        AwqMarlinLinear,
        ExllamaV2Linear,
        AwqExllamaV2Linear,
        MacheteLinear,
        AwqMacheteLinear,
        QQQLinear,
        QQQTorchLinear,
        SwordfishLinear,
        AwqSwordfishLinear,
        AwqGEMVLinear,
        AwqGEMVFastLinear,
        ParoLinear,
        ParoQuantTritonLinear,
    ],
)
def test_representative_qlinear_empty_shape_without_native_extension(linear_cls):
    # These forwards all check the common helper before touching backend state;
    # constructing a minimal shell keeps this test independent of extensions.
    module = object.__new__(linear_cls)
    module.out_features = 11
    module.adapter = None
    module.training = True
    if linear_cls in (AwqGEMMLinear, AwqGEMMTritonLinear):
        module.bits = 4
        module.group_size = 8
        module.fp32_accum = True
        module.qweight = torch.empty((8, 2), dtype=torch.int32)
        module.qzeros = torch.empty((1, 2), dtype=torch.int32)
        module.scales = torch.empty((1, 11), dtype=torch.float16)
        module.bias = None
    x = torch.empty((2, 0, 8), dtype=torch.float32, requires_grad=True)

    out = module.forward(x)
    assert out.shape == (2, 0, 11)
    assert out.dtype == x.dtype
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_awq_gemv_validation_matches_native_kernel_contract():
    ok, error = AwqGEMVLinear.validate(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        pack_dtype=torch.int32,
    )
    assert not ok
    assert isinstance(error, NotImplementedError)

    ok, error = AwqGEMVLinear.validate(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
    )
    assert ok and error is None

    # Positional arguments cover the constructor-style call that originally
    # exposed duplicate group_size forwarding when resolving -1.
    ok, error = AwqGEMVLinear.validate(
        4,
        -1,
        False,
        True,
        128,
        64,
        torch.int32,
    )
    assert ok and error is None

    ok, error = AwqGEMVLinear.validate(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=256,
        out_features=64,
        pack_dtype=torch.int32,
    )
    assert not ok
    assert isinstance(error, NotImplementedError)

    ok, error = AwqGEMVLinear.validate(
        bits=4,
        group_size=64,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=32,
        pack_dtype=torch.int32,
    )
    assert not ok
    assert isinstance(error, NotImplementedError)
