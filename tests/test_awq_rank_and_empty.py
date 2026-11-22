import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as gemm_awq
import gptqmodel.nn_modules.qlinear.gemm_awq_triton as gemm_awq_triton
from gptqmodel.nn_modules.qlinear import empty_linear_output, input_rows
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from gptqmodel.nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.nn_modules.qlinear.swordfish import AwqSwordfishLinear, SwordfishLinear
from gptqmodel.utils.awq import awq_runtime_available
from gptqmodel.utils.kernel import KernelConfig


# ParoTritonLinear imports Triton kernels at module scope; every other class
# below imports on a CPU-only host.
HAS_TRITON = importlib.util.find_spec("triton") is not None
if HAS_TRITON:
    from gptqmodel.nn_modules.qlinear.paro_triton import ParoTritonLinear
else:
    ParoTritonLinear = None


_TRITON_GEMM_MODULE = "gptqmodel.quantization.awq.modules.triton.gemm"


def _triton_gemm_module(monkeypatch):
    """Return the module ``AwqGemmTritonFn`` imports its kernels from.

    The dispatch under test replaces both kernel entry points with fakes, so
    it does not need Triton. On a host without Triton the real module cannot
    be imported; stand in an empty module for the duration of the test.
    """

    if HAS_TRITON:
        return importlib.import_module(_TRITON_GEMM_MODULE)
    stub = types.ModuleType(_TRITON_GEMM_MODULE)
    monkeypatch.setitem(sys.modules, _TRITON_GEMM_MODULE, stub)
    return stub


def _fake_quant_tensors(in_features=32, out_features=8, group_size=32):
    groups = in_features // group_size
    return (
        torch.ones((in_features, out_features // 8), dtype=torch.int32),
        torch.ones((groups, out_features), dtype=torch.float16),
        torch.zeros((groups, out_features // 8), dtype=torch.int32),
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
        triton_gemm = _triton_gemm_module(monkeypatch)
        monkeypatch.setattr(triton_gemm, "awq_dequantize_triton", fake_dequant, raising=False)
        monkeypatch.setattr(triton_gemm, "awq_gemm_triton", fake_gemm, raising=False)
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



def test_awq_triton_declared_shapes_match_kernel_contract():
    assert AwqGEMMTritonLinear.SUPPORTS_GROUP_SIZE == [-1, 32, 64, 128]
    assert AwqGEMMTritonLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY == [32]
    assert AwqGEMMTritonLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY == [8]


@pytest.mark.parametrize("backend", ["jit", "triton"])
@pytest.mark.parametrize("shape", [(2, 32), (2, 3, 32), (2, 3, 4, 32)])
def test_awq_gemm_autograd_supports_arbitrary_rank(monkeypatch, backend, shape):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)
    qweight, scales, qzeros = _fake_quant_tensors()
    x = torch.ones(shape, dtype=torch.float16, requires_grad=True)

    out = fn.apply(x, qweight, qzeros, scales, 4, 32, None, 8)
    assert out.shape == shape[:-1] + (8,)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.all(x.grad == 8)
    assert calls["gemm"] == 1
    assert calls["dequant"] == 1


@pytest.mark.parametrize("backend", ["jit", "triton"])
@pytest.mark.parametrize("shape", [(0, 8), (3, 0, 8), (0, 3, 8), (2, 3, 0, 8)])
def test_awq_gemm_empty_input_skips_forward_and_backward_backends(
    monkeypatch, backend, shape
):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)
    qweight, scales, qzeros = _fake_quant_tensors()
    x = torch.empty(shape, dtype=torch.float16, requires_grad=True)

    out = fn.apply(x, qweight, qzeros, scales, 4, 32, None, 8)
    assert out.shape == shape[:-1] + (8,)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert calls == {"dequant": 0, "gemm": 0}


@pytest.mark.parametrize(
    "backend, shape, expect_dequant",
    [
        # The JIT path switches at rows >= 640 when fp32 accumulation is on,
        # and keeps the historical rows > 1024 dispatch otherwise.
        ("jit", (639, 32), False),
        ("jit", (640, 32), True),
        ("jit", (1024, 32), True),
        ("jit", (1025, 32), True),
        ("jit", (32, 32, 32), True),
        ("triton", (127, 32), False),
        ("triton", (128, 32), False),
        ("triton", (129, 32), True),
        ("triton", (8, 16, 32), False),
    ],
)
def test_awq_gemm_heuristic_uses_logical_rows_at_threshold(
    monkeypatch, backend, shape, expect_dequant
):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)
    qweight, scales, qzeros = _fake_quant_tensors()
    x = torch.ones(shape, dtype=torch.float16)

    out = fn.apply(x, qweight, qzeros, scales, 4, 32, None, 8)
    assert out.shape == shape[:-1] + (8,)
    assert calls == {
        "dequant": int(expect_dequant),
        "gemm": int(not expect_dequant),
    }


@pytest.mark.parametrize(
    "shape, expect_dequant",
    [
        ((639, 32), False),
        ((640, 32), False),
        ((1024, 32), False),
        ((1025, 32), True),
    ],
)
def test_awq_jit_heuristic_respects_fp32_accum_opt_out(
    monkeypatch, shape, expect_dequant
):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, "jit", calls)
    qweight, scales, qzeros = _fake_quant_tensors()
    x = torch.ones(shape, dtype=torch.float16)

    fn.apply(x, qweight, qzeros, scales, 4, 32, None, 8, "cuda", False)

    assert calls == {
        "dequant": int(expect_dequant),
        "gemm": int(not expect_dequant),
    }


@pytest.mark.parametrize("backend", ["jit", "triton"])
@pytest.mark.parametrize("shape", [(2, 3, 32), (2, 3, 4, 32)])
def test_awq_gemm_inference_does_not_clone_batched_output(monkeypatch, backend, shape):
    """An untracked batched forward must not copy the full output (#5086).

    #5001 materialized the leading-dimension reshape unconditionally, so every
    batched inference forward paid for one extra output copy even though
    ``inference_mode`` never tracks the result. Only a grad-tracked result has
    to be materialized (#4974).
    """

    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)
    qweight, scales, qzeros = _fake_quant_tensors()

    original = torch.Tensor.clone
    clones = {"n": 0}

    def counting_clone(self, *args, **kwargs):
        clones["n"] += 1
        return original(self, *args, **kwargs)

    torch.Tensor.clone = counting_clone
    try:
        with torch.inference_mode():
            x = torch.ones(shape, dtype=torch.float16)
            out = fn.apply(x, qweight, qzeros, scales, 4, 32, None, 8)
    finally:
        torch.Tensor.clone = original

    assert out.shape == shape[:-1] + (8,)
    assert clones["n"] == 0


@pytest.mark.parametrize("backend", ["jit", "triton"])
def test_awq_gemm_tracked_batched_output_is_owned(monkeypatch, backend):
    """The #4974 guarantee: a tracked batched output must not be a view."""

    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)
    qweight, scales, qzeros = _fake_quant_tensors()
    x = torch.ones((2, 3, 32), dtype=torch.float16, requires_grad=True)

    out = fn.apply(x, qweight, qzeros, scales, 4, 32, None, 8)

    assert out.shape == (2, 3, 8)
    assert not out._is_view()
    out += 1.0
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def _awq_gemm_module(monkeypatch, backend):
    calls = {"dequant": 0, "gemm": 0}
    _patch_backend(monkeypatch, backend, calls)
    if backend == "triton":
        # Keep the module on the portable dequantise-then-GEMM branch so the
        # test exercises the shared Function epilogue without a native device.
        monkeypatch.setattr(
            gemm_awq_triton,
            "select_awq_triton_plan",
            lambda **kwargs: SimpleNamespace(fused=False),
        )
        monkeypatch.setattr(
            gemm_awq_triton, "mark_awq_triton_plan_warmed", lambda *args, **kwargs: None
        )
        cls = AwqGEMMTritonLinear
    else:
        cls = AwqGEMMLinear
    in_features, out_features, group_size = 128, 64, 32
    module = cls(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    module.qweight.copy_(
        torch.ones((in_features, out_features // 8), dtype=torch.int32)
    )
    module.qzeros.copy_(
        torch.zeros(
            (in_features // group_size, out_features // 8), dtype=torch.int32
        )
    )
    module.scales.copy_(
        torch.ones((in_features // group_size, out_features), dtype=torch.float16)
    )
    module.post_init()
    return module.eval()


@pytest.mark.parametrize("backend", ["jit", "triton"])
def test_awq_gemm_module_inference_does_not_clone_batched_output(
    monkeypatch, backend
):
    """The user-visible path: ``AwqGEMM*Linear`` under ``inference_mode``."""

    if backend == "jit" and (
        not torch.cuda.is_available() or not awq_runtime_available()
    ):
        pytest.skip("AwqGEMMLinear needs the native CUDA AWQ extension")
    if backend == "triton" and not HAS_TRITON:
        pytest.skip("AwqGEMMTritonLinear construction validates the installed triton")
    module = _awq_gemm_module(monkeypatch, backend)

    original = torch.Tensor.clone
    clones = {"n": 0}

    def counting_clone(self, *args, **kwargs):
        clones["n"] += 1
        return original(self, *args, **kwargs)

    torch.Tensor.clone = counting_clone
    try:
        with torch.inference_mode():
            x = torch.ones((2, 3, 128), dtype=torch.float16)
            out = module(x)
    finally:
        torch.Tensor.clone = original

    assert out.shape == (2, 3, 64)
    assert clones["n"] == 0


def test_awq_triton_rejects_declared_group_mismatching_buffers(monkeypatch):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, "triton", calls)
    qweight = torch.ones((32, 1), dtype=torch.int32)
    scales = torch.ones((2, 8), dtype=torch.float16)
    qzeros = torch.zeros((2, 1), dtype=torch.int32)
    x = torch.ones((1, 32), dtype=torch.float16)

    with pytest.raises(ValueError, match="declared group_size"):
        fn.apply(x, qweight, qzeros, scales, 4, 32, None, 8)
    assert calls == {"dequant": 0, "gemm": 0}


@pytest.mark.parametrize(
    "shape", [(0, 8), (3, 0, 8), (0, 3, 8), (2, 3, 0, 8), (2, 3, 4, 8)]
)
def test_linear_shape_helpers_preserve_rank_dtype_and_device(shape):
    x = torch.empty(shape, dtype=torch.bfloat16)
    assert input_rows(x) == x.numel() // x.shape[-1]
    out = empty_linear_output(x, 11)
    assert out.shape == shape[:-1] + (11,)
    assert out.dtype == x.dtype
    assert out.device == x.device


@pytest.mark.parametrize("shape", [(0, 8), (2, 0, 8), (2, 3, 0, 8)])
def test_empty_linear_output_keeps_autograd_connection(shape):
    x = torch.empty(shape, dtype=torch.float32, requires_grad=True)
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
        pytest.param(
            ParoTritonLinear,
            id="ParoTritonLinear",
            marks=pytest.mark.skipif(not HAS_TRITON, reason="ParoTritonLinear requires triton"),
        ),
    ],
)
def test_representative_qlinear_empty_shape_without_native_extension(linear_cls):
    # These forwards all check the common helper before touching backend state;
    # constructing a minimal shell keeps this test independent of extensions.
    module = object.__new__(linear_cls)
    module.out_features = 11
    module.adapter = None
    module.training = True
    module.kernel = KernelConfig()
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
