# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for #5942: the native EXL3 GEMM folds the input guard.

Before this change every FP16/BF16 forward paid a separate pre-scale launch, an
FP32 GEMM output and a re-scale epilogue (FP64 for BF16 with ``suh``) to keep
#3731's wide-input guarantee. The native GEMM now reduces each row's bound
while it narrows and rotates the activation, applies the row's power-of-two
scale in its epilogue and rounds once into the final dtype. These tests pin
the dispatch, the bit-exact agreement with the eager guard, the fallbacks that
still own their shapes, and CUDA Graph replay.
"""

import pytest
import torch

from gptqmodel.nn_modules import exllamav3 as exl3_bridge
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.utils.kernel import KernelConfig
from tests.accelerator_gates import requires_nvidia_cuda


requires_cuda = pytest.mark.skipif(
    not requires_nvidia_cuda(),
    reason="the native EXL3 dispatch needs an NVIDIA CUDA device",
)

_IN_FEATURES = 512
_OUT_FEATURES = 256


def _native_layer(
    bias: bool = False, fused_gemm: bool = False, kernel=None
) -> ExllamaV3Linear:
    """A real W4 ``mul1`` layer with random trellis, suh and svh buffers."""
    generator = torch.Generator().manual_seed(5942)
    trellis = torch.randint(
        -32768,
        32768,
        (_IN_FEATURES // 16, _OUT_FEATURES // 16, 48),
        generator=generator,
        dtype=torch.int16,
    )
    suh = ((2 * torch.randint(0, 2, (_IN_FEATURES,), generator=generator) - 1) * 0.5).half()
    svh = (torch.randn(_OUT_FEATURES, generator=generator) * 0.03).half()
    tensors = {
        "trellis": trellis,
        "suh": suh,
        "svh": svh,
        "mul1": torch.ones((), dtype=torch.int32),
    }
    if bias:
        tensors["bias"] = torch.linspace(
            -0.1, 0.1, _OUT_FEATURES, dtype=torch.float32
        )
    tensors = {name: tensor.to("cuda") for name, tensor in tensors.items()}
    if kernel is None and fused_gemm:
        kernel = KernelConfig(exl3_fused_gemm=True)
    layer = ExllamaV3Linear.from_tensors(
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        name="guard_5942",
        tensors=tensors,
        kernel=kernel,
    )
    return layer.eval()


def _eager_reference(layer: ExllamaV3Linear, x: torch.Tensor) -> torch.Tensor:
    """The pre-#5942 eager guard: pre-scale, GEMM in FP32, re-scale, bias."""
    inner = layer._ensure_inner()
    native_input, output_scale = layer._native_bf16_input(x)
    assert native_input is not None
    accum = inner.forward(
        native_input, {"include_bias": False}, out_dtype=torch.float32
    )
    fused = exl3_bridge._try_fused_rescale(
        accum,
        output_scale,
        getattr(layer, "bias", None),
        layer.out_features,
        result_has_bias=False,
        output_dtype=x.dtype,
    )
    if fused is not None:
        return fused.to(x.dtype)
    return layer._rescale_native_output(
        accum[..., : layer.out_features], output_scale
    ).to(x.dtype)


class _FakeBC:
    """Records ``run_guarded`` and writes a recognisable result."""

    def __init__(self, code: int = 0, fp16_gemv=False, int8_gemv=False, fused_gemm=True):
        self.code = code
        self.fp16_gemv = fp16_gemv
        self.int8_gemv = int8_gemv
        self.fused_gemm = fused_gemm
        self.calls = []

    def run_guarded(self, x, y, *, guard_mode, bound):
        self.calls.append((tuple(x.shape), y.dtype, guard_mode, bound))
        if self.code >= 0:
            y.fill_(7.0)
        return self.code


class _FakeInner:
    def __init__(self, bc, *, out_features=_OUT_FEATURES, q4nx=False, r4nx=False, suh=None):
        self.bc = bc
        self.out_features = out_features
        self.q4nx = q4nx
        self.r4nx = r4nx
        self.suh = suh
        self.forward_calls = []

    def forward(self, x, params, *, out_dtype=None):
        self.forward_calls.append((x.dtype, out_dtype))
        return torch.zeros((*x.shape[:-1], self.out_features), dtype=out_dtype)


def _layer_with_inner(inner, *, out_features=_OUT_FEATURES, bias=None):
    layer = object.__new__(ExllamaV3Linear)
    layer.out_features = out_features
    layer.in_features = _IN_FEATURES
    layer.bias = bias
    layer._ensure_inner = lambda: inner
    layer._ensure_torch_inner = lambda: pytest.fail(
        "the guarded native path must not reconstruct through Torch"
    )
    return layer


@requires_cuda
@pytest.mark.parametrize(
    "dtype, has_suh, expected_mode",
    [
        (torch.float16, True, 1),
        (torch.float16, False, 1),
        (torch.bfloat16, True, 3),
        (torch.bfloat16, False, 2),
    ],
)
def test_forward_takes_the_guarded_native_path(dtype, has_suh, expected_mode):
    bc = _FakeBC()
    inner = _FakeInner(bc, suh=torch.ones(_IN_FEATURES) if has_suh else None)
    layer = _layer_with_inner(inner)

    x = torch.ones((2, _IN_FEATURES), dtype=dtype, device="cuda")
    with torch.inference_mode():
        output = layer.forward(x)

    assert bc.calls == [((2, _IN_FEATURES), dtype, expected_mode, exl3_bridge._FP16_SAFE_AMAX)]
    assert inner.forward_calls == []
    assert output.dtype == dtype and output.shape == (2, _OUT_FEATURES)
    assert torch.equal(output, torch.full_like(output, 7.0))


@requires_cuda
def test_guarded_path_uses_the_gemm_padded_width():
    """The GEMM may round N up; the module still exposes its own width."""
    bc = _FakeBC()
    inner = _FakeInner(bc, out_features=384)
    layer = _layer_with_inner(inner, out_features=256)

    x = torch.ones((1, _IN_FEATURES), dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        output = layer.forward(x)

    assert bc.calls[0][0] == (1, _IN_FEATURES)
    assert output.shape == (1, 256)


@requires_cuda
@pytest.mark.parametrize(
    "case",
    [
        "rows_over_32",
        "r4nx",
        "fp16_gemv",
        "int8_gemv",
        "q4nx_unfused",
        "no_bc",
        "op_declined",
    ],
)
def test_guarded_path_declines_shapes_it_does_not_own(case):
    bc = _FakeBC(
        code=-1 if case == "op_declined" else 0,
        fp16_gemv=case == "fp16_gemv",
        int8_gemv=case == "int8_gemv",
        fused_gemm=case != "q4nx_unfused",
    )
    inner = _FakeInner(
        bc,
        q4nx=case == "q4nx_unfused",
        r4nx=case == "r4nx",
        suh=torch.ones(_IN_FEATURES),
    )
    if case == "no_bc":
        inner.bc = None
    layer = _layer_with_inner(inner)

    rows = 33 if case == "rows_over_32" else 1
    x = torch.ones((rows, _IN_FEATURES), dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        result = layer._guarded_native_forward(x)

    assert result is None
    if case in ("no_bc", "r4nx", "rows_over_32", "q4nx_unfused", "fp16_gemv", "int8_gemv"):
        assert bc.calls == []
    else:
        assert len(bc.calls) == 1


@requires_cuda
def test_guarded_path_accepts_q4nx_large_prefill():
    """Q4NX keeps the tensor-core GEMM above 32 rows, so the guard follows."""
    bc = _FakeBC()
    inner = _FakeInner(bc, q4nx=True, suh=torch.ones(_IN_FEATURES))
    layer = _layer_with_inner(inner)

    x = torch.ones((33, _IN_FEATURES), dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        result = layer._guarded_native_forward(x)

    assert result is not None and result.shape == (33, _OUT_FEATURES)
    assert bc.calls[0][2] == 1


def test_guarded_path_declines_non_cuda_input():
    bc = _FakeBC()
    inner = _FakeInner(bc, suh=torch.ones(_IN_FEATURES))
    layer = _layer_with_inner(inner)

    assert layer._guarded_native_forward(torch.ones((1, _IN_FEATURES))) is None
    assert bc.calls == []


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 4, 32])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("outlier", [0.0, 6000.0])
def test_guarded_forward_is_bit_identical_to_the_eager_guard(dtype, rows, bias, outlier):
    layer = _native_layer(bias=bias)
    inner = layer._ensure_inner()
    bc = inner.bc
    calls = []
    original = bc.run_guarded

    def spy(x, y, **kwargs):
        calls.append(kwargs["guard_mode"])
        return original(x, y, **kwargs)

    bc.run_guarded = spy
    generator = torch.Generator(device="cuda").manual_seed(11)
    x = (torch.randn((rows, _IN_FEATURES), generator=generator, device="cuda") * 0.1).to(dtype)
    if outlier:
        x[0] = outlier

    with torch.inference_mode():
        guarded = layer(x)
        reference = _eager_reference(layer, x)

    assert calls == [1 if dtype == torch.float16 else 3]
    assert guarded.dtype == dtype and guarded.shape == (rows, _OUT_FEATURES)
    assert torch.equal(
        guarded.view(torch.int16), reference.view(torch.int16)
    ), (guarded.float() - reference.float()).abs().max()


@requires_cuda
def test_missing_guarded_op_falls_back_to_the_eager_guard(monkeypatch):
    """A stale prebuilt extension predates the folded op and must still work."""
    from gptqmodel.exllamav3 import ext as exl3_ext

    layer = _native_layer()
    original = exl3_ext._runtime_op

    def selective(name):
        if name == "bc_linear_exl3_run_guarded":
            raise AttributeError(name)
        return original(name)

    monkeypatch.setattr(exl3_ext, "_runtime_op", selective)
    x = torch.ones((2, _IN_FEATURES), dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        actual = layer(x)
        reference = _eager_reference(layer, x)

    assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))


@requires_cuda
def test_static_decline_is_cached_and_cleared(monkeypatch):
    """A policy decline is remembered, so the eager fallback probes once."""
    layer = _native_layer(kernel=KernelConfig(exl3_fp16_gemv=True))
    inner = layer._ensure_inner()
    assert inner.bc.fp16_gemv
    probes = []
    original = layer._ensure_inner

    def counting_probe():
        probes.append(1)
        return original()

    monkeypatch.setattr(layer, "_ensure_inner", counting_probe)
    x = torch.ones((1, _IN_FEATURES), dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        assert layer._guarded_native_forward(x) is None
        assert layer._guarded_native_forward(x) is None

    assert len(probes) == 1
    assert layer._guarded_native_unavailable is True
    layer._drop_inner()
    assert layer._guarded_native_unavailable is False


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_guarded_forward_handles_nonfinite_rows(dtype):
    """Non-finite rows keep the exact reconstruction instead of the GEMM."""
    layer = _native_layer()
    torch_inner = layer._ensure_torch_inner()
    bc = layer._ensure_inner().bc
    calls = []
    original = bc.run_guarded

    def spy(x, y, **kwargs):
        calls.append(kwargs["guard_mode"])
        return original(x, y, **kwargs)

    bc.run_guarded = spy
    x = torch.ones((2, _IN_FEATURES), dtype=dtype, device="cuda")
    x[0, 3] = float("inf")
    x[1, 5] = float("nan")

    with torch.inference_mode():
        actual = layer(x)
        reference = torch_inner(x)

    assert calls == []
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)


@requires_cuda
def test_guarded_forward_is_cuda_graph_safe():
    """A replay re-reads the static input and recomputes the row bound."""
    layer = _native_layer()
    inner = layer._ensure_inner()
    bc = inner.bc
    calls = []
    original = bc.run_guarded

    def spy(x, y, **kwargs):
        calls.append(kwargs["guard_mode"])
        return original(x, y, **kwargs)

    bc.run_guarded = spy
    x = torch.ones((2, _IN_FEATURES), dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        layer(x)  # warm up the inner handle and the finiteness probe

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = layer(x)
    assert calls == [1, 1]

    with torch.inference_mode():
        for value in (1.0, 6000.0, 0.25):
            x.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            reference = layer(x)
            assert torch.equal(
                static_output.view(torch.int16), reference.view(torch.int16)
            ), f"replay diverged at {value}"

    assert len(calls) == 5


@requires_cuda
def test_guarded_forward_handles_bf16_product_above_fp32():
    """BF16 + suh reduces the bound in FP64, so a >FP32 product stays finite."""
    layer = _native_layer()
    inner = layer._ensure_inner()
    # Only a few columns carry the huge checkpoint scale, so the row bound is
    # dominated by their product while the final projection stays in range.
    inner.suh[:8] = 65504.0
    x = torch.ones((1, _IN_FEATURES), dtype=torch.bfloat16, device="cuda")
    x[0, :8] = 1e35
    # |x * suh| ~ 6.5e39 overflows FP32 (3.4e38) but not the FP64 bound.
    product = x[0, :8].double() * inner.suh[:8].double()
    assert float(product.max()) > torch.finfo(torch.float32).max

    with torch.inference_mode():
        guarded = layer(x)
        reference = _eager_reference(layer, x)

    # The true projection is genuinely huge, so BF16 output saturation is
    # expected. The point is that an FP32 bound would become ``inf`` and turn
    # the whole row into the bias; the FP64 bound keeps the real values.
    assert not torch.equal(guarded, torch.zeros_like(guarded))
    torch.testing.assert_close(guarded, reference, rtol=0, atol=0, equal_nan=True)
