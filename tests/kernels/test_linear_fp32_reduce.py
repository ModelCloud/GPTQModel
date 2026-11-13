# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import weakref

import pytest
import torch

from gptqmodel.nn_modules.qlinear._numerics import linear_fp32_reduce, linear_wide_bias
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from tests.test_torch_weight_storage import state_tensors


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["matrix", "batched", "transposed", "sliced"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_rounds_after_bias(dtype, layout, with_bias):
    value = 300.0 if with_bias else 3.0
    x = torch.full((2, 3, 8 if layout == "sliced" else 4), value, device="cuda", dtype=dtype)
    if layout == "matrix":
        x = x.reshape(-1, 4)
    elif layout == "transposed":
        x = x.transpose(0, 1)
    elif layout == "sliced":
        x = x[..., ::2]
    weight = torch.full((8, 4), 100.0, device="cuda", dtype=dtype)
    bias = torch.full((8,), -60000.0, device="cuda", dtype=dtype) if with_bias else None
    oracle = x.double() @ weight.double().t()
    if bias is not None:
        oracle += bias.double()

    for _ in range(2):
        actual = linear_fp32_reduce(x, weight, bias)
        assert actual.dtype == dtype
        assert actual.shape == x.shape[:-1] + (8,)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, oracle.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vptq_noncontiguous_bias_and_weight_storage(dtype, monkeypatch):
    module = (
        VPTQLinear(
            bits=1,
            in_features=128,
            out_features=64,
            bias=True,
            vector_lens=[-1, 4],
            num_centroids=[-1, 16],
            num_res_centroids=[-1, -1],
            group_size=128,
            group_num=-1,
            enable_norm=False,
            enable_perm=False,
            dtype=dtype,
        )
        .to("cuda")
        .eval()
    )
    module.indices.zero_()
    module.bias.data.fill_(-60000.0)
    x = torch.full((2, 3, 128), 30.0, device="cuda", dtype=dtype).transpose(0, 1)
    references = []
    original = module.dequantize

    def tracked(*args, **kwargs):
        weight = original(*args, **kwargs)
        references.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(module, "dequantize", tracked)
    allowed = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(module)}
    for centroid in (32.0, 16.0):
        module.centroids.weight.data.fill_(centroid)
        oracle = x.double().sum(-1, keepdim=True) * centroid + module.bias.double()
        for _ in range(2):
            with torch.inference_mode():
                actual = module(x)
            assert actual.dtype == dtype
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, oracle.to(dtype), rtol=0, atol=0)
            assert references and all(reference() is None for reference in references)
            assert all(tensor.untyped_storage().data_ptr() in allowed for tensor in state_tensors(module))


# The user-visible FP16 regression from the issue: a unit dot of sixteen 4096
# activations is exactly 65536 (legal in FP32/BF16, `inf` in FP16) and a stored
# FP32 -65536 bias must cancel it to zero.
_CANCELLATION_ROWS = 16
_CANCELLATION_VALUE = 4096.0
_CANCELLATION_BIAS = -(_CANCELLATION_ROWS * _CANCELLATION_VALUE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_fp32_reduce_never_narrows_wide_bias(dtype):
    """A bias wider than the activation must be added after a wide reduction."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    x = torch.full((1, _CANCELLATION_ROWS), _CANCELLATION_VALUE, device=device, dtype=dtype)
    weight = torch.ones((1, _CANCELLATION_ROWS), device=device, dtype=dtype)
    bias = torch.full((1,), _CANCELLATION_BIAS, device=device, dtype=torch.float32)
    oracle = (x.double() @ weight.double().t() + bias.double()).to(dtype)

    for _ in range(2):
        actual = linear_fp32_reduce(x, weight, bias)
        assert actual.dtype is dtype
        assert actual.shape == (1, 1)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["matrix", "batched", "transposed", "sliced"])
def test_linear_fp32_reduce_wide_bias_layouts(dtype, layout):
    """Wide-bias cancellation must hold for every input layout the module accepts."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    width = 8 if layout == "sliced" else _CANCELLATION_ROWS
    x = torch.full((2, 3, width * 2 if layout == "sliced" else width), _CANCELLATION_VALUE, device=device, dtype=dtype)
    if layout == "matrix":
        x = x.reshape(-1, width)
    elif layout == "transposed":
        x = x.transpose(0, 1)
    elif layout == "sliced":
        x = x[..., ::2]
    assert x.shape[-1] == width
    weight = torch.ones((1, width), device=device, dtype=dtype)
    bias = torch.full((1,), -(width * _CANCELLATION_VALUE), device=device, dtype=torch.float32)
    oracle = (x.double() @ weight.double().t() + bias.double()).to(dtype)

    actual = linear_fp32_reduce(x, weight, bias)

    assert actual.dtype is dtype
    assert actual.shape == x.shape[:-1] + (1,)
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_wide_bias_preserves_broadcast_bias_shape(dtype):
    """A wide bias that only broadcasts after the output reshape must still apply.

    The wide epilogue flattens the activation for its 2-D GEMM. A per-batch bias
    such as (2, 1, 48) does not broadcast against the flattened (6, 48) product,
    so the bias has to be added after the output is reshaped back.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    generator = torch.Generator(device=device).manual_seed(11)
    x = (torch.rand((2, 3, 32), generator=generator, device=device) - 0.5).to(dtype)
    weight = ((torch.rand((48, 32), generator=generator, device=device) - 0.5) / 8).to(dtype)
    # FP32 keeps the stored width; the narrow bias fails to broadcast against the
    # flattened (6, 48) product and used to raise a shape error.
    bias = (torch.rand((2, 1, 48), generator=generator, device=device) - 0.5)
    oracle = (x.double() @ weight.double().t() + bias.double()).to(dtype)

    for actual in (linear_wide_bias(x, weight, bias), linear_fp32_reduce(x, weight, bias)):
        assert actual.shape == (2, 3, 48)
        assert actual.dtype is dtype
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vptq_wide_bias_cancels_without_dense_cache(dtype):
    """VPTQ must add its wide bias after a wide reduction and retain no dense weight."""

    module = (
        VPTQLinear(
            bits=1,
            in_features=_CANCELLATION_ROWS,
            out_features=1,
            bias=True,
            vector_lens=[-1, 4],
            num_centroids=[-1, 16],
            num_res_centroids=[-1, -1],
            group_size=_CANCELLATION_ROWS,
            group_num=-1,
            enable_norm=False,
            enable_perm=False,
            dtype=dtype,
        )
        .eval()
    )
    module.indices.zero_()
    module.centroids.weight.data.fill_(1.0)
    # The checkpoint stores the bias in FP32; a FP16 buffer cannot hold it.
    module.register_buffer("bias", torch.tensor([_CANCELLATION_BIAS], dtype=torch.float32))
    assert module.bias.dtype is torch.float32
    x = torch.full((1, _CANCELLATION_ROWS), _CANCELLATION_VALUE, dtype=dtype)

    references = []
    original = module.dequantize
    allowed = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(module)}

    def tracked(*args, **kwargs):
        weight = original(*args, **kwargs)
        references.append(weakref.ref(weight))
        return weight

    module.dequantize = tracked
    for _ in range(2):
        with torch.inference_mode():
            actual = module(x)
        assert actual.dtype is dtype
        assert torch.isfinite(actual).all()
        assert float(actual.item()) == 0.0

    # Only the packed checkpoint buffers may persist after forward.
    assert references and all(reference() is None for reference in references)
    assert {tensor.untyped_storage().data_ptr() for tensor in state_tensors(module)} <= allowed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vptq_cuda_fused_bias_gate_follows_live_values(dtype):
    """The fused-kernel gate must read the live bias, including in-place writes.

    The exact representability check reads device memory, so an answer cached
    under a version counter would survive ``.data.copy_``/``.data.fill_``, which
    do not bump it. The gate is therefore recomputed per call.
    """

    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("VPTQ_CUDA is CUDA-only")

    from gptqmodel.nn_modules.qlinear._numerics import bias_narrows_losslessly
    from gptqmodel.nn_modules.qlinear.vptq_cuda import VPTQCudaLinear

    module = (
        VPTQCudaLinear(
            bits=1,
            in_features=_CANCELLATION_ROWS,
            out_features=1,
            bias=True,
            vector_lens=[-1, 4],
            num_centroids=[-1, 16],
            num_res_centroids=[-1, -1],
            group_size=_CANCELLATION_ROWS,
            group_num=-1,
            enable_norm=False,
            enable_perm=False,
            dtype=dtype,
        )
        .eval()
        .to("cuda")
    )
    module.indices.zero_()
    module.centroids.weight.data.fill_(1.0)
    module.bias.copy_(torch.tensor([0.5], device="cuda", dtype=torch.float32))
    assert bias_narrows_losslessly(module.bias, dtype) is True

    # ``-65536 - 1`` is exact in neither FP16 nor BF16. An in-place write must
    # be observed on the very next call.
    inexact = float(_CANCELLATION_BIAS) - 1.0
    module.bias.copy_(torch.tensor([inexact], device="cuda", dtype=torch.float32))
    assert bias_narrows_losslessly(module.bias, dtype) is False

    # ``load_quant_result`` repacks through ``.data.copy_`` (no version bump);
    # the value still has to be read fresh.
    module.bias.data.copy_(torch.tensor([0.5], device="cuda", dtype=torch.float32))
    assert bias_narrows_losslessly(module.bias, dtype) is True


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vptq_cuda_unrepresentable_wide_bias_skips_fused_kernel(dtype, monkeypatch):
    """A bias the activation dtype cannot hold must not reach the native kernel.

    The fused kernel adds the bias after its own wide accumulation but requires
    the bias storage dtype to match the output, so an inexact value such as
    ``-(rows * value) - 1`` has to route through the FP32 epilogue instead.
    """

    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("VPTQ_CUDA is CUDA-only")

    from gptqmodel.nn_modules.qlinear._numerics import bias_narrows_losslessly
    from gptqmodel.nn_modules.qlinear.vptq_cuda import VPTQCudaLinear

    module = (
        VPTQCudaLinear(
            bits=1,
            in_features=_CANCELLATION_ROWS,
            out_features=1,
            bias=True,
            vector_lens=[-1, 4],
            num_centroids=[-1, 16],
            num_res_centroids=[-1, -1],
            group_size=_CANCELLATION_ROWS,
            group_num=-1,
            enable_norm=False,
            enable_perm=False,
            dtype=dtype,
        )
        .to("cuda")
        .eval()
    )
    module.indices.zero_()
    module.centroids.weight.data.fill_(1.0)
    # One unit below the cancelling value: not representable in FP16 or BF16, so
    # the cancelling dot must still be evaluated wide.
    bias_value = _CANCELLATION_BIAS - 1.0
    module.register_buffer(
        "bias", torch.tensor([bias_value], dtype=torch.float32, device="cuda")
    )
    assert not bias_narrows_losslessly(module.bias, dtype)
    # One row is inside the fused-kernel row limit; only the wide bias skips it.
    x = torch.full((1, _CANCELLATION_ROWS), _CANCELLATION_VALUE, device="cuda", dtype=dtype)

    fused_calls = []

    def fail_fused(*args, **kwargs):
        fused_calls.append(True)
        raise AssertionError("an inexact wide bias must not reach the native VPTQ kernel")

    monkeypatch.setattr("gptqmodel.utils.vptq_cuda.vptq_cuda_linear", fail_fused)

    with torch.inference_mode():
        actual = module(x)

    assert not fused_calls
    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    oracle_weight = module.dequantize().to(dtype=torch.float64)
    oracle = (x.double() @ oracle_weight.t() + module.bias.double()).to(dtype)
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
except Exception:
    NVFP4Tensor = None


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp4_accelerator_forward_is_graph_capturable(dtype):
    """The FP4 accelerator wrapper shares the value-dependent gate.

    The stored bias must be a value the activation dtype cannot represent.
    ``-65536`` is exact in BF16, so a BF16 fixture with that value never reaches
    `bias_narrows_losslessly`: it takes the narrow ephemeral cast and the stock
    torchao ``F.linear``. ``0.1`` is inexact in both dtypes, so both
    parameterizations select `linear_wide_bias` and the capture/compile
    short-circuit is genuinely pinned for each.

    The replay also reads the *live* bias: ``.data.fill_`` does not bump a
    version counter, so a graph that baked the captured value in, or that
    cached a representability answer, would drift from the oracle after the
    write. This is the FP4 counterpart of
    ``test_vptq_bias_epilogue_captures_and_replays_live_values``.
    """

    import gptqmodel.nn_modules.qlinear.fp4 as fp4_module

    weight = torch.randn(16, 64, dtype=torch.bfloat16) * 0.04
    packed = NVFP4Tensor.to_nvfp4(weight, block_size=16)
    module = (
        fp4_module.TorchFP4Linear(
            in_features=64,
            out_features=16,
            weight=packed.qdata,
            weight_scale=packed.scale,
            weight_block_size=16,
            orig_dtype=torch.bfloat16,
            bias=torch.full((16,), 0.1, device="cuda", dtype=torch.float32),
        )
        .to("cuda")
        .eval()
    )
    x = torch.full((1, 64), 4096.0, device="cuda", dtype=dtype)

    # Exact in neither FP16 nor BF16, so both dtypes take the wide epilogue.
    assert module._epilogue_bias(x, module.bias)[1] is True

    with torch.inference_mode():
        expected = module(x)
    # ``0.1`` keeps every FP16/BF16 output finite, so the comparison validates
    # arithmetic instead of comparing ``-inf`` with ``-inf``.
    assert torch.isfinite(expected).all()
    density = module._native_weight().dequantize(x.dtype)
    oracle = (x.double() @ density.double().t() + module.bias.double()).to(dtype)
    torch.testing.assert_close(expected, oracle, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        with torch.inference_mode():
            module(x)
    torch.cuda.current_stream().wait_stream(side_stream)

    with torch.inference_mode():
        with torch.cuda.graph(graph):
            captured = module(x)
    graph.replay()
    assert captured.dtype is dtype
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)

    # `.data.fill_` is invisible to the version counter, so a captured graph
    # that baked the bias value in or cached the gate answer would keep the old
    # result. Re-read the live buffer on every replay instead. The values stay
    # inside the FP16/BF16 range so the comparison measures the epilogue rather
    # than saturating at the same `-inf` on both sides; `0.5` and `3.5` are
    # exact in both dtypes and `0.1` is exact in neither.
    for live_bias in (0.5, 0.1, 3.5):
        module.bias.data.fill_(live_bias)
        graph.replay()
        torch.cuda.synchronize()
        live_oracle = (x.double() @ density.double().t() + module.bias.double()).to(dtype)
        torch.testing.assert_close(captured, live_oracle, rtol=0, atol=0)

    module.bias.data.fill_(0.1)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    with torch.inference_mode():
        traced = compiled(x)
    torch.testing.assert_close(traced, expected, rtol=0, atol=0)
