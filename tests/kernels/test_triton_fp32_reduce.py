# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The Triton dequant path must round once, after an FP32 reduction."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear._numerics import _wants_fp32_reduce, matmul_fp32_reduce
from tests.test_torch_weight_storage import build_case


# `quant_matmul` lives behind a module that imports Triton at import time, so it
# must not be imported here: a module-scope import would turn "Triton missing"
# into a collection error instead of the skip below.
def _quant_matmul():
    from gptqmodel.nn_modules.triton_utils.dequant import quant_matmul

    return quant_matmul


def _require_triton_gpu():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    pytest.importorskip("triton")


def _fp32_error_bound(x, weight, dtype):
    """Tight bound for "FP32 reduction, rounded once at the epilogue".

    Against an FP64 oracle a single output rounding can move a result by up to
    half an ulp of ``dtype``. The FP32 reduction adds at most a few ``eps32``
    per accumulated product. An activation-dtype reduction exceeds this bound,
    which is what the kernel used to do on CUDA.
    """
    half_ulp = 0.5 * torch.finfo(dtype).eps * (x.double() @ weight.double()).abs()
    reduction = 8 * torch.finfo(torch.float32).eps * (x.abs().double() @ weight.abs().double())
    return half_ulp + reduction


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64, 128])
def test_quant_matmul_matches_fp64_oracle(dtype, rows):
    """Triton output must satisfy the FP32-reduction bound against FP64."""
    _require_triton_gpu()

    module, _ = build_case(bits=4, dtype=dtype, group_size=128, in_features=1024, out_features=512)
    module = module.to("cuda")
    x = (torch.randn((rows, 1024), generator=torch.Generator().manual_seed(rows)) / 8).to("cuda", dtype)
    weight = module.dequantize_weight()

    with torch.inference_mode():
        actual = _quant_matmul()(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            module.bits, module.pack_dtype_bits, module.maxq,
        )

    assert actual.dtype == dtype
    assert torch.isfinite(actual).all(), "non-finite forward output"
    oracle = x.double() @ weight.double()
    bound = _fp32_error_bound(x, weight, dtype)

    # Visible output error stays inside the repo's inference limit. Both sides
    # are low-precision, so allow the mandatory output rounding as well as the
    # 2e-3/2e-3 limit: at magnitude 2 a bf16 half-ulp is 7.8e-3 absolute, which
    # the absolute limit alone would reject on a perfectly accurate kernel.
    visible_bound = torch.maximum(
        2e-3 + 2e-3 * oracle.abs(),
        torch.finfo(dtype).eps * oracle.abs(),
    )
    visible_excess = (actual.double() - oracle).abs() - visible_bound
    assert int((visible_excess > 0).sum()) == 0, (
        f"{dtype} M={rows}: {(visible_excess > 0).sum().item()}/{visible_excess.numel()} outputs "
        f"exceed the visible-error limit; max excess={visible_excess.max().item():.3e}"
    )
    # Arithmetic error: the kernel is allowed exactly one output rounding on
    # top of an FP32 reduction. This is a real constraint on the kernel --
    # an activation-dtype accumulation violates it well before the 2e-3 limit.
    excess = (actual.double() - oracle).abs() - bound
    assert int((excess > 0).sum()) == 0, (
        f"{dtype} M={rows}: {(excess > 0).sum().item()}/{excess.numel()} outputs exceed the "
        f"FP32-reduction bound; max excess={excess.max().item():.3e}"
    )

    # The plain low-precision matmul is measurably worse on this shape, which is
    # the drift the Triton path used to have.
    if _wants_fp32_reduce(x):
        narrow = (x @ weight).double()
        assert (actual.double() - oracle).abs().max() <= (narrow - oracle).abs().max()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 32, 128])
def test_quant_matmul_backward_matches_fp64_oracle(dtype, rows):
    """The ``transpose=True`` backward branch keeps the same FP32-reduction bound."""
    _require_triton_gpu()

    module, _ = build_case(bits=4, dtype=dtype, group_size=128, in_features=1024, out_features=512)
    module = module.to("cuda")
    weight = module.dequantize_weight()
    grad = (torch.randn((rows, 512), generator=torch.Generator().manual_seed(rows)) / 8).to("cuda", dtype)

    with torch.inference_mode():
        actual = _quant_matmul()(
            grad, module.qweight, module.scales, module.qzeros, module.g_idx,
            module.bits, module.pack_dtype_bits, module.maxq, transpose=True,
        )

    assert actual.dtype == dtype
    assert torch.isfinite(actual).all(), "non-finite backward output"
    oracle = grad.double() @ weight.double().t()
    bound = _fp32_error_bound(grad, weight.t(), dtype)
    excess = (actual.double() - oracle).abs() - bound
    assert int((excess > 0).sum()) == 0, (
        f"{dtype} M={rows}: {(excess > 0).sum().item()}/{excess.numel()} backward outputs exceed "
        f"the FP32-reduction bound; max excess={excess.max().item():.3e}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quant_matmul_width_reduction_matches_torch_backend(dtype):
    """Triton and Torch dequant backends must agree bit-for-bit on CUDA."""
    _require_triton_gpu()

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
    from gptqmodel.quantization import FORMAT

    source, _ = build_case(bits=4, dtype=dtype, group_size=128, in_features=1024, out_features=512)
    modules = []
    for cls in (TorchLinear, TritonV2Linear):
        module = cls(bits=4, group_size=128, sym=False, desc_act=True,
                     in_features=1024, out_features=512, bias=False, format=FORMAT.GPTQ_V2)
        module.load_state_dict(source.state_dict())
        module.post_init()
        module.eval()
        module.to("cuda", dtype)
        modules.append(module)

    # M=32 is the smallest batch where cuBLAS switches to a split reduction and
    # the old narrow path drifted from Torch.
    x = torch.randn((32, 1024), device="cuda", dtype=dtype) / 8
    with torch.inference_mode():
        torch_out = modules[0](x)
        triton_out = modules[1](x)

    torch.testing.assert_close(triton_out, torch_out, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_reduce_is_device_gated(dtype):
    """The wide request is CUDA-only; HIP already accumulates in FP32."""
    from gptqmodel.utils.rocm import IS_ROCM

    x = torch.zeros((1, 8), dtype=dtype)
    cpu = torch.zeros((8, 4), dtype=dtype)
    wide_on_cuda = not IS_ROCM
    assert _wants_fp32_reduce(x) is False  # CPU
    # The gate must not depend on the weight, only on device/dtype/runtime.
    assert matmul_fp32_reduce(x, cpu).dtype == dtype
    if torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16):
        assert _wants_fp32_reduce(torch.zeros((1, 8), dtype=dtype, device="cuda")) is wide_on_cuda
    # FP32 activations never take the wide path.
    assert _wants_fp32_reduce(torch.zeros((1, 8), dtype=torch.float32)) is False


@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_quant_matmul_oracle_rejects_nonfinite(monkeypatch, backward, value):
    _require_triton_gpu()

    def nonfinite_matmul(x, qweight, scales, qzeros, g_idx, *args, transpose=False):
        columns = g_idx.numel() if transpose else scales.shape[1]
        return torch.full((x.shape[0], columns), value, device=x.device, dtype=x.dtype)

    monkeypatch.setattr(f"{__name__}._quant_matmul", lambda: nonfinite_matmul)
    check = test_quant_matmul_backward_matches_fp64_oracle if backward else test_quant_matmul_matches_fp64_oracle
    with pytest.raises(AssertionError, match="non-finite"):
        try:
            check(torch.float16, 1)
        except pytest.skip.Exception as exc:
            pytest.fail(f"Non-finite output was hidden by a skip: {exc}")
