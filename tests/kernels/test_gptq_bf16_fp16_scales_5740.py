# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the pair-contract case; the dispatch/oracle cases carry
# ``pytest.mark.cuda`` and skip unless the device is an Ada (SM 8.9) GPU.
"""Fused GPTQ GEMV/small-M GEMM with FP16 checkpoint scales and BF16 activations.

Issue #5740: ``TorchLinear._can_use_rocm_fused`` required
``self.scales.dtype == x.dtype``.  GPTQ checkpoints always store FP16 scales
(``_scales_for_storage``) and the loader restores that dtype after a BF16 load,
so every BF16 model missed the fused kernels and ran the dense dequantize +
matmul fallback, which also casts the whole weight per forward.

The fused kernels load each scale and widen it to FP32 before the product, so
the stored scale operand does not have to equal the activation dtype.  These
tests pin the admitted pair, its numerics against the non-fused path and an
independent FP64 oracle, and the pairs that must stay on the dense path.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear, fused_scale_pair_supported
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.nn_modules.triton_utils.gptq_gemm import gptq_gemm
from gptqmodel.nn_modules.triton_utils.gptq_gemv import gptq_gemv
from gptqmodel.utils import arch


TOL = 2e-3
# FP32 accumulation-order bound between the fused kernel and the non-fused
# reduction: measured 4.3e-5 at most on the shapes below, so 1e-3 keeps a wide
# margin while still failing on a wrong decode or a narrowed activation.
DRIFT = 1e-3


@pytest.fixture(autouse=True)
def _arch_cache():
    arch.clear_arch_cache()
    yield
    arch.clear_arch_cache()


def _is_ada_host() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and arch.is_ada(0)
    )


requires_ada = pytest.mark.skipif(not _is_ada_host(), reason="Ada (SM 8.9) CUDA GPU required")


# --------------------------------------------------------------------------- #
# CPU-only: the admitted scale/activation dtype pairs.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("scale_dtype", "activation_dtype", "expected"),
    [
        # An equal pair keeps its previous behaviour.
        (torch.float16, torch.float16, True),
        (torch.bfloat16, torch.bfloat16, True),
        (torch.float32, torch.float32, True),
        (torch.float64, torch.float64, True),
        # The checkpoint pair this fix admits: FP16 scales stored by GPTQ,
        # BF16 activations of the model (#5740).
        (torch.float16, torch.bfloat16, True),
        (torch.bfloat16, torch.float16, True),
        # Unmeasured scale operands keep the dense path.
        (torch.float16, torch.float32, False),
        (torch.float32, torch.bfloat16, False),
        (torch.float32, torch.float16, False),
        (torch.float64, torch.bfloat16, False),
    ],
)
def test_fused_scale_pair_contract(scale_dtype, activation_dtype, expected):
    assert fused_scale_pair_supported(scale_dtype, activation_dtype) is expected


# --------------------------------------------------------------------------- #
# Kernel-oracle cases (real hardware only).
# --------------------------------------------------------------------------- #

def _make_module(in_features, out_features, dtype, group_size=128,
                 desc_act=False, bias=False, bits=4, cls=TorchLinear):
    module = cls(
        bits=bits, group_size=group_size, sym=True, desc_act=desc_act,
        in_features=in_features, out_features=out_features,
        pack_dtype=torch.int32, bias=bias, register_buffers=True,
    ).to(device="cuda", dtype=dtype).eval()
    module.optimize = lambda *args, **kwargs: None
    module.qzero_format(format=2)
    with torch.no_grad():
        module.qweight.copy_(
            torch.randint(-2**31, 2**31, module.qweight.shape, device="cuda", dtype=torch.int32)
        )
        module.qzeros.copy_(
            torch.randint(-2**31, 2**31, module.qzeros.shape, device="cuda", dtype=torch.int32)
        )
        module.scales.uniform_(0.001, 0.02)
        if desc_act:
            module.g_idx.copy_(module.g_idx[torch.randperm(in_features, device="cuda")])
            module.g_idx[0] = -1
        if bias:
            module.bias.uniform_(-0.05, 0.05)
    module.post_init()
    return module


def _fp64_oracle(x, module, *, narrow=True):
    """Independent decode + FP64 accumulation; never calls a kernel.

    ``narrow`` rounds the reconstructed weight to the activation dtype, which
    is what both the fused kernel and the dense fallback do.  Pass ``False``
    when the decoded weight does not fit the activation dtype.
    """
    pack_factor = 32 // module.bits
    maxq = (1 << module.bits) - 1
    group_index = module.g_idx.to(torch.int64)
    shifts = torch.arange(pack_factor, device=x.device, dtype=torch.int64) * module.bits
    codes = ((module.qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & maxq)
    codes = codes.reshape(module.in_features, module.out_features)
    zeros = ((module.qzeros.to(torch.int64).unsqueeze(-1) >> shifts[None, None, :]) & maxq)
    zeros = zeros.reshape(module.scales.shape[0], module.out_features)
    weights = (codes - zeros[group_index]).double() * module.scales[group_index].double()
    if narrow:
        weights = weights.to(x.dtype).double()
    out = x.double() @ weights
    if module.bias is not None:
        out = out + module.bias.double()
    return out


def _dense_wide(x, module, dtype):
    """The non-fused path's FP32 reduction, from the same decoded weight."""
    weight = module.dequantize_weight(dtype=dtype)
    if module.bias is None:
        return torch.mm(x, weight, out_dtype=torch.float32)
    return torch.addmm(module.bias.float(), x, weight, out_dtype=torch.float32)


def _fused_wide(x, module):
    """Call the fused kernel the gate selects, returning its FP32 result."""
    if x.shape[0] == 1:
        assert module._can_use_rocm_gemv(x) is True, "M=1 did not select the fused GEMV"
        return gptq_gemv(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=torch.float32, bits=module.bits,
        )
    assert module._can_use_rocm_gemm(x) is True, "small-M did not select the fused GEMM"
    return gptq_gemm(
        x, module.qweight, module.scales, module.qzeros, module.g_idx,
        out_dtype=torch.float32, bits=module.bits,
        uniform_group=module._gptq_gemm_uniform_group_for_current_g_idx(),
    )


def _assert_mixed_pair_case(dtype, shape, m, group_size, desc_act, bias, bits):
    in_features, out_features = shape
    torch.manual_seed(5740 + m + bits)
    module = _make_module(in_features, out_features, dtype, group_size, desc_act, bias, bits)
    # What a BF16 load of a GPTQ checkpoint leaves behind: FP16 scales.
    module.scales = module.scales.half()
    assert module.scales.dtype is torch.float16
    assert module.qweight.dtype is torch.int32, "packed weights must stay packed"

    x = torch.randn(m, in_features, device="cuda", dtype=dtype)
    with torch.inference_mode():
        fused_wide = _fused_wide(x, module)
        visible = module(x)
        # Force the dense fallback on the same tensors and dtype: this is the
        # non-fused reference for the identical stored weight.
        module._rocm_gemv_enabled = False
        module._rocm_gemm_enabled = False
        try:
            dense_visible = module(x)
        finally:
            module._rocm_gemv_enabled = True
            module._rocm_gemm_enabled = True
        dense_wide = _dense_wide(x, module, dtype)
        reference = _fp64_oracle(x, module)

    if module.bias is not None:
        fused_wide = fused_wide + module.bias.float()

    assert visible.dtype is dtype
    assert dense_visible.dtype is dtype
    # The fused kernel produced the visible output: one rounding of its FP32
    # epilogue, not the dense path's separate rounding.
    assert torch.equal(visible, fused_wide.to(dtype))
    # The fallback is exactly one rounding of its own FP32 reduction, so the
    # two paths can be compared before that rounding.
    assert torch.equal(dense_visible, dense_wide.to(dtype))

    drift = (fused_wide - dense_wide).abs().max().item()
    assert drift <= DRIFT, f"fused vs non-fused FP32 drift {drift:.3e} exceeds {DRIFT:g}"
    torch.testing.assert_close(fused_wide, dense_wide, rtol=TOL, atol=TOL)

    # Independent expected values: the FP64 oracle agrees with the fused FP32
    # result to the same bound.
    oracle_drift = (fused_wide.double() - reference).abs().max().item()
    assert oracle_drift <= DRIFT, f"fused vs FP64 oracle drift {oracle_drift:.3e} exceeds {DRIFT:g}"
    torch.testing.assert_close(fused_wide, reference.float(), rtol=TOL, atol=TOL)


_SHAPES = [
    # (in_features, out_features), group_size, desc_act, bias, bits, M
    ((2048, 1024), 128, False, False, 4, 1),
    ((896, 1280), 128, True, False, 4, 1),
    ((2048, 1024), 128, False, True, 4, 1),
    ((2048, 1024), 128, False, False, 8, 1),
    ((2048, 1024), 128, False, False, 4, 8),
    ((2048, 1024), 128, False, False, 2, 8),
    ((2048, 1024), 128, False, False, 4, 64),
]


@requires_ada
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("shape,group_size,desc_act,bias,bits,m", _SHAPES)
def test_bf16_activations_with_fp16_scales_match_non_fused_path(
    dtype, shape, group_size, desc_act, bias, bits, m,
):
    _assert_mixed_pair_case(dtype, shape, m, group_size, desc_act, bias, bits)


@requires_ada
@pytest.mark.cuda
def test_tritonv2_uses_the_fused_kernel_for_the_mixed_pair():
    module = _make_module(2048, 1024, torch.bfloat16, cls=TritonV2Linear)
    module.scales = module.scales.half()
    x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert module._can_use_rocm_gemv(x) is True
        visible = module(x)
        raw = gptq_gemv(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=torch.float32, bits=module.bits,
        )
        reference = _fp64_oracle(x, module)
    assert visible.dtype is torch.bfloat16
    assert torch.equal(visible, raw.to(torch.bfloat16))
    torch.testing.assert_close(raw, reference.float(), rtol=TOL, atol=TOL)
    assert (raw.double() - reference).abs().max().item() <= DRIFT


@requires_ada
@pytest.mark.cuda
def test_mixed_pair_gate_rejects_unmeasured_scale_dtype():
    # FP32 scales beside BF16 activations is not a measured fused pair: the
    # kernels only widen FP16/BF16 scale operands in their tests.
    module = _make_module(2048, 1024, torch.bfloat16)
    module.scales = module.scales.float()
    x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    assert fused_scale_pair_supported(module.scales.dtype, x.dtype) is False
    assert module._can_use_rocm_fused(x) is False
    assert module._can_use_rocm_gemv(x) is False
    assert module._can_use_rocm_gemm(torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)) is False

    with torch.inference_mode():
        out = module(x)
        reference = _fp64_oracle(x, module)
    assert out.dtype is torch.bfloat16
    assert torch.isfinite(out).all()
    # One output-dtype rounding of the FP64 reduction.
    assert (out.double() - reference).abs().max().item() <= 2 * torch.finfo(out.dtype).eps * reference.abs().max().item()


@requires_ada
@pytest.mark.cuda
def test_mixed_pair_gate_rejects_decode_that_overflows_the_activation_dtype():
    # The pair is admitted, but these FP16 scales decode weights beyond FP16:
    # the fused kernels stage the decoded weight in the activation dtype, so
    # the range guard (#5606) must keep this on the wide dense path.
    module = _make_module(2048, 1024, torch.float16)
    with torch.no_grad():
        module.scales.mul_(60000.0 / 0.02)
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16) * 1e-6
    assert fused_scale_pair_supported(module.scales.dtype, x.dtype) is True
    assert module._decoded_weight_fits(torch.float16) is False
    assert module._can_use_rocm_fused(x) is False
    assert module._can_use_rocm_gemv(x) is False

    with torch.inference_mode():
        out = module(x)
        reference = _fp64_oracle(x, module, narrow=False)
    assert out.dtype is torch.float16
    assert torch.isfinite(out).all()
    assert (out.double() - reference).abs().max().item() <= 2 * torch.finfo(out.dtype).eps * reference.abs().max().item()
