import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU required",
)


def _unpack_awq_codes(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int64)
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=packed.device)
    codes = (packed.to(torch.int64).unsqueeze(-1) >> shifts) & 15
    return codes.index_select(-1, order).flatten(-2)


def test_rocm_awq_gemm_auto_selects_triton_without_native_jit(monkeypatch):
    def unexpected_native_jit():
        raise AssertionError("native AWQ CUDA assembly must not compile on ROCm")

    monkeypatch.setattr(
        "gptqmodel.nn_modules.qlinear.gemm_awq.awq_runtime_available",
        unexpected_native_jit,
    )
    selected = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.ROCM,
        backend=BACKEND.AUTO,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert selected is AwqGEMMTritonLinear


@pytest.mark.parametrize("rows", [1, 16, 128, 256])
@pytest.mark.parametrize(
    "in_features,out_features,group_size", [(512, 512, 32), (4096, 4096, 128)]
)
def test_rocm_awq_triton_matches_independent_torch_oracle(
    rows, in_features, out_features, group_size
):
    torch.manual_seed(19 + rows + group_size)
    device = torch.device("cuda:0")
    groups = in_features // group_size
    qweight = torch.randint(
        -(2**31), 2**31 - 1,
        (in_features, out_features // 8), device=device, dtype=torch.int32,
    )
    qzeros = torch.randint(
        -(2**31), 2**31 - 1,
        (groups, out_features // 8), device=device, dtype=torch.int32,
    )
    qweight[0].zero_()
    qweight[1].fill_(-1)
    qzeros[0].zero_()
    qzeros[1].fill_(-1)
    scales = torch.rand(
        groups, out_features, device=device, dtype=torch.float16
    ).mul_(0.1).add_(0.01).contiguous()
    inputs = torch.randn(rows, in_features, device=device, dtype=torch.float16)

    module = AwqGEMMTritonLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        register_buffers=True,
    ).to(device).eval()
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.post_init()

    group_indices = torch.arange(in_features, device=device) // group_size
    codes = _unpack_awq_codes(qweight).float()
    zeros = _unpack_awq_codes(qzeros)[group_indices].float()
    dense_weight = ((codes - zeros) * scales[group_indices].float()).half()
    reference_fp32 = inputs.float() @ dense_weight.float()
    reference_visible = reference_fp32.half()

    with torch.inference_mode():
        actual = module(inputs)
    torch.cuda.synchronize(device)

    assert actual.dtype == inputs.dtype
    assert actual.shape == reference_visible.shape
    torch.testing.assert_close(actual, reference_visible, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual.float(), reference_fp32, rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------------------
# Portable Triton GEMV for the FORMAT.GEMV packing.
#
# The GEMV layout is *not* the GEMM layout: qweight is [N, K/8] with plain
# little-nibble order, qzeros is [N, ceil(groups/8)] padded up to a word, and
# scales is [N, padded_groups] FP16.  The oracle below unpacks that layout with
# its own arithmetic; it never calls the kernel or reuses its intermediates.
# ---------------------------------------------------------------------------


def _gemv_layout_dense(qweight, qzeros, scales, group_size):
    """FP64 dense weight from the GEMV packed buffers (independent oracle)."""

    nibbles = torch.arange(0, 8, device=qweight.device, dtype=torch.int64)
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * 8
    codes = ((qweight.to(torch.int64)[:, :, None] >> (nibbles * 4)) & 0xF).reshape(
        out_features, in_features
    )
    zeros = ((qzeros.to(torch.int64)[:, :, None] >> (nibbles * 4)) & 0xF).reshape(
        qzeros.shape[0], -1
    )
    group = torch.arange(in_features, device=qweight.device) // group_size
    return (codes.to(torch.float64) - zeros.to(torch.float64)[:, group]) * scales.to(
        torch.float64
    )[:, group]


def _build_gemv_modules(in_features, out_features, group_size, device, dtype):
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear, AwqGEMVTritonLinear

    groups = in_features // group_size
    scales = (
        torch.rand(out_features, groups, device=device, dtype=torch.float16) * 0.1 + 0.01
    ).contiguous()
    zeros = torch.randint(0, 15, (out_features, groups), device=device, dtype=torch.int32)
    # A dense reference layer only drives the inherited packer; it is released
    # before any forward so it cannot act as a weight cache.
    dense_layer = torch.nn.Linear(in_features, out_features, bias=False).to(device=device)

    modules = []
    for cls in (AwqGEMVLinear, AwqGEMVTritonLinear):
        module = cls(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        ).to(device).eval()
        module.pack(dense_layer, scales.clone(), zeros.clone())
        module.post_init()
        modules.append(module)
    del dense_layer
    return modules


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize(
    "in_features,out_features,group_size",
    [(512, 512, 128), (512, 512, 64), (2048, 2048, 128)],
)
def test_rocm_awq_gemv_triton_matches_independent_oracle(
    dtype, rows, in_features, out_features, group_size
):
    torch.manual_seed(101 + rows + group_size + int(dtype == torch.bfloat16))
    device = torch.device("cuda:0")
    native, triton = _build_gemv_modules(in_features, out_features, group_size, device, dtype)

    # Same inherited packer => byte-identical packed buffers.
    assert torch.equal(triton.qweight, native.qweight)
    assert torch.equal(triton.qzeros, native.qzeros)
    assert torch.equal(triton.scales, native.scales)

    inputs = torch.randn(rows, in_features, device=device, dtype=dtype)
    dense = _gemv_layout_dense(triton.qweight, triton.qzeros, triton.scales, group_size)
    truth = inputs.to(torch.float64) @ dense.t()
    rounded = truth.to(dtype).to(torch.float64)

    with torch.inference_mode():
        first = triton(inputs)
        # Repeated forwards must keep returning identical results and must not
        # grow any dense/dequantized attribute on the module.
        for _ in range(3):
            repeated = triton(inputs)
    torch.cuda.synchronize(device)

    assert first.dtype == inputs.dtype
    assert repeated.dtype == inputs.dtype
    assert torch.equal(first, repeated)
    # No dense/dequantized weight may survive the forwards.
    for name in ("dense_weight", "dequantized_weight", "_dense_cache", "_expanded_weight"):
        assert getattr(triton, name, None) is None, f"persistent dense cache: {name}"
    assert triton.qweight.dtype == torch.int32 and triton.qzeros.dtype == torch.int32

    # Arithmetic check: the kernel's FP32-accumulated result must match the FP64
    # truth after the one mandatory output cast.  FP16 stays inside 2e-3; BF16 is
    # limited by its ~3 decimal digits, so it is checked as <=1 ULP of the
    # already-rounded truth rather than a flat relative bound (docs/rocm-rdna4.md
    # records the same 1.25-ULP floor for the ParoQuant BF16 output).
    delta = (first.to(torch.float64) - rounded).abs()
    ulp = torch.finfo(dtype).eps * rounded.abs()
    assert torch.all(delta <= ulp * 1.25 + 1e-6), (
        f"max delta {delta.max().item():.3e} exceeds 1.25 ULP {float((ulp * 1.25).max()):.3e}"
    )
    if dtype == torch.float16:
        torch.testing.assert_close(first, rounded.to(dtype), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rows", [1, 8])
def test_rocm_awq_gemv_triton_is_never_auto_selected(rows):
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear, AwqGEMVTritonLinear
    from gptqmodel.utils.importer import (
        get_auto_backend_kernel_mapping,
        get_kernel_for_backend,
    )

    mapping = get_auto_backend_kernel_mapping()
    auto_entries = mapping[METHOD.AWQ][FORMAT.GEMV]
    assert list(auto_entries) == [BACKEND.AWQ_GEMV], (
        "the Triton GEMV must stay explicit-only so CUDA keeps the native kernel "
        f"and ROCm keeps its actionable refusal; got {list(auto_entries)}"
    )
    assert get_kernel_for_backend(BACKEND.AWQ_GEMV, METHOD.AWQ, FORMAT.GEMV) is AwqGEMVLinear
    assert (
        get_kernel_for_backend(BACKEND.AWQ_GEMV_TRITON, METHOD.AWQ, FORMAT.GEMV)
        is AwqGEMVTritonLinear
    )


def test_awq_gemv_triton_launch_table_has_rdna4_and_default():
    from gptqmodel.quantization.awq.modules.triton import gemv as gemv_module

    assert "rdna4" in gemv_module._AWQ_GEMV_LAUNCH_CONFIGS
    cfg = gemv_module.launch_config("gfx1201")
    assert cfg == gemv_module._AWQ_GEMV_LAUNCH_CONFIGS["rdna4"]
    # Unregistered families keep the portable fallback.
    assert gemv_module.launch_config("gfx942") == gemv_module._DEFAULT_AWQ_GEMV_LAUNCH
    # Ada (SM 8.9) has its own measured entry now (see test_awq_gemv_ada.py);
    # an unmeasured CUDA target still falls back.
    assert gemv_module.launch_config((8, 9)) == gemv_module._AWQ_GEMV_LAUNCH_CONFIGS["ada"]
    assert gemv_module.launch_config((8, 0)) == gemv_module._DEFAULT_AWQ_GEMV_LAUNCH


def _gemv_pack_to_gemm(qweight, qzeros, scales, group_size):
    """Re-pack GEMV buffers into the ``FORMAT.GEMM`` layout.

    ``FORMAT.GEMM`` stores ``qweight [K, N/8]`` with the AWQ
    ``[0,2,4,6,1,3,5,7]`` nibble interleave along the *output* axis,
    ``qzeros [groups, N/8]`` and ``scales [groups, N]``.  This is an independent
    transcode of the packed GEMV words (it never calls either kernel), used only
    so the two kernels consume identical arithmetic.
    """

    device = qweight.device
    nibbles = torch.arange(0, 8, device=device, dtype=torch.int64)
    out_features, words_k = qweight.shape
    in_features = words_k * 8
    shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int64)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device)

    # [N, K] GEMV plain codes -> [K, N] -> per-8 word, GEMM nibble order.
    codes = ((qweight.to(torch.int64)[:, :, None] >> (nibbles * 4)) & 0xF).reshape(
        out_features, in_features
    )
    codes_g = codes.t().reshape(in_features, out_features // 8, 8).index_select(-1, order)
    qweight_g = (codes_g << shifts).sum(-1).contiguous().to(torch.int32)  # [K, N/8]

    groups = in_features // group_size
    zcodes = ((qzeros.to(torch.int64)[:, :, None] >> (nibbles * 4)) & 0xF).reshape(
        out_features, -1
    )[:, :groups]
    zcodes_g = zcodes.t().reshape(groups, out_features // 8, 8).index_select(-1, order)
    qzeros_g = (zcodes_g << shifts).sum(-1).contiguous().to(torch.int32)  # [groups, N/8]

    scales_g = scales[:, :groups].t().contiguous()  # [groups, N]
    return qweight_g, qzeros_g, scales_g


def test_awq_gemv_and_gemm_modules_agree_at_decode():
    """At M==1 the GEMV and GEMM kernels must produce the same result.

    docs/rocm-rdna4.md states they are at *parity* at decode (0.81-1.38x, i.e.
    neither is a decode speedup over the other).  This pins the numerical side
    of that claim so a divergence in either kernel is caught; the timings
    themselves cannot be asserted on a shared GPU.
    """

    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTritonLinear

    torch.manual_seed(31337)
    device = torch.device("cuda:0")
    in_features, out_features, group_size = 1024, 512, 128
    groups = in_features // group_size

    dense_layer = torch.nn.Linear(in_features, out_features, bias=False).to(device)
    gemv = AwqGEMVTritonLinear(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features, bias=False,
        register_buffers=True,
    ).to(device).eval()
    gemv.pack(
        dense_layer,
        torch.rand(out_features, groups, device=device, dtype=torch.float16) * 0.1 + 0.01,
        torch.randint(0, 15, (out_features, groups), device=device, dtype=torch.int32),
    )
    gemv.post_init()
    del dense_layer

    qweight_g, qzeros_g, scales_g = _gemv_pack_to_gemm(
        gemv.qweight, gemv.qzeros, gemv.scales, group_size
    )
    gemm = AwqGEMMTritonLinear(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features, bias=False,
        register_buffers=True,
    ).to(device).eval()
    gemm.post_init()
    assert gemm.qweight.shape == qweight_g.shape
    assert gemm.qzeros.shape == qzeros_g.shape
    assert gemm.scales.shape == scales_g.shape
    gemm.qweight.copy_(qweight_g)
    gemm.qzeros.copy_(qzeros_g)
    gemm.scales.copy_(scales_g)

    inputs = torch.randn(1, in_features, device=device, dtype=torch.float16)
    with torch.inference_mode():
        out_gemv = gemv(inputs)
        out_gemm = gemm(inputs)
    torch.cuda.synchronize(device)

    assert out_gemv.dtype == torch.float16 and out_gemm.dtype == torch.float16
    torch.testing.assert_close(out_gemv, out_gemm, rtol=2e-3, atol=2e-3)
