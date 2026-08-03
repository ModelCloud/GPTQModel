# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""GPU accuracy tests for the planar (gptq_p) Triton kernels at 3/5/6/7-bit.

Verifies against the Torch planar reference (`planar_unpack_rows/cols` +
`_dequantize_from_codes`):
1. `planar_dequant` is bit-exact vs the eager planar dequant path
2. `planar_matmul` (fused dequant+matmul) matches the dequant reference
   within fp16/bf16 matmul tolerance across decode/prefill batch shapes
3. desc_act-style shuffled g_idx and sym=True variants stay accurate
4. `TorchLinear.dequantize_weight` routes planar modules to the Triton
   dequant kernel on CUDA
5. `TritonV2Linear.forward` matches the Torch planar reference forward
"""

import pytest
import torch
import torch.nn as nn


torch_cuda_available = torch.cuda.is_available()

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch_cuda_available, reason="planar Triton kernels require CUDA"),
]

PLANAR_KERNEL_BITS = (3, 5, 6, 7)


def _make_inputs(bits: int, in_features: int, out_features: int, group_size: int,
                 desc_act: bool = False, seed: int = 0):
    torch.manual_seed(seed + bits)
    maxq = (1 << bits) - 1
    groups = in_features // group_size
    linear = nn.Linear(in_features, out_features, bias=True)
    scales = torch.rand(out_features, groups) * 0.01 + 0.005
    zeros = torch.randint(0, maxq + 1, (out_features, groups)).float()
    if desc_act:
        perm = torch.randperm(in_features)
        g_idx = (perm // group_size).to(torch.int32)
    else:
        g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    return linear, scales, zeros, g_idx


def _packed_module(bits: int, in_features: int = 256, out_features: int = 128,
                   group_size: int = 32, desc_act: bool = False, sym: bool = False,
                   seed: int = 0, cls=None):
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import FORMAT

    module_cls = cls or TorchLinear
    linear, scales, zeros, g_idx = _make_inputs(
        bits, in_features, out_features, group_size, desc_act=desc_act, seed=seed
    )
    module = module_cls(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        format=FORMAT.GPTQ_P,
        register_buffers=False,
    )
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())
    return module


def _reference_dequant(module) -> torch.Tensor:
    """CPU planar reference: unpack logical codes then dequantize."""
    from gptqmodel.utils.planar_packing import planar_unpack_cols, planar_unpack_rows

    qweight = module.qweight.cpu()
    qzeros = module.qzeros.cpu()
    scales = module.scales.cpu()
    g_idx = module.g_idx.cpu().long()

    zeros = planar_unpack_cols(qzeros, module.bits).reshape(scales.shape)
    weight = planar_unpack_rows(qweight, module.bits)
    return scales[g_idx] * (weight - zeros[g_idx])


def _to_cuda(module):
    module.qweight = module.qweight.cuda()
    module.qzeros = module.qzeros.cuda()
    module.scales = module.scales.cuda()
    module.g_idx = module.g_idx.cuda()
    if module.bias is not None:
        module.bias = module.bias.cuda()
    return module


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("desc_act", [False, True])
def test_planar_dequant_bit_exact(bits: int, desc_act: bool):
    from gptqmodel.nn_modules.triton_utils.planar import planar_dequant

    module = _packed_module(bits, desc_act=desc_act)
    ref = _reference_dequant(module)
    _to_cuda(module)

    out = planar_dequant(
        torch.float16, module.qweight, module.scales, module.qzeros, module.g_idx, module.bits
    )
    assert out.shape == ref.shape
    assert out.dtype == torch.float16
    assert torch.equal(out.cpu(), ref.to(torch.float16))


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
def test_planar_dequant_uneven_group_size(bits: int):
    from gptqmodel.nn_modules.triton_utils.planar import planar_dequant

    module = _packed_module(bits, in_features=384, out_features=64, group_size=128)
    ref = _reference_dequant(module)
    _to_cuda(module)

    out = planar_dequant(
        torch.float16, module.qweight, module.scales, module.qzeros, module.g_idx, module.bits
    )
    assert torch.equal(out.cpu(), ref.to(torch.float16))


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
def test_planar_dequant_rejects_unaligned_rows(bits: int):
    # The block kernel covers exactly in_features // 32 blocks; unaligned rows
    # would silently stay uninitialized, so the contract is explicit.
    from gptqmodel.nn_modules.triton_utils.planar import planar_dequant

    module = _packed_module(bits)
    _to_cuda(module)
    g_idx_unaligned = module.g_idx[:-8]
    with pytest.raises(ValueError, match="divisible by 32"):
        planar_dequant(
            torch.float16, module.qweight, module.scales, module.qzeros, g_idx_unaligned, module.bits
        )


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 8, 33, 128])
def test_planar_matmul_matches_reference(bits: int, dtype: torch.dtype, batch: int):
    from gptqmodel.nn_modules.triton_utils.planar import planar_matmul

    module = _packed_module(bits)
    ref_w = _reference_dequant(module).to(dtype)
    _to_cuda(module)

    torch.manual_seed(bits + batch)
    x = (torch.randn(batch, module.in_features, dtype=dtype) * 0.5).cuda()
    ref = (x.cpu().float() @ ref_w.float()).to(dtype)

    out = planar_matmul(x, module.qweight, module.scales, module.qzeros, module.g_idx, module.bits)
    assert out.shape == ref.shape
    tol = 2e-2 if dtype == torch.float16 else 1e-1
    assert torch.allclose(out.cpu().float(), ref.float(), atol=tol, rtol=1e-2)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("desc_act,sym", [(True, False), (False, True)])
def test_planar_matmul_desc_act_sym(bits: int, desc_act: bool, sym: bool):
    from gptqmodel.nn_modules.triton_utils.planar import planar_matmul

    module = _packed_module(bits, desc_act=desc_act, sym=sym)
    ref_w = _reference_dequant(module).to(torch.float16)
    _to_cuda(module)

    torch.manual_seed(bits)
    x = (torch.randn(4, module.in_features, dtype=torch.float16) * 0.5).cuda()
    ref = (x.cpu().float() @ ref_w.float()).to(torch.float16)

    out = planar_matmul(x, module.qweight, module.scales, module.qzeros, module.g_idx, module.bits)
    assert torch.allclose(out.cpu().float(), ref.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_planar_gemv_matches_reference(bits: int, dtype: torch.dtype, batch: int):
    from gptqmodel.nn_modules.triton_utils.planar import planar_gemv

    module = _packed_module(bits)
    ref_w = _reference_dequant(module).to(dtype)
    _to_cuda(module)

    torch.manual_seed(bits + batch)
    x = (torch.randn(batch, module.in_features, dtype=dtype) * 0.5).cuda()
    ref = (x.cpu().float() @ ref_w.float()).to(dtype)

    out = planar_gemv(x, module.qweight, module.scales, module.qzeros, module.g_idx, module.bits)
    assert out.shape == ref.shape
    assert out.dtype == dtype
    tol = 2e-2 if dtype == torch.float16 else 1e-1
    assert torch.allclose(out.cpu().float(), ref.float(), atol=tol, rtol=1e-2)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("desc_act,sym", [(True, False), (False, True)])
def test_planar_gemv_desc_act_sym(bits: int, desc_act: bool, sym: bool):
    """desc_act shuffles g_idx, exercising the non-GROUP_UNIFORM kernel branch."""
    from gptqmodel.nn_modules.triton_utils.planar import planar_gemv

    module = _packed_module(bits, desc_act=desc_act, sym=sym)
    ref_w = _reference_dequant(module).to(torch.float16)
    _to_cuda(module)

    torch.manual_seed(bits)
    x = (torch.randn(2, module.in_features, dtype=torch.float16) * 0.5).cuda()
    ref = (x.cpu().float() @ ref_w.float()).to(torch.float16)

    out = planar_gemv(x, module.qweight, module.scales, module.qzeros, module.g_idx, module.bits)
    assert torch.allclose(out.cpu().float(), ref.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
def test_planar_matmul_n_not_multiple_of_block(bits: int):
    """out_features that is a multiple of 32 but not of BLOCK_N (64) must stay in bounds."""
    from gptqmodel.nn_modules.triton_utils.planar import planar_matmul

    module = _packed_module(bits, out_features=160)
    ref_w = _reference_dequant(module).to(torch.float16)
    _to_cuda(module)

    torch.manual_seed(bits)
    x = (torch.randn(4, module.in_features, dtype=torch.float16) * 0.5).cuda()
    ref = (x.cpu().float() @ ref_w.float()).to(torch.float16)

    out = planar_matmul(x, module.qweight, module.scales, module.qzeros, module.g_idx, module.bits)
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out.cpu().float(), ref.float(), atol=2e-2, rtol=1e-2)


def _pangolin_available() -> bool:
    from gptqmodel.utils.pangolin import ensure_pangolin_runtime_available

    return ensure_pangolin_runtime_available()


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_pangolin_gemv_matches_reference(bits: int, dtype: torch.dtype, batch: int):
    if not _pangolin_available():
        pytest.skip("pangolin native CUDA extension unavailable")
    from gptqmodel.utils.pangolin import pangolin_gemv

    module = _packed_module(bits)
    ref_w = _reference_dequant(module).to(dtype)
    _to_cuda(module)

    torch.manual_seed(bits + batch)
    x = (torch.randn(batch, module.in_features, dtype=dtype) * 0.5).cuda()
    ref = (x.cpu().float() @ ref_w.float()).to(dtype)

    out = pangolin_gemv(
        x, module.qweight, module.scales.to(dtype), module.qzeros, module.g_idx, module.bits
    )
    assert out.shape == ref.shape
    assert out.dtype == dtype
    tol = 2e-2 if dtype == torch.float16 else 1e-1
    assert torch.allclose(out.cpu().float(), ref.float(), atol=tol, rtol=1e-2)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
def test_pangolin_gemv_sym_metadata(bits: int):
    if not _pangolin_available():
        pytest.skip("pangolin native CUDA extension unavailable")
    from gptqmodel.utils.pangolin import pangolin_gemv

    module = _packed_module(bits, sym=True)
    ref_w = _reference_dequant(module).to(torch.float16)
    _to_cuda(module)

    torch.manual_seed(bits)
    x = (torch.randn(2, module.in_features, dtype=torch.float16) * 0.5).cuda()
    ref = (x.cpu().float() @ ref_w.float()).to(torch.float16)

    out = pangolin_gemv(x, module.qweight, module.scales, module.qzeros, module.g_idx, module.bits)
    assert torch.allclose(out.cpu().float(), ref.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
def test_torch_linear_routes_planar_to_triton(bits: int):
    module = _packed_module(bits)
    ref = _reference_dequant(module)
    _to_cuda(module)
    module.eval()

    assert module.planar
    assert module._can_use_triton_dequant()
    out = module._dequantize_weight_triton()
    assert out.device.type == "cuda"
    assert torch.equal(out.cpu(), ref.to(torch.float16))


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("batch", [1, 64])
def test_tritonv2_forward_matches_torch_reference(bits: int, batch: int):
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear

    # TritonV2 3-bit construction keeps the continuous fused-path contract
    # (sym=True, desc_act=False); other planar 3-bit configs select TorchLinear.
    module = _packed_module(bits, sym=(bits == 3), cls=TritonV2Linear)
    ref_w = _reference_dequant(module).to(torch.float16)
    bias = module.bias.detach().clone()
    _to_cuda(module)
    module.eval()

    torch.manual_seed(bits + batch)
    x = (torch.randn(batch, module.in_features, dtype=torch.float16) * 0.5).cuda()
    ref = ((x.cpu().float() @ ref_w.float()) + bias.float()).to(torch.float16)

    with torch.inference_mode():
        out = module(x)
    assert out.shape == ref.shape
    assert torch.allclose(out.cpu().float(), ref.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
def test_tritonv2_forward_zero_rows(bits: int):
    """Zero-row inputs (e.g. unused MoE experts) must return empty, not raise."""
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear

    module = _packed_module(bits, sym=(bits == 3), cls=TritonV2Linear)
    _to_cuda(module)
    module.eval()

    x = torch.empty(0, module.in_features, dtype=torch.float16).cuda()
    with torch.inference_mode():
        out = module(x)
    assert out.shape == (0, module.out_features)
    assert out.dtype == torch.float16
    assert out.device.type == "cuda"


# Laguna S 2.1 attention head shapes (3072x48, 3072x72) have non-/32 N;
# 112x64 additionally exercises K (in_features) padding.
_AUTO_PAD_SHAPES = ((3072, 48, 32), (3072, 72, 32), (112, 64, 16))


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("shape", _AUTO_PAD_SHAPES)
@pytest.mark.parametrize("batch", [1, 8, 33])
def test_planar_auto_pad_forward(bits: int, shape, batch: int):
    """Non-/32 logical dims are padded once at pack time; forward pads K and slices N."""
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear

    in_features, out_features, group_size = shape
    module = _packed_module(
        bits, in_features=in_features, out_features=out_features,
        group_size=group_size, sym=(bits == 3), cls=TritonV2Linear,
    )
    padded_in = (in_features + 31) // 32 * 32
    padded_out = (out_features + 31) // 32 * 32
    assert module.padded_in_features == padded_in
    assert module.padded_out_features == padded_out
    assert module.qweight.shape == (padded_in // 32 * bits, padded_out)
    assert module.g_idx.shape[0] == padded_in
    assert module.bias.shape[0] == out_features

    ref_w = _reference_dequant(module).to(torch.float16)
    assert ref_w.shape == (padded_in, padded_out)
    # padded rows/columns must contribute exactly zero
    assert ref_w[in_features:, :].abs().max().item() == 0.0 if padded_in != in_features else True
    assert ref_w[:, out_features:].abs().max().item() == 0.0 if padded_out != out_features else True

    bias = module.bias.detach().clone()
    _to_cuda(module)
    module.eval()

    torch.manual_seed(bits + batch)
    x = (torch.randn(batch, in_features, dtype=torch.float16) * 0.5).cuda()
    ref = ((x.cpu().float() @ ref_w[:in_features, :out_features].float()) + bias.float()).to(torch.float16)

    with torch.inference_mode():
        out = module(x)
    assert out.shape == (batch, out_features)
    assert out.dtype == torch.float16
    assert torch.allclose(out.cpu().float(), ref.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
def test_pangolin_auto_pad_routing(bits: int):
    """Pangolin must accept the padded (aligned) buffers for non-/32 logical shapes."""
    if not _pangolin_available():
        pytest.skip("pangolin native CUDA extension unavailable")
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear

    module = _packed_module(bits, in_features=3072, out_features=48, sym=(bits == 3), cls=TritonV2Linear)
    ref_w = _reference_dequant(module).to(torch.float16)
    _to_cuda(module)
    module.eval()

    x = (torch.randn(1, module.in_features, dtype=torch.float16) * 0.5).cuda()
    native = module._forward_pangolin(x)
    assert native is not None, "decode-shape padded planar module should route to Pangolin"
    ref = (x.cpu().float() @ ref_w[:, :module.padded_out_features].float()).to(torch.float16)
    assert torch.allclose(native.cpu().float(), ref.float(), atol=2e-2, rtol=1e-2)


def test_continuous_3bit_not_planar():
    """Continuous (non-GPTQ_P) 3-bit must stay continuous: format-aware validation."""
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import FORMAT

    cont = TorchLinear(
        bits=3, group_size=32, sym=True, desc_act=False,
        in_features=256, out_features=128, bias=False, register_buffers=True,
    )
    assert not cont.planar

    planar = TorchLinear(
        bits=3, group_size=32, sym=True, desc_act=False,
        in_features=256, out_features=48, bias=False,
        format=FORMAT.GPTQ_P, register_buffers=True,
    )
    assert planar.planar
    assert planar.padded_out_features == 64

    ok, err = TorchLinear.validate(
        bits=3, group_size=32, desc_act=False, sym=True,
        in_features=256, out_features=48, pack_dtype=torch.int32,
        format=FORMAT.GPTQ_P,
    )
    assert ok, err
