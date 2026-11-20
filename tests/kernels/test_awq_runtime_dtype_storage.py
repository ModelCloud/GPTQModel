# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint scale/bias storage must survive forwards in another dtype.

Issue #4178: ``AwqGEMMLinear`` converted the registered ``scales`` and ``bias``
buffers in place on every forward. One BF16 forward on an FP16 checkpoint
permanently narrowed them, so later FP16 forwards drifted and a save-after-load
persisted the rounded values. The CPU int4pack transforms had the same mutation
for their packed affine metadata.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.awq.utils.packing_utils import AWQ_ORDER, pack_awq_columns


def _build_checkpoint(in_features, out_features, group_size, *, seed=0, sym=True):
    torch.manual_seed(seed)
    bits = 4
    groups = in_features // group_size
    codes = torch.randint(0, 1 << bits, (in_features, out_features), dtype=torch.int32)
    if sym:
        zeros = torch.full((groups, out_features), 1 << (bits - 1), dtype=torch.int32)
    else:
        zeros = torch.randint(0, 1 << bits, (groups, out_features), dtype=torch.int32)
    qweight = pack_awq_columns(codes, bits, order=AWQ_ORDER)
    qzeros = pack_awq_columns(zeros, bits, order=AWQ_ORDER)
    scales = (torch.rand(groups, out_features, dtype=torch.float32) * 0.02 + 0.001).to(torch.float16)
    bias = torch.randn(out_features, dtype=torch.float16) * 0.02
    return codes, zeros, qweight, qzeros, scales, bias


def _oracle(codes, zeros, scales, bias, x, group_size):
    """Decode the packed AWQ words independently and accumulate in FP64."""

    device = x.device
    groups = torch.arange(codes.shape[0], device=device) // group_size
    weight = (codes.double().to(device) - zeros.double().to(device)[groups]) * scales.double().to(device)[groups]
    output = x.double() @ weight
    if bias is not None:
        output = output + bias.double().to(device)
    return output


def _relerr(actual, expected):
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _copy_checkpoint(module, qweight, qzeros, scales, bias):
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    if bias is not None:
        module.bias.copy_(bias)
    module.post_init()


def _assert_checkpoint_buffers(module, scales, bias):
    assert module.scales.dtype == scales.dtype
    assert torch.equal(module.scales.cpu(), scales)
    assert module.bias.dtype == bias.dtype
    assert torch.equal(module.bias.cpu(), bias)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AwqGEMMLinear requires CUDA")
def test_awq_gemm_forward_does_not_rewrite_checkpoint_scale_or_bias():
    ok, error = AwqGEMMLinear.validate_once()
    if not ok:
        pytest.skip(f"native AWQ GEMM extension unavailable: {error}")

    codes, zeros, qweight, qzeros, scales, bias = _build_checkpoint(2048, 1024, 128)
    module = AwqGEMMLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=1024,
        bias=True,
        register_buffers=True,
    )
    _copy_checkpoint(module, qweight, qzeros, scales, bias)
    module = module.cuda().eval()

    _assert_checkpoint_buffers(module, scales, bias)

    x = torch.randn(1, 2048, dtype=torch.float16, device="cuda")
    expected = _oracle(codes, zeros, scales, bias, x, 128)

    first = module(x)
    assert first.dtype == torch.float16
    assert _relerr(first, expected) < 1e-3
    _assert_checkpoint_buffers(module, scales, bias)

    # contracts/kernel/forward-dtype.md: a later dtype switch is a caller bug.
    with pytest.raises(RuntimeError, match="forward-dtype.md"):
        module(x.to(torch.bfloat16))
    _assert_checkpoint_buffers(module, scales, bias)

    after = module(x)
    assert _relerr(after, expected) < 1e-3
    torch.testing.assert_close(after, first, rtol=0, atol=0)
    _assert_checkpoint_buffers(module, scales, bias)


def test_awq_torch_forward_does_not_rewrite_checkpoint_scale_or_bias():
    codes, zeros, qweight, qzeros, scales, bias = _build_checkpoint(2048, 1024, 128)
    module = AwqTorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=1024,
        bias=True,
        register_buffers=True,
    )
    _copy_checkpoint(module, qweight, qzeros, scales, bias)
    module.eval()

    _assert_checkpoint_buffers(module, scales, bias)

    x = torch.randn(1, 2048, dtype=torch.float16)
    expected = _oracle(codes, zeros, scales, bias, x, 128)

    first = module(x)
    assert first.dtype == torch.float16
    assert _relerr(first, expected) < 1e-3
    _assert_checkpoint_buffers(module, scales, bias)

    # contracts/kernel/forward-dtype.md: a later dtype switch is a caller bug.
    with pytest.raises(RuntimeError, match="forward-dtype.md"):
        module(x.to(torch.bfloat16))
    _assert_checkpoint_buffers(module, scales, bias)

    after = module(x)
    assert _relerr(after, expected) < 1e-3
    torch.testing.assert_close(after, first, rtol=0, atol=0)
    _assert_checkpoint_buffers(module, scales, bias)


def _int4pack_available():
    return hasattr(torch.ops.aten, "_convert_weight_to_int4pack_for_cpu") and hasattr(
        torch.ops.aten, "_weight_int4pack_mm_for_cpu"
    )


@pytest.mark.skipif(not _int4pack_available(), reason="CPU int4pack ops are unavailable")
@pytest.mark.parametrize(
    "kernel_cls",
    [
        pytest.param(TorchFusedAwqLinear, id="torch_fused_awq"),
        pytest.param(TorchAtenAwqLinear, id="torch_aten_awq"),
    ],
)
def test_cpu_int4pack_bf16_keeps_checkpoint_scales(kernel_cls):
    codes, zeros, qweight, qzeros, scales, bias = _build_checkpoint(256, 128, 32)
    module = kernel_cls(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=256,
        out_features=128,
        bias=True,
        register_buffers=True,
    )
    _copy_checkpoint(module, qweight, qzeros, scales, bias)
    module.eval()

    x = torch.randn(4, 256, dtype=torch.bfloat16)
    expected = _oracle(codes, zeros, scales, bias, x, 32)

    out = module(x)
    assert out.dtype == torch.bfloat16
    # The runtime affine metadata is FP32 for a non-BF16-representable FP16
    # scale; the registered checkpoint buffer itself must not be narrowed.
    assert torch.equal(module.scales, scales)
    assert module.scales.dtype == torch.float16
    assert _relerr(out, expected) < 2e-3


def _build_gptq_checkpoint(kernel_cls, in_features=256, out_features=128, group_size=32, seed=2):
    torch.manual_seed(seed)
    module = kernel_cls(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        format=FORMAT.GPTQ_V2,
        register_buffers=True,
    )
    rows = torch.arange(in_features)[:, None]
    columns = torch.arange(out_features)[None, :]
    codes = ((3 * rows + 5 * columns + 2) % 16).to(torch.int32)
    scales = (torch.rand(in_features // group_size, out_features, dtype=torch.float32) * 0.02 + 0.001).to(torch.float16)
    bias = torch.randn(out_features, dtype=torch.float16) * 0.02
    for lane in range(8):
        module.qweight |= (codes[lane::8] & 15) << (4 * lane)
    # 0x88888888 in every packed word: the symmetric zero code 8.
    module.qzeros.fill_(-2004318072)
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    module.g_idx.copy_(torch.arange(in_features, dtype=torch.int32) // group_size)
    module.qzero_format(2)
    module.post_init()
    module.eval()
    return module, codes, scales, bias


@pytest.mark.skipif(not _int4pack_available(), reason="CPU int4pack ops are unavailable")
@pytest.mark.parametrize(
    "kernel_cls",
    [
        pytest.param(TorchFusedLinear, id="torch_fused"),
        pytest.param(TorchAtenLinear, id="torch_aten"),
    ],
)
def test_cpu_int4pack_gptq_bf16_keeps_checkpoint_scales(kernel_cls):
    module, codes, scales, bias = _build_gptq_checkpoint(kernel_cls)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    zeros = torch.full((codes.shape[0] // 32, 128), 8, dtype=torch.int32)
    expected = _oracle(codes, zeros, scales, bias, x, 32)

    out = module(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(module.scales, scales)
    assert module.scales.dtype == torch.float16
    assert _relerr(out, expected) < 2e-3
