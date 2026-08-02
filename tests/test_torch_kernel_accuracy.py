# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Accuracy tests for the merged Torch kernel (continuous + planar layouts).

Verifies, for every supported bit width and both checkpoint layouts:
1. dequantize_weight() bit-exactly matches the logical-code dequant reference
2. forward() matches x @ W_ref against the dequantized reference weights
3. planar (gptq_p) and continuous layouts produce identical forward outputs
   for the widths that support both (2, 3, 4, 8)
4. desc_act-style shuffled g_idx and sym=True variants stay accurate
5. float16 and bfloat16 activation dtypes
"""

import os


# Keep this suite on CPU so it works on CPU-only runners.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT


pytestmark = [pytest.mark.cpu]

ALL_BITS = (2, 3, 4, 5, 6, 7, 8)
DUAL_LAYOUT_BITS = (2, 3, 4, 8)  # widths supporting both continuous and planar


def _format_cases():
    cases = []
    for bits in ALL_BITS:
        cases.append((bits, FORMAT.GPTQ_P))
        if bits in DUAL_LAYOUT_BITS:
            cases.append((bits, FORMAT.GPTQ_V2))
    return cases


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


def _new_module(bits: int, fmt: FORMAT, in_features: int, out_features: int,
                group_size: int, desc_act: bool = False, sym: bool = False) -> TorchLinear:
    return TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        format=fmt,
        register_buffers=False,
    )


def _reference_weight(linear: nn.Linear, scales: torch.Tensor, zeros: torch.Tensor,
                      g_idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Dequantized [in, out] reference built from logical codes (packer rounding)."""
    maxq = (1 << bits) - 1
    weight = linear.weight.data  # [out, in]
    scale_full = scales[:, g_idx.long()]
    zero_full = zeros[:, g_idx.long()]
    codes = torch.round((weight + zero_full * scale_full) / scale_full).clamp(0, maxq)
    # Module stores scales as float16; mirror that precision.
    ref = (codes - zero_full) * scale_full.to(torch.float16).float()
    return ref.T.contiguous()  # [in, out]


def _packed_module_and_reference(bits: int, fmt: FORMAT, in_features: int = 64,
                                 out_features: int = 32, group_size: int = 32,
                                 desc_act: bool = False, seed: int = 0):
    linear, scales, zeros, g_idx = _make_inputs(
        bits, in_features, out_features, group_size, desc_act=desc_act, seed=seed
    )
    module = _new_module(bits, fmt, in_features, out_features, group_size, desc_act=desc_act)
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())
    module._init_wf_unsqueeze_buffers()
    ref = _reference_weight(linear, scales, zeros, g_idx, bits)
    return module, ref, linear


@pytest.mark.parametrize("bits,fmt", _format_cases())
def test_dequantize_weight_matches_reference(bits: int, fmt: FORMAT):
    module, ref, _ = _packed_module_and_reference(bits, fmt)
    dequant = module.dequantize_weight().float()
    assert dequant.shape == ref.shape
    assert torch.allclose(dequant, ref, atol=1e-4, rtol=0)


@pytest.mark.parametrize("bits,fmt", _format_cases())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_matches_reference(bits: int, fmt: FORMAT, dtype: torch.dtype):
    module, ref, linear = _packed_module_and_reference(bits, fmt)
    module.eval()

    torch.manual_seed(bits)
    x = torch.randn(4, 64, dtype=dtype) * 0.5
    with torch.inference_mode():
        out = module(x)

    ref_out = x.float() @ ref + linear.bias.data.float()
    assert out.shape == (4, 32)
    atol = 5e-3 if dtype == torch.float16 else 3e-2
    assert torch.allclose(out.float(), ref_out, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("bits,fmt", _format_cases())
def test_forward_batched_shapes(bits: int, fmt: FORMAT):
    module, ref, linear = _packed_module_and_reference(bits, fmt)
    module.eval()

    torch.manual_seed(bits)
    x = torch.randn(2, 3, 64, dtype=torch.float16) * 0.5
    with torch.inference_mode():
        out = module(x)

    ref_out = x.float().reshape(-1, 64) @ ref + linear.bias.data.float()
    assert out.shape == (2, 3, 32)
    assert torch.allclose(out.float().reshape(-1, 32), ref_out, atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("bits", DUAL_LAYOUT_BITS)
def test_planar_and_continuous_forward_identical(bits: int):
    linear, scales, zeros, g_idx = _make_inputs(bits, 64, 32, 32)

    m_planar = _new_module(bits, FORMAT.GPTQ_P, 64, 32, 32)
    m_planar.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())
    m_planar._init_wf_unsqueeze_buffers()

    m_continuous = _new_module(bits, FORMAT.GPTQ_V2, 64, 32, 32)
    m_continuous.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())
    m_continuous._init_wf_unsqueeze_buffers()

    m_planar.eval()
    m_continuous.eval()

    assert torch.equal(m_planar.dequantize_weight(), m_continuous.dequantize_weight())

    torch.manual_seed(bits)
    x = torch.randn(4, 64, dtype=torch.float16) * 0.5
    with torch.inference_mode():
        out_planar = m_planar(x)
        out_continuous = m_continuous(x)
    assert torch.equal(out_planar, out_continuous)


@pytest.mark.parametrize("bits", ALL_BITS)
def test_forward_desc_act_shuffled_g_idx(bits: int):
    fmt = FORMAT.GPTQ_P if bits in (3, 5, 6, 7) else FORMAT.GPTQ_V2
    module, ref, linear = _packed_module_and_reference(bits, fmt, desc_act=True, seed=7)
    module.eval()

    torch.manual_seed(bits)
    x = torch.randn(4, 64, dtype=torch.float16) * 0.5
    with torch.inference_mode():
        out = module(x)

    ref_out = x.float() @ ref + linear.bias.data.float()
    assert torch.allclose(out.float(), ref_out, atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("bits", ALL_BITS)
def test_forward_sym_zero_point(bits: int):
    fmt = FORMAT.GPTQ_P if bits in (3, 5, 6, 7) else FORMAT.GPTQ_V2
    in_features, out_features, group_size = 64, 32, 32
    maxq = (1 << bits) - 1

    torch.manual_seed(100 + bits)
    linear = nn.Linear(in_features, out_features, bias=True)
    groups = in_features // group_size
    scales = torch.rand(out_features, groups) * 0.01 + 0.005
    # Symmetric quantization centers the zero point.
    zeros = torch.full((out_features, groups), float((maxq + 1) // 2))
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    module = _new_module(bits, fmt, in_features, out_features, group_size, sym=True)
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())
    module._init_wf_unsqueeze_buffers()
    module.eval()

    ref = _reference_weight(linear, scales, zeros, g_idx, bits)
    assert torch.allclose(module.dequantize_weight().float(), ref, atol=1e-4, rtol=0)

    x = torch.randn(4, in_features, dtype=torch.float16) * 0.5
    with torch.inference_mode():
        out = module(x)
    ref_out = x.float() @ ref + linear.bias.data.float()
    assert torch.allclose(out.float(), ref_out, atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("bits", ALL_BITS)
def test_forward_larger_shapes(bits: int):
    fmt = FORMAT.GPTQ_P if bits in (3, 5, 6, 7) else FORMAT.GPTQ_V2
    in_features, out_features, group_size = 256, 128, 64
    module, ref, linear = _packed_module_and_reference(
        bits, fmt, in_features=in_features, out_features=out_features,
        group_size=group_size, seed=42,
    )
    module.eval()

    torch.manual_seed(bits)
    x = torch.randn(8, in_features, dtype=torch.float16) * 0.5
    with torch.inference_mode():
        out = module(x)

    ref_out = x.float() @ ref + linear.bias.data.float()
    assert out.shape == (8, out_features)
    assert torch.allclose(out.float(), ref_out, atol=2e-2, rtol=1e-2)
