# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the mocked launch-table cases; the kernel-oracle cases carry
# ``pytest.mark.cuda`` and skip unless the device is an Ada (SM 8.9) GPU.
"""Ada (RTX 4090, SM 8.9) Triton dequant: launch table + Torch-oracle A/B.

The oracle decodes the packed int32 words with Torch integer shifts on the CPU
and never calls the Triton kernel.  Verified for bits 2/4/8 in FP16 and BF16.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.triton_utils.dequant import (
    _DEFAULT_DEQUANT_LAUNCH,
    _DEQUANT_LAUNCH_CONFIGS,
    DequantLaunch,
    dequant,
    launch_config,
)
from gptqmodel.utils import arch


TOL = 2e-3


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
# CPU-only: launch-table dispatch.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("arch_id", "expected"),
    [
        ("sm_89", DequantLaunch(x_block=512, num_warps=1, num_stages=2)),
        ((8, 9), DequantLaunch(x_block=512, num_warps=1, num_stages=2)),
        ("gfx950", DequantLaunch(x_block=1024, num_warps=1, num_stages=1)),
        ("gfx1201", DequantLaunch(x_block=2048, num_warps=4, num_stages=1)),
        ("gfx942", _DEFAULT_DEQUANT_LAUNCH),
        ("sm_80", _DEFAULT_DEQUANT_LAUNCH),
        ("sm_90", _DEFAULT_DEQUANT_LAUNCH),
        ("", _DEFAULT_DEQUANT_LAUNCH),
        (None, _DEFAULT_DEQUANT_LAUNCH),
    ],
)
def test_dequant_launch_routes_ada(arch_id, expected):
    assert launch_config(arch_id) == expected


def test_dequant_ada_is_an_explicit_measured_entry():
    assert "ada" in _DEQUANT_LAUNCH_CONFIGS
    assert "rdna4" in _DEQUANT_LAUNCH_CONFIGS and "cdna4" in _DEQUANT_LAUNCH_CONFIGS
    # The Ada tile is a measurement, not an inherited default.
    assert launch_config("sm_89") == _DEQUANT_LAUNCH_CONFIGS["ada"]
    assert launch_config("sm_89") != _DEFAULT_DEQUANT_LAUNCH


# --------------------------------------------------------------------------- #
# Kernel-oracle cases (real hardware only).
# --------------------------------------------------------------------------- #

def _make_module(in_features, out_features, dtype, group_size=128,
                 desc_act=False, bits=4):
    module = TorchLinear(
        bits=bits, group_size=group_size, sym=True, desc_act=desc_act,
        in_features=in_features, out_features=out_features,
        pack_dtype=torch.int32, bias=False, register_buffers=True,
    ).to(device="cuda", dtype=dtype).eval()
    module.optimize = lambda *args, **kwargs: None
    module.qzero_format(format=2)
    with torch.no_grad():
        module.qweight.copy_(torch.randint(-2**31, 2**31, module.qweight.shape, device="cuda", dtype=torch.int32))
        module.qzeros.copy_(torch.randint(-2**31, 2**31, module.qzeros.shape, device="cuda", dtype=torch.int32))
        module.scales.uniform_(0.001, 0.02)
        if desc_act:
            module.g_idx.copy_(module.g_idx[torch.randperm(in_features, device="cuda")])
            module.g_idx[0] = -1
    module.post_init()
    return module


def _oracle_weight(module, dtype):
    """Independent CPU int-shift decode; never calls the Triton kernel."""
    pack_factor = 32 // module.bits
    maxq = (1 << module.bits) - 1
    qweight = module.qweight.detach().cpu().to(torch.int64)
    qzeros = module.qzeros.detach().cpu().to(torch.int64)
    scales = module.scales.detach().cpu().to(torch.float64)
    group_index = module.g_idx.detach().cpu().to(torch.int64)
    shifts = torch.arange(pack_factor, dtype=torch.int64) * module.bits
    codes = ((qweight.unsqueeze(1) >> shifts[None, :, None]) & maxq)
    codes = codes.reshape(module.in_features, module.out_features)
    zeros = ((qzeros.unsqueeze(-1) >> shifts[None, None, :]) & maxq)
    zeros = zeros.reshape(module.scales.shape[0], module.out_features)
    weights = (codes.to(torch.float64) - zeros[group_index].to(torch.float64)) * scales[group_index]
    return weights.to(dtype)


def _dense_weight_candidates(module):
    shape = (module.in_features, module.out_features)
    found = []
    for name, value in vars(module).items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        if value.dim() == 2 and tuple(value.shape) in (shape, (shape[1], shape[0])):
            found.append(f"{name}{tuple(value.shape)}")
    for name, value in module.named_buffers():
        if value.is_floating_point() and value.dim() == 2 and tuple(value.shape) in (shape, (shape[1], shape[0])):
            found.append(f"buffer:{name}{tuple(value.shape)}")
    return found


_SHAPES = [
    (896, 128, 128),
    (2048, 1024, 128),
    (4096, 4096, 128),
    (5120, 12288, 128),
]


@requires_ada
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("in_features,out_features,group_size", _SHAPES)
def test_ada_dequant_matches_independent_torch_oracle(
    dtype, bits, in_features, out_features, group_size,
):
    torch.manual_seed(1291)
    module = _make_module(in_features, out_features, dtype, group_size, bits=bits)
    expected = _oracle_weight(module, dtype)

    packed_bits = 32
    with torch.inference_mode():
        out = dequant(
            dtype,
            module.qweight,
            module.scales,
            module.qzeros,
            module.g_idx,
            bits,
            packed_bits,
            module.maxq,
        )
    torch.cuda.synchronize()
    assert out.shape == (in_features, out_features)
    assert out.dtype == dtype
    assert _dense_weight_candidates(module) == [], "dequant retained a dense weight"
    torch.testing.assert_close(out.detach().cpu(), expected, rtol=TOL, atol=TOL)

    # The module-level path must produce the same tensor and must not persist a
    # dense weight across repeated calls (no expanded-weight cache).
    with torch.inference_mode():
        first = module.dequantize_weight()
        second = module.dequantize_weight()
    torch.cuda.synchronize()
    assert first.dtype == dtype and second.dtype == dtype
    assert _dense_weight_candidates(module) == [], "repeated dequant retained a dense weight"
    torch.testing.assert_close(first.detach().cpu(), expected, rtol=TOL, atol=TOL)
    torch.testing.assert_close(second.detach().cpu(), expected, rtol=TOL, atol=TOL)


@requires_ada
@pytest.mark.cuda
def test_ada_dequant_dispatches_triton_with_ada_tile():
    module = _make_module(2048, 1024, torch.float16, bits=4)
    assert module._can_use_triton_dequant() is True
    assert launch_config(module.qweight.device) == _DEQUANT_LAUNCH_CONFIGS["ada"]
