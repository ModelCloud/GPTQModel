# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

import re
import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.gptq import GPTQ


# ------------------------ Backend / HW detection ------------------------

def _is_cuda_build() -> bool:
    return torch.cuda.is_available() and torch.version.cuda is not None

def _is_rocm_build() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None

def _nvidia_sm() -> int | None:
    if not _is_cuda_build():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except Exception:
        return None

def _is_hopper_or_newer() -> bool:
    sm = _nvidia_sm()
    return sm is not None and sm >= 90  # SM90 = Hopper/Blackwell era

def _is_mi300_or_newer() -> bool:
    if not _is_rocm_build():
        return False
    try:
        name = torch.cuda.get_device_name(None)
    except Exception:
        return False
    return bool(re.search(r"MI3\d{2}", name.upper()))

BACKEND_SUPPORTED = (_is_hopper_or_newer() or _is_mi300_or_newer())

# ------------------------ FP8 dtype inventory ------------------------

def _available_fp8_dtypes():
    names = [
        "float8_e4m3fn", "float8_e4m3fnuz",
        "float8_e5m2",   "float8_e5m2fnuz",
        "float8_e4m3fn_fast", "float8_e5m2_fast",
    ]
    dts = []
    for n in names:
        dt = getattr(torch, n, None)
        if dt is not None:
            dts.append(dt)
    return tuple(dts)

FP8_DTYPES = _available_fp8_dtypes()

pytestmark = [
    pytest.mark.skipif(not BACKEND_SUPPORTED, reason="Need SM90+ or MI300-class for HW FP8 path."),
    pytest.mark.skipif(len(FP8_DTYPES) == 0, reason="This PyTorch build exposes no FP8 dtypes."),
]

def _pick_fp8_dtype_prefer_e4m3():
    for name in ["float8_e4m3fn", "float8_e4m3fnuz", "float8_e4m3fn_fast"]:
        dt = getattr(torch, name, None)
        if dt is not None:
            return dt
    return FP8_DTYPES[0]

# ------------------------ Utilities ------------------------

def _device() -> torch.device:
    idx = torch.cuda.current_device()
    return torch.device("cuda", idx)

def _mk_linear(out_features: int, in_features: int, device: torch.device) -> nn.Linear:
    # We'll replace .weight param anyway.
    return nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.float16)

def _expand_scale_like_out_in(base_oi: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Broadcast scale to [out, in] for reference building, matching gptq logic."""
    s = scale.to(device=base_oi.device, dtype=base_oi.dtype)
    if s.ndim == 0:
        return s
    if s.ndim == 1 and s.numel() == base_oi.shape[0]:
        return s.view(base_oi.shape[0], *([1] * (base_oi.ndim - 1)))
    if s.ndim == 2 and s.shape[0] == base_oi.shape[0]:
        out, in_ = base_oi.shape
        G = s.shape[1]
        assert in_ % G == 0, "in_features must be divisible by group count"
        reps = in_ // G
        return s.repeat_interleave(reps, dim=1)
    if s.shape == base_oi.shape:
        return s
    raise AssertionError(f"Unsupported scale shape {tuple(s.shape)} for base {tuple(base_oi.shape)}")

def _mk_fp8_weight_from_base(base_fp16: torch.Tensor, scale: torch.Tensor | float, fp8_dtype: torch.dtype) -> torch.Tensor:
    """Simulate packing: divide by scale, cast to FP8."""
    if isinstance(scale, torch.Tensor):
        s = _expand_scale_like_out_in(base_fp16, scale)
        scaled = base_fp16 / s
    else:
        scaled = base_fp16 / float(scale)
    return scaled.to(fp8_dtype)

def _reference_dequant_from_fp8_param(
    w_fp8: torch.Tensor,
    *,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor | float | None,
    is_inverse: bool,
) -> torch.Tensor:
    """
    Build the *exact* expected dequant result:
      expected = (w_fp8.to(fp16)) * (scale or 1/scale_inv)  [with proper broadcasting]
    """
    ref = w_fp8.to(torch.float16)
    if scale is not None:
        if isinstance(scale, torch.Tensor):
            s = scale.to(device=w_fp8.device, dtype=torch.float16)
        else:
            s = torch.tensor(scale, device=w_fp8.device, dtype=torch.float16)
        if is_inverse:
            # avoid divide-by-zero surprises
            s = 1.0 / s.clamp_min(1e-8)
        s = _expand_scale_like_out_in(ref, s) if s.ndim > 0 else s
        ref = ref * s
    return ref

def _assert_eq_close(a: torch.Tensor, b: torch.Tensor, msg=""):
    # very tight atol because both sides are the *same* FP8 round-trip graph
    a32 = a.detach().cpu().to(torch.float32)
    b32 = b.detach().cpu().to(torch.float32)
    assert torch.allclose(a32, b32, rtol=0.0, atol=1e-6), (
        f"{msg} | max|diff|={float((a32-b32).abs().max())}"
    )

# ------------------------ Tests ------------------------

@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features", [(6, 8), (8, 16)])
def test_fp8_per_tensor_scale_inv_hw(fp8_dtype, out_features, in_features):
    dev = _device()
    torch.cuda.set_device(dev.index or 0)
    lin = _mk_linear(out_features, in_features, device=dev)

    torch.manual_seed(0)
    base = torch.randn(out_features, in_features, device=dev, dtype=torch.float16)

    scale = torch.tensor(0.25, device=dev, dtype=torch.float16)
    w_fp8 = _mk_fp8_weight_from_base(base, scale, fp8_dtype)

    with torch.no_grad():
        lin.weight = nn.Parameter(w_fp8)  # keep FP8 dtype
    lin.weight_scale_inv = (1.0 / scale).to(torch.float16)

    g = GPTQ(lin)
    W = g._clone_module(device=dev)

    expected = _reference_dequant_from_fp8_param(
        w_fp8, fp8_dtype=fp8_dtype, scale=lin.weight_scale_inv, is_inverse=True
    )
    _assert_eq_close(W, expected, "per-tensor inverse scale dequant mismatch")


@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features", [(6, 8), (10, 12)])
def test_fp8_per_channel_scale_inv_hw(fp8_dtype, out_features, in_features):
    dev = _device()
    torch.cuda.set_device(dev.index or 0)
    lin = _mk_linear(out_features, in_features, device=dev)

    torch.manual_seed(1)
    base = torch.randn(out_features, in_features, device=dev, dtype=torch.float16)

    scale = torch.linspace(0.2, 0.6, out_features, device=dev, dtype=torch.float16)
    w_fp8 = _mk_fp8_weight_from_base(base, scale, fp8_dtype)

    with torch.no_grad():
        lin.weight = nn.Parameter(w_fp8)
    lin.weight_scale_inv = (1.0 / scale).to(torch.float16)

    g = GPTQ(lin)
    W = g._clone_module(device=dev)

    expected = _reference_dequant_from_fp8_param(
        w_fp8, fp8_dtype=fp8_dtype, scale=lin.weight_scale_inv, is_inverse=True
    )
    _assert_eq_close(W, expected, "per-channel inverse scale dequant mismatch")


@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features,G", [(6, 8, 2), (8, 16, 4)])
def test_fp8_per_group_scale_inv_hw(fp8_dtype, out_features, in_features, G):
    assert in_features % G == 0
    dev = _device()
    torch.cuda.set_device(dev.index or 0)
    lin = _mk_linear(out_features, in_features, device=dev)

    torch.manual_seed(2)
    base = torch.randn(out_features, in_features, device=dev, dtype=torch.float16)

    scale = (0.2 + 0.05 * torch.arange(G, device=dev, dtype=torch.float16)).expand(out_features, G).clone()
    for o in range(out_features):
        scale[o] += (o % 3) * 0.03

    w_fp8 = _mk_fp8_weight_from_base(base, scale, fp8_dtype)

    with torch.no_grad():
        lin.weight = nn.Parameter(w_fp8)
    lin.weight_scale_inv = (1.0 / scale).to(torch.float16)

    g = GPTQ(lin)
    W = g._clone_module(device=dev)

    expected = _reference_dequant_from_fp8_param(
        w_fp8, fp8_dtype=fp8_dtype, scale=lin.weight_scale_inv, is_inverse=True
    )
    _assert_eq_close(W, expected, "per-group inverse scale dequant mismatch")


@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features", [(6, 8), (8, 16)])
def test_fp8_scale_non_inverse_path_hw(fp8_dtype, out_features, in_features):
    dev = _device()
    torch.cuda.set_device(dev.index or 0)
    lin = _mk_linear(out_features, in_features, device=dev)

    torch.manual_seed(3)
    base = torch.randn(out_features, in_features, device=dev, dtype=torch.float16)

    scale = torch.linspace(0.3, 0.7, out_features, device=dev, dtype=torch.float16)
    w_fp8 = _mk_fp8_weight_from_base(base, scale, fp8_dtype)

    with torch.no_grad():
        lin.weight = nn.Parameter(w_fp8)
    lin.weight_scale = scale  # direct scale

    g = GPTQ(lin)
    W = g._clone_module(device=dev)

    expected = _reference_dequant_from_fp8_param(
        w_fp8, fp8_dtype=fp8_dtype, scale=lin.weight_scale, is_inverse=False
    )
    _assert_eq_close(W, expected, "non-inverse scale dequant mismatch")


@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features", [(6, 8), (8, 16)])
def test_fp8_no_scale_fallback_hw(fp8_dtype, out_features, in_features):
    dev = _device()
    torch.cuda.set_device(dev.index or 0)
    lin = _mk_linear(out_features, in_features, device=dev)

    torch.manual_seed(4)
    base = torch.randn(out_features, in_features, device=dev, dtype=torch.float16)

    # No scale attrs -> expected is just FP8->FP16 cast
    w_fp8 = base.to(fp8_dtype)

    with torch.no_grad():
        lin.weight = nn.Parameter(w_fp8)

    g = GPTQ(lin)
    W = g._clone_module(device=dev)

    expected = _reference_dequant_from_fp8_param(
        w_fp8, fp8_dtype=fp8_dtype, scale=None, is_inverse=False
    )
    _assert_eq_close(W, expected, "no-scale FP8 cast fallback mismatch")
