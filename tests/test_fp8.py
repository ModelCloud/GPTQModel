# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import re
import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.gptq import GPTQ  # noqa: E402


# ------------------------ Backend / HW detection ------------------------

def _is_cuda_build() -> bool:
    return torch.cuda.is_available() and torch.version.cuda is not None

def _is_rocm_build() -> bool:
    # In ROCm wheels, torch.version.hip is non-None; device interface is still "cuda"
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
    return sm is not None and sm >= 90  # SM90 == Hopper (H100/H200)

def _is_mi300_or_newer() -> bool:
    if not _is_rocm_build():
        return False
    try:
        name = torch.cuda.get_device_name(None)  # current device
    except Exception:
        return False
    # Be conservative: require MI300 in name.
    # If you deploy on other FP8-capable AMD parts, extend this regex.
    return bool(re.search(r"MI3\d{2}", name.upper()))

def _backend_name() -> str:
    if _is_cuda_build() and _is_hopper_or_newer():
        return "cuda"
    if _is_rocm_build() and _is_mi300_or_newer():
        return "rocm"
    return "unsupported"

BACKEND = _backend_name()

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
    pytest.mark.skipif(
        BACKEND == "unsupported",
        reason="No detected HW FP8 backend (need NVIDIA Hopper SM90+ or AMD MI300).",
    ),
    pytest.mark.skipif(
        len(FP8_DTYPES) == 0,
        reason="This PyTorch build exposes no FP8 dtypes.",
    ),
]

def _pick_fp8_dtype_prefer_e4m3():
    for name in ["float8_e4m3fn", "float8_e4m3fnuz", "float8_e4m3fn_fast"]:
        dt = getattr(torch, name, None)
        if dt is not None:
            return dt
    return FP8_DTYPES[0]

# ------------------------ Utilities ------------------------

def _relative_atol_for_fp8(dtype: torch.dtype) -> float:
    # Looser tolerance for e5m2
    name = str(dtype)
    if "e4m3" in name:
        return 6e-2
    if "e5m2" in name:
        return 1e-1
    return 1e-1

def _assert_close(a: torch.Tensor, b: torch.Tensor, fp8_dtype: torch.dtype, msg=""):
    atol = _relative_atol_for_fp8(fp8_dtype)
    a = a.detach().cpu().to(torch.float32)
    b = b.detach().cpu().to(torch.float32)
    assert torch.allclose(a, b, rtol=0.0, atol=atol), (
        f"{msg} | max|diff|={float((a-b).abs().max())}, atol={atol}"
    )

def _mk_linear(out_features: int, in_features: int, device: torch.device) -> nn.Linear:
    lin = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.float16)
    return lin

def _mk_fp8_weight_from_base(base_fp16: torch.Tensor, scale: torch.Tensor | float, fp8_dtype: torch.dtype) -> torch.Tensor:
    """
    Create FP8 weight that, after dequant with 'scale' (or its inverse), reconstructs base_fp16 approximately.
    Assumes base is [out, in].
    """
    if isinstance(scale, torch.Tensor):
        s = scale.to(device=base_fp16.device, dtype=base_fp16.dtype)
        if s.ndim == 0:
            scaled = base_fp16 / s
        elif s.ndim == 1 and s.numel() == base_fp16.shape[0]:
            scaled = base_fp16 / s.view(base_fp16.shape[0], *([1] * (base_fp16.ndim - 1)))
        elif s.ndim == 2 and s.shape[0] == base_fp16.shape[0]:
            out, in_ = base_fp16.shape
            G = s.shape[1]
            assert in_ % G == 0, "in_features must be divisible by group count"
            reps = in_ // G
            s_expand = s.repeat_interleave(reps, dim=1)
            scaled = base_fp16 / s_expand
        elif s.shape == base_fp16.shape:
            scaled = base_fp16 / s
        else:
            raise AssertionError(f"Unsupported scale shape {s.shape} for base {tuple(base_fp16.shape)}")
    else:
        scaled = base_fp16 / float(scale)

    return scaled.to(fp8_dtype)

def _current_device() -> torch.device:
    # In ROCm builds, device string is still "cuda"
    return torch.device("cuda", torch.cuda.current_device())

# ------------------------ Parametrization ------------------------

DEVICES = [pytest.param(_current_device(), id=BACKEND, marks=pytest.mark.cuda)]

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features", [(6, 8), (8, 16)])
def test_fp8_per_tensor_scale_inv_hw(device, fp8_dtype, out_features, in_features):
    torch.cuda.set_device(device.index if device.index is not None else 0)
    lin = _mk_linear(out_features, in_features, device=device)

    torch.manual_seed(0)
    base = torch.randn(out_features, in_features, device=device, dtype=torch.float16)

    scale = torch.tensor(0.25, device=device, dtype=torch.float16)
    w_fp8 = _mk_fp8_weight_from_base(base, scale, fp8_dtype)

    with torch.no_grad():
        lin.weight.copy_(w_fp8.to(lin.weight.dtype))
    lin.weight_scale_inv = (1.0 / scale).to(torch.float16)

    g = GPTQ(lin)
    # clone_module normalizes to [out, in] and dequants on the module's device
    W = g._clone_module(device=device)

    _assert_close(W, base, fp8_dtype, "HW per-tensor inverse scale dequant mismatch")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features", [(6, 8), (10, 12)])
def test_fp8_per_channel_scale_inv_hw(device, fp8_dtype, out_features, in_features):
    torch.cuda.set_device(device.index if device.index is not None else 0)
    lin = _mk_linear(out_features, in_features, device=device)

    torch.manual_seed(1)
    base = torch.randn(out_features, in_features, device=device, dtype=torch.float16)

    scale = torch.linspace(0.2, 0.6, out_features, device=device, dtype=torch.float16)
    w_fp8 = _mk_fp8_weight_from_base(base, scale, fp8_dtype)

    with torch.no_grad():
        lin.weight.copy_(w_fp8.to(lin.weight.dtype))
    lin.weight_scale_inv = (1.0 / scale).to(torch.float16)

    g = GPTQ(lin)
    W = g._clone_module(device=device)

    _assert_close(W, base, fp8_dtype, "HW per-channel inverse scale dequant mismatch")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features,G", [(6, 8, 2), (8, 16, 4)])
def test_fp8_per_group_scale_inv_hw(device, fp8_dtype, out_features, in_features, G):
    torch.cuda.set_device(device.index if device.index is not None else 0)
    assert in_features % G == 0

    lin = _mk_linear(out_features, in_features, device=device)

    torch.manual_seed(2)
    base = torch.randn(out_features, in_features, device=device, dtype=torch.float16)

    scale = (0.2 + 0.05 * torch.arange(G, device=device, dtype=torch.float16)).expand(out_features, G).clone()
    for o in range(out_features):
        scale[o] += (o % 3) * 0.03

    w_fp8 = _mk_fp8_weight_from_base(base, scale, fp8_dtype)

    with torch.no_grad():
        lin.weight.copy_(w_fp8.to(lin.weight.dtype))
    lin.weight_scale_inv = (1.0 / scale).to(torch.float16)

    g = GPTQ(lin)
    W = g._clone_module(device=device)

    _assert_close(W, base, fp8_dtype, "HW per-group inverse scale dequant mismatch")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features", [(6, 8), (8, 16)])
def test_fp8_scale_non_inverse_path_hw(device, fp8_dtype, out_features, in_features):
    torch.cuda.set_device(device.index if device.index is not None else 0)
    lin = _mk_linear(out_features, in_features, device=device)

    torch.manual_seed(3)
    base = torch.randn(out_features, in_features, device=device, dtype=torch.float16)

    scale = torch.linspace(0.3, 0.7, out_features, device=device, dtype=torch.float16)
    w_fp8 = _mk_fp8_weight_from_base(base, scale, fp8_dtype)

    with torch.no_grad():
        lin.weight.copy_(w_fp8.to(lin.weight.dtype))
    lin.weight_scale = scale  # direct (non-inverse) scale

    g = GPTQ(lin)
    W = g._clone_module(device=device)

    _assert_close(W, base, fp8_dtype, "HW non-inverse scale dequant mismatch")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fp8_dtype", [_pick_fp8_dtype_prefer_e4m3(), *_available_fp8_dtypes()])
@pytest.mark.parametrize("out_features,in_features", [(6, 8), (8, 16)])
def test_fp8_no_scale_fallback_hw(device, fp8_dtype, out_features, in_features):
    torch.cuda.set_device(device.index if device.index is not None else 0)
    lin = _mk_linear(out_features, in_features, device=device)

    torch.manual_seed(4)
    base = torch.randn(out_features, in_features, device=device, dtype=torch.float16)

    # No scale attrs -> expect simple FP8->FP16 cast
    w_fp8 = base.to(fp8_dtype)

    with torch.no_grad():
        lin.weight.copy_(w_fp8.to(lin.weight.dtype))

    g = GPTQ(lin)
    W = g._clone_module(device=device)

    expected = w_fp8.to(torch.float16)
    _assert_close(W, expected, fp8_dtype, "HW no-scale FP8 cast fallback mismatch")
