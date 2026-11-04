# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the launch/selection cases; the kernel-oracle cases need ROCm.
"""ROCm enablement for the fused GGUF Triton kernels.

The GGUF Triton kernels use plain ``tl.load``/``tl.dot`` and only the NVIDIA
capability specializations (Q1_0 u32 layout, fixed decode/prefill configs) are
CUDA-specific; ``_cuda_device_capability`` already returns ``None`` on ROCm, so
ROCm runs the generic launch configs.  These tests check the declaration, the
generic-config selection, and the numerics against an independent oracle.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.gguf import (
    GGUFTorchLinear,
    _dequantize_gguf_tensor_numpy,
)
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_rocm = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU required",
)


def test_gguf_triton_declares_rocm_support():
    assert DEVICE.ROCM in GGUFTritonKernel.SUPPORTS_DEVICES
    assert DEVICE.CUDA in GGUFTritonKernel.SUPPORTS_DEVICES


def test_gguf_triton_rocm_uses_generic_configs(monkeypatch):
    """No NVIDIA capability specialization may be selected on a ROCm host."""

    from gptqmodel.nn_modules.qlinear import gguf_triton as gguf_triton_module

    monkeypatch.setattr(gguf_triton_module, "IS_ROCM", True)
    gguf_triton_module._CUDA_DEVICE_CAPABILITY_CACHE.clear()
    device = torch.device("cuda:0")

    assert gguf_triton_module._cuda_device_capability(device) is None
    # Every capability-gated selector must answer "no specialization".
    assert not gguf_triton_module._select_q1_0_g128_u32_layout(
        capability=None, in_features=6144, out_features=2048
    )
    assert gguf_triton_module._select_q1_0_g128_u32_fixed_launch_config(
        capability=None, rows=1, in_features=6144, out_features=2048
    ) is None
    assert gguf_triton_module._select_q1_0_g128_fixed_launch_config(
        capability=None, rows=1, in_features=6144, cols=2048
    ) is None


def _build_pair(bits, in_features, out_features, seed=41):
    torch.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.03)

    torch_module = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=False,
    )
    torch_module.pack_original(linear, scales=None, zeros=None)
    torch_module.post_init()

    triton_module = GGUFTritonKernel(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    triton_module.load_state_dict(torch_module.state_dict(), strict=True)
    return torch_module, triton_module


@requires_rocm
@pytest.mark.parametrize(
    ("bits", "qtype", "in_features", "out_features"),
    [
        ("q4_k", "Q4_K", 512, 512),
        ("q5_k", "Q5_K", 512, 512),
        ("q6_k", "Q6_K", 512, 512),
        ("q1_0", "Q1_0", 1024, 512),
    ],
)
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_gguf_triton_rocm_matches_independent_oracle(
    bits, qtype, in_features, out_features, rows
):
    device = torch.device("cuda:0")
    torch_module, triton_module = _build_pair(bits, in_features, out_features)
    torch_module = torch_module.to(device).eval()
    triton_module = triton_module.to(device).eval()

    # Independent FP32 oracle over the stored GGUF blocks (numpy, not Triton).
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(
            torch_module.qweight.detach().cpu().numpy(), qtype
        )
    ).to(device)

    inputs = torch.randn(rows, in_features, device=device, dtype=torch.float16)
    truth = inputs.to(torch.float64) @ dense.t().to(torch.float64)
    rounded = truth.to(torch.float16).to(torch.float64)

    with torch.inference_mode():
        first = triton_module(inputs)
        for _ in range(3):
            repeated = triton_module(inputs)
    torch.cuda.synchronize(device)

    assert first.dtype == inputs.dtype
    assert torch.equal(first, repeated)

    # Arithmetic error against the FP64 truth (visible BF16/FP16 rounding is
    # reported separately via ``rounded``).
    arithmetic = (first.to(torch.float64) - truth).abs().max().item()
    scale = truth.abs().max().item()
    assert arithmetic / scale <= 2e-3, f"{qtype} arithmetic rel error {arithmetic / scale:.3e}"

    # Visible drift from the once-rounded reference stays inside the
    # documented 2e-3 inference budget (per-element ULP checks are undefined
    # at the tiny outputs of a random 512-wide dot product).
    delta = (first.to(torch.float64) - rounded).abs().max().item()
    assert delta / scale <= 2e-3, f"{qtype} visible drift {delta / scale:.3e}"
    assert first.dtype == torch.float16

    # Packed GGUF blocks and derived metadata are the only retained tensors.
    assert triton_module.qweight.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32)
    for name in ("dense_weight", "dequantized_weight", "weight", "_dense_cache"):
        assert getattr(triton_module, name, None) is None, f"persistent dense cache: {name}"


@requires_rocm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("bits", "qtype", "in_features", "out_features"),
    [("q4_k", "Q4_K", 4096, 4096), ("q6_k", "Q6_K", 4096, 11008)],
)
def test_gguf_triton_rocm_output_dtype_matches_input(bits, qtype, in_features, out_features, dtype):
    """Output dtype == input dtype and repeated forwards are bit-stable."""

    device = torch.device("cuda:0")
    torch_module, triton_module = _build_pair(bits, in_features, out_features)
    triton_module = triton_module.to(device).eval()

    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(torch_module.qweight.detach().cpu().numpy(), qtype)
    ).to(device)

    inputs = torch.randn(8, in_features, device=device, dtype=dtype)
    truth = inputs.to(torch.float64) @ dense.t().to(torch.float64)

    with torch.inference_mode():
        first = triton_module(inputs)
        outputs = [triton_module(inputs) for _ in range(3)]
    torch.cuda.synchronize(device)

    assert first.dtype == inputs.dtype
    for repeated in outputs:
        assert repeated.dtype == inputs.dtype
        assert torch.equal(first, repeated), "kernel is not deterministic across forwards"

    # GGUF dequantisation is exact in FP32, so the leftover error is the FP32
    # accumulation against an FP64 truth plus exactly one output rounding
    # step.  Budget it as 1.25 ULP of the once-rounded reference (scaled by the
    # output magnitude, since per-element ULP is undefined for near-zero
    # entries) -- never by comparing the kernel to itself.
    rounded = truth.to(dtype).to(torch.float64)
    delta = (first.to(torch.float64) - rounded).abs().max().item()
    ulp = torch.finfo(dtype).eps * rounded.abs().max().item()
    assert delta <= 1.25 * ulp + 1e-6, (
        f"{qtype}/{dtype} drift {delta:.3e} exceeds 1.25 ULP ({1.25 * ulp:.3e})"
    )
    if dtype == torch.float16:
        rel = (first.to(torch.float64) - truth).abs().max().item() / truth.abs().max().item()
        assert rel <= 2e-3, f"{qtype}/{dtype} rel error {rel:.3e}"
