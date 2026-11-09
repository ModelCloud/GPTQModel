# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the mocked selector cases; the kernel A/B cases need an Ada GPU.
"""SM89 (RTX 4090) fixed launch configs for the fused GGUF Q4_K/Q5_K/Q6_K kernels.

Two kinds of coverage:

* mocked selector cases (no GPU) prove that only the measured SM89 ``(K, N)``
  shapes get a fixed tile and that SM80/other families keep the autotuned path;
* on an Ada GPU, the fixed launch must be **bit-identical** to the kernel's own
  autotuned launch for the same inputs, and repeated forwards must not retain a
  dense weight.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear import gguf_triton as gguf_triton_module
from gptqmodel.nn_modules.qlinear.gguf import _dequantize_gguf_tensor_numpy
from gptqmodel.nn_modules.qlinear.gguf_triton import (
    _GGUF_SM89_QK_SHAPES,
    _select_q4_k_fixed_launch_config,
    _select_q5_k_fixed_launch_config,
    _select_q6_k_fixed_launch_config,
    fused_q4_k_matmul,
    fused_q5_k_matmul,
    fused_q6_k_matmul,
)
from gptqmodel.utils import arch


requires_ada = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip is None
        and arch.is_ada(0)
    ),
    reason="Ada (SM 8.9) CUDA GPU required",
)


# --------------------------------------------------------------------------- #
# CPU-only: selector.
# --------------------------------------------------------------------------- #

def _select(qtype: str, rows: int, k: int, n: int, capability=(8, 9)):
    selector = {
        "Q4_K": _select_q4_k_fixed_launch_config,
        "Q5_K": _select_q5_k_fixed_launch_config,
        "Q6_K": _select_q6_k_fixed_launch_config,
    }[qtype]
    return selector(capability=capability, rows=rows, in_features=k, out_features=n)


@pytest.mark.parametrize(
    ("qtype", "rows", "shape"),
    [
        ("Q4_K", 1, (1024, 128)),
        ("Q4_K", 8, (1024, 128)),
        ("Q4_K", 64, (2048, 2048)),
        ("Q4_K", 127, (4096, 4096)),
        ("Q4_K", 8, (4096, 11008)),
        ("Q5_K", 1, (2048, 2048)),
        ("Q5_K", 8, (4096, 4096)),
        ("Q5_K", 127, (4096, 11008)),
        ("Q6_K", 1, (1024, 128)),
        ("Q6_K", 8, (2048, 1024)),
        ("Q6_K", 64, (2048, 2048)),
        ("Q6_K", 127, (4096, 4096)),
    ],
)
def test_sm89_selector_returns_the_measured_entry(qtype, rows, shape):
    config = _select(qtype, rows, *shape)
    assert config is not None
    assert set(config) == {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "num_warps", "num_stages"}
    # The returned dict is a copy, never a live table entry.
    config["BLOCK_SIZE_M"] = 1
    assert _GGUF_SM89_QK_SHAPES[qtype][
        gguf_triton_module._gguf_qk_row_band(rows)
    ][shape]["BLOCK_SIZE_M"] != 1


@pytest.mark.parametrize(
    ("capability", "rows", "shape"),
    [
        # Only SM 8.9 is tuned; SM80/SM86/SM90/SM100 keep the autotuner.
        ((8, 0), 1, (1024, 128)),
        ((8, 6), 8, (1024, 128)),
        ((9, 0), 64, (4096, 4096)),
        ((10, 0), 8, (4096, 11008)),
        (None, 1, (2048, 2048)),
        # Unmeasured row bands.
        ((8, 9), 2, (1024, 128)),
        ((8, 9), 7, (1024, 128)),
        ((8, 9), 128, (4096, 4096)),
        # Unmeasured (K, N) shapes even on SM89 / measured rows.
        ((8, 9), 1, (4096, 4096)),
        ((8, 9), 8, (5120, 12288)),
        ((8, 9), 1, (512, 512)),
        # Unpadded width: the dispatcher passes ``padded_in_features``, so the
        # raw 896 key must never match (896 pads to 1024).
        ((8, 9), 1, (896, 128)),
        ((8, 9), 8, (896, 128)),
    ],
)
def test_selector_keeps_autotuned_outside_the_measured_grid(capability, rows, shape):
    for qtype in ("Q4_K", "Q5_K", "Q6_K"):
        assert _select(qtype, rows, *shape, capability=capability) is None


def test_unpadded_narrow_shape_never_matches():
    # 896 is padded to 1024 by the GGUF block size before dispatch, so the raw
    # width is not a legal key and must fall through to the autotuner.
    assert _select("Q4_K", 1, 896, 128) is None
    assert _select("Q6_K", 1, 896, 128) is None


def test_q5_k_has_no_unmeasured_m1_wide_entry():
    # Q5_K 4096-wide decode lost against the autotuner, so it must stay unwired.
    assert _select("Q5_K", 1, 4096, 4096) is None
    assert _select("Q5_K", 1, 4096, 11008) is None


# --------------------------------------------------------------------------- #
# Kernel A/B (real hardware): fixed launch == autotuned launch, bit for bit.
# --------------------------------------------------------------------------- #

def _build_modules(bits, in_features, out_features, seed=41):
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear

    torch.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.03)

    torch_module = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        bias=False, register_buffers=False,
    )
    torch_module.pack_original(linear, scales=None, zeros=None)
    torch_module.post_init()

    triton_module = gguf_triton_module.GGUFTritonKernel(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        bias=False, register_buffers=True,
    )
    triton_module.load_state_dict(torch_module.state_dict(), strict=True)
    return torch_module, triton_module.to("cuda").eval()


@requires_ada
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("bits", "qtype", "shape"),
    [
        ("q4_k", "Q4_K", (1024, 128)),
        ("q4_k", "Q4_K", (2048, 2048)),
        ("q4_k", "Q4_K", (4096, 4096)),
        ("q5_k", "Q5_K", (2048, 2048)),
        ("q5_k", "Q5_K", (4096, 4096)),
        ("q6_k", "Q6_K", (1024, 128)),
        ("q6_k", "Q6_K", (4096, 11008)),
    ],
)
@pytest.mark.parametrize("rows", [8, 64])
def test_ada_fixed_launch_is_bit_identical_to_autotuned(bits, qtype, shape, rows):
    in_features, out_features = shape
    torch_module, module = _build_modules(bits, in_features, out_features)

    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(torch_module.qweight.detach().cpu().numpy(), qtype)
    ).to("cuda")
    inputs = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)

    cache = module._get_triton_cache(inputs.device)
    assert _select(qtype, rows, in_features, out_features) is not None

    with torch.inference_mode():
        fixed = module(inputs)
        # The same packed blocks through the autotuned generic launcher.
        if qtype == "Q4_K":
            autotuned = fused_q4_k_matmul(inputs, cache["qs"], cache["scale"], cache["min"])
        elif qtype == "Q5_K":
            autotuned = fused_q5_k_matmul(
                inputs, cache["qs"], cache["qh"], cache["scale"], cache["min"]
            )
        else:
            autotuned = fused_q6_k_matmul(inputs, cache["ql"], cache["qh"], cache["scale"])
        repeated = [module(inputs) for _ in range(3)]
    torch.cuda.synchronize()

    assert fixed.dtype == inputs.dtype
    # Exact bit equality: the fixed tile only changes the launch config; the
    # kernel arithmetic is identical, so any difference is a real regression.
    assert torch.equal(fixed, autotuned), f"{qtype} fixed tile changed the numerics"
    for again in repeated:
        assert torch.equal(fixed, again), f"{qtype} forward is not deterministic"

    # Independent FP64 oracle over the stored blocks (never the Triton kernel).
    truth = inputs.to(torch.float64) @ dense.t().to(torch.float64)
    rel = (fixed.to(torch.float64) - truth).abs().max().item() / truth.abs().max().item()
    assert rel <= 2e-3, f"{qtype} rel error {rel:.3e}"

    # Only the packed GGUF blocks + derived block metadata may persist.
    for name in ("dense_weight", "dequantized_weight", "weight", "_dense_cache"):
        assert getattr(module, name, None) is None, f"persistent dense cache: {name}"
    for entry in module._gguf_triton_cache.values():
        for value in entry.values():
            if torch.is_tensor(value):
                assert tuple(value.shape) != (in_features, out_features) or value.dtype in (
                    torch.uint8, torch.int8, torch.int16, torch.int32,
                )
