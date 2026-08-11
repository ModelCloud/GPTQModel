# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""CPU correctness tests for the planar (gptq_p) Pangolin GEMV kernel."""

import os
from contextlib import ExitStack, contextmanager

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import FORMAT
from gptqmodel.utils.pangolin import (
    PANGOLIN_SUPPORTED_M,
    ensure_pangolin_cpu_runtime_available,
    g_idx_block_uniform,
    pangolin_gemv,
)


pytestmark = [pytest.mark.cpu]


def test_g_idx_block_uniform_cache_tracks_in_place_mutation():
    g_idx = torch.zeros(64, dtype=torch.int32)
    assert g_idx_block_uniform(g_idx)
    g_idx[1] = 1
    assert not g_idx_block_uniform(g_idx)


@contextmanager
def _env_flag(name: str, value: str):
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def _make_planar_module(
    bits: int,
    in_features: int,
    out_features: int,
    group_size: int = 32,
):
    """Build a CPU TorchLinear with planar packing for the requested bit width."""
    torch.manual_seed(bits)
    linear = nn.Linear(in_features, out_features, bias=False)
    scales = torch.rand(out_features, in_features // group_size) * 0.01 + 0.005
    zeros = torch.randint(0, (1 << bits), (out_features, in_features // group_size)).float()
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)

    kwargs = {
        "bits": bits,
        "group_size": group_size,
        "sym": False,
        "desc_act": False,
        "in_features": in_features,
        "out_features": out_features,
        "bias": False,
        "register_buffers": False,
    }
    if bits == 3:
        kwargs["format"] = FORMAT.GPTQ_P
    module = TorchLinear(**kwargs)
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())
    return module


def _reference_forward(module: TorchLinear, x: torch.Tensor) -> torch.Tensor:
    """Reference path: dequantize dense weights, then matmul in float32."""
    weight = module.dequantize_weight()[: module.in_features, : module.out_features]
    return torch.matmul(x.to(torch.float32), weight.to(torch.float32))


@pytest.mark.parametrize("bits", [3, 5, 6, 7])
@pytest.mark.parametrize("M", PANGOLIN_SUPPORTED_M)
@pytest.mark.parametrize("disable", [None, "avx512", "avx2", "scalar"])
def test_pangolin_cpu_kernel_correctness(bits: int, M: int, disable: str):
    """Native CPU Pangolin output must match the dequant+matmul reference."""
    if not ensure_pangolin_cpu_runtime_available():
        pytest.skip("Pangolin CPU kernel not available")

    in_features = 256
    out_features = 128
    module = _make_planar_module(bits, in_features, out_features)
    x = torch.randn(M, in_features, dtype=torch.bfloat16)

    ctxs = []
    if disable in ("avx512", "avx2", "scalar"):
        ctxs.append(_env_flag("GPTQMODEL_PANGOLIN_CPU_DISABLE_AVX512", "1"))
    if disable == "scalar":
        ctxs.append(_env_flag("GPTQMODEL_PANGOLIN_CPU_DISABLE_AVX2", "1"))

    with ExitStack() as stack:
        for ctx in ctxs:
            stack.enter_context(ctx)
        out = pangolin_gemv(x, module.qweight, module.scales, module.qzeros, module.g_idx, bits)

    ref = _reference_forward(module, x)
    if not torch.allclose(out.float(), ref, rtol=0.02, atol=0.01):
        diff = (out.float() - ref).abs()
        max_abs = diff.max().item()
        max_rel = (diff / ref.abs().clamp(min=1e-6)).max().item()
        assert False, (
            f"bits={bits} M={M} disable={disable} "
            f"max abs={max_abs:.4f} max rel={max_rel:.4f} exceeds tolerance"
        )
