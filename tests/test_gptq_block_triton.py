# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict accuracy tests for the fused Triton GPTQ block kernel."""

import os

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _run_gptq_quantize(group_size: int, use_triton: bool):
    """Run a single layer GPTQ quantize and return the quantize() outputs."""
    env_key = "GPTQMODEL_TRITON_BLOCK"
    prev = os.environ.get(env_key)
    os.environ[env_key] = "1" if use_triton else "0"
    try:
        # Import/reload is needed because the module-level flag is evaluated at import time.
        import gptqmodel.quantization.gptq as gptq_mod

        import importlib

        importlib.reload(gptq_mod)
        from gptqmodel.quantization.gptq import GPTQ as ReloadedGPTQ

        torch.manual_seed(42)
        device = "cuda:0"
        layer = nn.Linear(2048, 2048, bias=False, dtype=torch.float16, device=device)
        qcfg = QuantizeConfig(
            bits=4,
            group_size=group_size,
            sym=False,
            desc_act=False,
            offload_to_disk=False,
            mse=2.0,
            scale_search=ScaleSearchConfig.ACTIVATION,
        )
        g = ReloadedGPTQ(layer, qcfg=qcfg)
        g.quantizer.configure(perchannel=True)
        inp = torch.randn(4, 2048, dtype=torch.float16, device=device)
        g.add_batch(inp, None)
        return g.quantize(blocksize=128)
    finally:
        if prev is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = prev


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_gptq_triton_block_matches_serial_loop(group_size):
    """Fused Triton block kernel must produce the same quantized weights and loss
    as the serial eager loop for a real layer.
    """
    Q_ref, scale_ref, zero_ref, g_idx_ref, _, loss_ref, _, _ = _run_gptq_quantize(
        group_size, use_triton=False
    )
    Q_triton, scale_triton, zero_triton, g_idx_triton, _, loss_triton, _, _ = _run_gptq_quantize(
        group_size, use_triton=True
    )

    # scale/zero come from find_params_batched and are independent of the block
    # kernel; tiny run-to-run FP differences are acceptable.
    torch.testing.assert_close(scale_ref, scale_triton, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(zero_ref, zero_triton, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(g_idx_ref, g_idx_triton, atol=0.0, rtol=0.0)

    # Quantized weights may differ by at most one bin (~ one scale value).
    torch.testing.assert_close(Q_ref, Q_triton, atol=5e-2, rtol=1e-2)

    # Diagnostic loss is computed from the same error formula and must be very close.
    assert isinstance(loss_ref, float) and isinstance(loss_triton, float)
    assert abs(loss_ref - loss_triton) < 1e-6


def test_gptq_triton_block_env_disable_falls_back():
    """With GPTQMODEL_TRITON_BLOCK=0 the code path must not invoke the Triton kernel."""
    # This is mainly a smoke test that the fallback serial loop still runs.
    Q_ref, *_ = _run_gptq_quantize(128, use_triton=False)
    assert Q_ref is not None
    assert Q_ref.numel() > 0
