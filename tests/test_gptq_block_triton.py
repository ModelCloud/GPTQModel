# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict accuracy tests for the fused Triton GPTQ block kernel."""

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _serial_block(W1, Q1, Err1, Hinv1, scale, zero, maxq, group_size):
    """Exact eager reference for one asymmetric grouped-GPTQ block."""

    for i in range(W1.shape[1]):
        w = W1[:, i]
        q_scale = scale[:, i // group_size]
        q_zero = zero[:, i // group_size]
        q = q_scale * (torch.clamp(torch.round(w / q_scale) + q_zero, 0, maxq) - q_zero)
        err = (w - q) / Hinv1[i, i]
        Q1[:, i] = q
        Err1[:, i] = err
        W1[:, i:] = torch.addr(W1[:, i:], err, Hinv1[i, i:], alpha=-1.0)


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


@pytest.mark.parametrize("group_size", [64, 128])
def test_gptq_triton_block_records_prepared_operands_on_launch_stream(group_size):
    """Local contiguous copies must remain allocator-owned until Triton finishes.

    Activation scale search can hand the wrapper strided views. The wrapper's
    contiguous copies are local variables, so failing to record their external
    Triton use permits a free-threaded worker to recycle them after return.
    """

    import gptqmodel.quantization._gptq_block_triton as block_module

    gptq_block_triton = block_module.gptq_block_triton

    if gptq_block_triton is None:
        pytest.skip("Triton GPTQ block kernel is unavailable")

    torch.manual_seed(7)
    device = torch.device("cuda:0")
    rows, count = 96, 128
    groups = count // group_size
    maxq = 15

    weights = torch.randn(rows, count, device=device, dtype=torch.float32)
    hessian_factor = torch.randn(count, count, device=device, dtype=torch.float32)
    hessian_inverse = hessian_factor @ hessian_factor.T + 0.5 * torch.eye(count, device=device)
    hessian_inverse = torch.linalg.cholesky(torch.linalg.inv(hessian_inverse), upper=True)

    # Force wrapper-local contiguous copies for all three read-only operands.
    hessian_storage = torch.empty(count, count * 2, device=device, dtype=torch.float32)
    hessian_storage[:, ::2] = hessian_inverse
    hessian_inverse = hessian_storage[:, ::2]
    scale_storage = torch.rand(rows, groups * 2, device=device, dtype=torch.float32) + 0.05
    zero_storage = torch.full((rows, groups * 2), 8.0, device=device, dtype=torch.float32)
    scale = scale_storage[:, ::2]
    zero = zero_storage[:, ::2]
    assert not hessian_inverse.is_contiguous()
    assert not scale.is_contiguous()
    assert not zero.is_contiguous()

    ref_weights = weights.clone()
    ref_quantized = torch.empty_like(ref_weights)
    ref_errors = torch.empty_like(ref_weights)
    _serial_block(
        ref_weights,
        ref_quantized,
        ref_errors,
        hessian_inverse.contiguous(),
        scale.contiguous(),
        zero.contiguous(),
        maxq,
        group_size,
    )

    actual_weights = weights.clone()
    actual_quantized = torch.empty_like(actual_weights)
    actual_errors = torch.empty_like(actual_weights)
    recorded_operands = []
    original_record_stream = torch.Tensor.record_stream

    class TrackingLock:
        def __init__(self):
            self.enter_count = 0

        def __enter__(self):
            self.enter_count += 1

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    launch_lock = TrackingLock()

    def capture_record_stream(tensor, stream):
        recorded_operands.append((tensor, stream))
        return original_record_stream(tensor, stream)

    with (
        patch.object(block_module, "_TRITON_LAUNCH_LOCK", launch_lock),
        patch.object(torch.Tensor, "record_stream", capture_record_stream),
    ):
        gptq_block_triton(
            actual_weights,
            actual_quantized,
            actual_errors,
            hessian_inverse,
            scale,
            zero,
            maxq,
            group_size,
        )
        torch.cuda.synchronize(device)

    assert launch_lock.enter_count == 1
    assert len(recorded_operands) == 6
    assert all(tensor.is_contiguous() for tensor, _ in recorded_operands)
    assert all(stream.device == device for _, stream in recorded_operands)
    torch.testing.assert_close(actual_quantized, ref_quantized, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(actual_errors, ref_errors, atol=2e-4, rtol=1e-5)
