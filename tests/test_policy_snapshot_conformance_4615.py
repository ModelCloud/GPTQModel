# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Conformance test for the requested-vs-resolved policy contract (#4615, row 4).

The proposal states the contract as:

    Mutable user configuration is input to resolution, never the execution
    state. Resolve a deep immutable, value-keyed snapshot at the API boundary
    and pass that same snapshot to all consumers.

The individual mechanisms already exist: #4497 keeps the derived policy in
``_resolved_kernel`` instead of the caller's ``kernel`` field, #4553
content-keys the dynamic-rule cache, #4585 makes forward dispatch read the
module snapshot, and #4606 inherits an explicit enum load choice. Testing each
of those helpers separately cannot prove the composition, so this file drives
one flow through all of them:

1. the caller mutates the request *after* resolution and execution must keep
   the resolved values;
2. the loader/quantize consumer and the forward-time dispatch consumer must
   observe one and the same snapshot;
3. snapshots and the resolution cache must be keyed by values, so a warmed
   cache cannot serve a stale policy.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.backend import BACKEND, RuntimeConfig
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import _dynamic_cache_key
from gptqmodel.utils.backend_config import resolve_quantize_backend
from gptqmodel.utils.kernel import (
    KernelConfig,
    configure_kernel,
    reset_kernel_config,
    resolve_kernel_config,
)


@pytest.fixture(autouse=True)
def _isolated_process_policy():
    reset_kernel_config()
    yield
    reset_kernel_config()


def _torch_layer(kernel: KernelConfig) -> TorchLinear:
    """Build a CPU TorchLinear whose dispatch state comes from ``kernel``."""

    layer = TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
        backend=BACKEND.GPTQ_TORCH,
        kernel=kernel,
        register_buffers=True,
        dtype=torch.float16,
    ).eval()
    layer.qweight.fill_(0x11111111)
    layer.qzeros.zero_()
    layer.scales.fill_(1)
    layer.g_idx.zero_()
    return layer


def test_requested_config_is_input_never_execution_state():
    """One flow: resolve, build, mutate the request, then keep executing."""

    requested = KernelConfig(
        torch_streaming=True,
        torch_stream_tile=16,
        torch_triton_dequant=False,
        torch_rocm_gemv=False,
        max_packers=2,
    )
    qcfg = QuantizeConfig(bits=4, kernel=requested)

    # --- The API boundary resolves once and owns the result.
    backend, snapshot = resolve_quantize_backend(BACKEND.GPTQ_TORCH, qcfg)
    assert backend is BACKEND.GPTQ_TORCH
    assert snapshot is qcfg._resolved_kernel

    # --- Consumer 1: the loader/quantize path (make_quant, loopers).
    loader_kernel = qcfg.quant_linear_init_kwargs()["kernel"]
    assert loader_kernel is snapshot
    assert resolve_kernel_config(qcfg) is snapshot

    # --- Consumer 2: a module built from the loader policy, and the helper
    # every forward-dispatch site calls to read the module's own snapshot.
    layer = _torch_layer(loader_kernel)
    assert layer.kernel is not snapshot  # the module owns its own copy
    assert layer.kernel == snapshot  # ... with the snapshot's values
    assert resolve_kernel_config(layer) is layer.kernel
    assert layer._streaming_enabled is True
    assert layer._stream_tile_cols == 16
    assert layer._rocm_gemv_enabled is False

    x = torch.arange(64, dtype=torch.float16).reshape(2, 32) / 64
    with torch.inference_mode():
        before = layer(x)

    # --- The caller mutates its request AFTER resolution, three ways: in
    # place on the legacy object, by replacing the config field, and by a
    # later process-wide load.
    requested.torch_stream_tile = 4096
    requested.torch_streaming = False
    requested.torch_rocm_gemv = True
    qcfg.kernel = KernelConfig(torch_stream_tile=1, torch_streaming=False)
    qcfg.runtime = RuntimeConfig(max_packers=99)
    configure_kernel(KernelConfig(torch_stream_tile=7, torch_streaming=False))

    # (1) Execution keeps the resolved snapshot, not the mutated request.
    assert snapshot.torch_stream_tile == 16
    assert snapshot.torch_streaming is True
    assert snapshot.torch_rocm_gemv is False
    assert resolve_kernel_config(qcfg) is snapshot
    assert qcfg.quant_linear_init_kwargs()["kernel"] is snapshot

    # (2) Both consumers still observe that same snapshot.
    assert layer.kernel == snapshot
    assert resolve_kernel_config(layer) is layer.kernel
    assert layer.kernel.torch_stream_tile == 16
    assert layer.kernel.torch_rocm_gemv is False
    assert layer._streaming_enabled is True
    assert layer._stream_tile_cols == 16
    assert layer._rocm_gemv_enabled is False
    with torch.inference_mode():
        after = layer(x)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_resolved_snapshot_is_value_keyed():
    """Snapshot identity is its values; the resolution cache is content-keyed."""

    # (3a) Identical requested values resolve to equal snapshots; a different
    # value resolves to a different snapshot.
    left = KernelConfig(torch_stream_tile=16, max_packers=2)
    right = KernelConfig(torch_stream_tile=16, max_packers=2)
    other = KernelConfig(torch_stream_tile=32, max_packers=2)

    qcfg_left = QuantizeConfig(bits=4, kernel=left)
    qcfg_right = QuantizeConfig(bits=4, kernel=right)
    qcfg_other = QuantizeConfig(bits=4, kernel=other)

    _, snap_left = resolve_quantize_backend(BACKEND.GPTQ_TORCH, qcfg_left)
    _, snap_right = resolve_quantize_backend(BACKEND.GPTQ_TORCH, qcfg_right)
    _, snap_other = resolve_quantize_backend(BACKEND.GPTQ_TORCH, qcfg_other)

    assert snap_left == snap_right
    assert snap_left.to_dict() == snap_right.to_dict()
    assert snap_left != snap_other

    # (3b) A fresh resolution sees a changed request while the snapshot that
    # was already issued keeps its old value.
    left.torch_stream_tile = 48
    _, fresh = resolve_quantize_backend(BACKEND.GPTQ_TORCH, qcfg_left)
    assert fresh.torch_stream_tile == 48
    assert fresh != snap_left
    assert snap_left.torch_stream_tile == 16

    # (3c) The dynamic-rule cache is keyed by content, not by object identity:
    # equal trees share a key and an in-place typed-leaf edit changes it, so a
    # warmed cache cannot serve a stale policy.
    a = {"^layer$": {"mse": {"norm": 2.0}}}
    b = {"^layer$": {"mse": {"norm": 2.0}}}
    assert _dynamic_cache_key(a) == _dynamic_cache_key(b)
    b["^layer$"]["mse"]["norm"] = 4.0
    assert _dynamic_cache_key(a) != _dynamic_cache_key(b)


def test_runtime_request_is_resolved_once_and_survives_replacement():
    """Control: the ``runtime=`` path already resolves to an independent copy."""

    qcfg = QuantizeConfig(
        bits=4,
        runtime=RuntimeConfig(max_packers=2, gptq_fused_block_update=False),
    )
    _, snapshot = resolve_quantize_backend(BACKEND.GPTQ_TORCH, qcfg)
    assert snapshot.max_packers == 2
    assert snapshot.gptq_fused_block_update is False
    assert resolve_kernel_config(qcfg) is snapshot
    assert qcfg.quant_linear_init_kwargs()["kernel"] is snapshot

    qcfg.runtime = RuntimeConfig(max_packers=99)
    assert resolve_kernel_config(qcfg) is snapshot
    assert resolve_kernel_config(qcfg).max_packers == 2
