# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QQQ workspace sizing (#4215).

``qqq_gemm.cu`` uses ``workspace`` as the zeroed ``locks`` array indexed by
output tile, so it needs ``ceil(n / 64) * max_par`` int32 slots. Sizing it from
``n // 128 * max_par`` zero-sized it for ``n = 64`` (illegal memory access) and
undersized it whenever ``n % 128 != 0``.
"""

import pytest
import torch

import gptqmodel.nn_modules.qlinear.qqq as qqq_mod
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, _workspace_numel


IN_FEATURES = 512


def _build(out_features: int, monkeypatch, *, register_buffers: bool = True) -> QQQLinear:
    # The Marlin-derived JIT extension is not built in this environment, and the
    # sizing is pure Python: skip the runtime/device probes so the module can be
    # constructed on CPU.
    monkeypatch.setattr(qqq_mod, "qqq_runtime_available", lambda: True)
    monkeypatch.setattr(
        QQQLinear, "validate_device", classmethod(lambda cls, device: None)
    )
    # `validate()` consults the once-gate before `_validate`, and on ROCm that
    # gate returns QQQ_ROCM_UNSUPPORTED before either stub above is reached.
    # The workspace sizing under test is pure Python and backend-independent, so
    # stub the gate rather than skipping this file on ROCm.
    monkeypatch.setattr(
        QQQLinear, "validate_once", classmethod(lambda cls: (True, None))
    )
    return QQQLinear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=IN_FEATURES,
        out_features=out_features,
        bias=False,
        register_buffers=register_buffers,
    )


@pytest.mark.parametrize(
    ("out_features", "expected"),
    [
        (64, 16),    # one 64-column tile: was zero-sized before the fix
        (128, 32),
        (192, 48),   # 192 % 128 != 0: was undersized (16 slots)
        (256, 64),
        (320, 80),   # 320 % 128 != 0: was undersized (32 slots)
    ],
)
def test_workspace_covers_kernel_lock_slots(out_features, expected, monkeypatch):
    module = _build(out_features, monkeypatch)
    assert module.workspace.dtype == torch.int32
    assert module.workspace.numel() == expected
    assert module.workspace.numel() == _workspace_numel(out_features, module.max_par)
    # The kernel indexes one lock per 64-column tile per parallel problem.
    assert module.workspace.numel() >= (out_features + 63) // 64
    assert not module.workspace.any(), "locks must start zeroed"


@pytest.mark.parametrize("out_features", [64, 192])
def test_pack_allocates_the_same_workspace(out_features, monkeypatch):
    module = _build(out_features, monkeypatch, register_buffers=False)
    assert not hasattr(module, "workspace")

    linear = torch.nn.Linear(IN_FEATURES, out_features, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight) * 0.02)
    scales = (linear.weight.abs().amax(dim=1, keepdim=True) / 7.0).to(torch.float16)
    module.pack(linear, scales)

    assert module.workspace.numel() == _workspace_numel(out_features, module.max_par)
    assert module.workspace.numel() >= (out_features + 63) // 64


def test_forward_allocates_a_fresh_lock_array_per_call(monkeypatch):
    """Two streams on the same module must not share split-K counters (#4665).

    ``qqq_gemm.cu`` spins on and resets the lock array, so a per-module buffer
    let concurrent calls serving the same layers deadlock the device. Each
    forward must pass its own zeroed array (EXL3's per-call locks precedent).
    """

    module = _build(192, monkeypatch)
    seen = []

    def fake_mul(A, B, C, D, s1, s2, s3, workspace, **kwargs):
        seen.append(workspace)
        D.zero_()

    monkeypatch.setattr(qqq_mod, "mul", fake_mul)
    x = torch.zeros((2, IN_FEATURES), dtype=torch.float16)

    module(x)
    module(x)

    assert len(seen) == 2
    assert seen[0] is not seen[1]
    assert seen[0] is not module.workspace
    assert seen[1] is not module.workspace
    for workspace in seen:
        assert workspace.dtype == torch.int32
        assert workspace.numel() == _workspace_numel(module.out_features, module.max_par)
        assert not workspace.any()
