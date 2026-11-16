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
