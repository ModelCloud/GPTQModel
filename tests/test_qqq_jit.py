# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gptqmodel.nn_modules.qlinear.qqq as qqq_module


def _build_module() -> qqq_module.QQQLinear:
    return qqq_module.QQQLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=True,
    )


def test_qqq_forward_uses_jit_kernel(monkeypatch):
    module = _build_module()
    calls = {}

    monkeypatch.setattr(qqq_module, "qqq_runtime_available", lambda: True)

    def fake_gemm(A, B, C, D, s1, s2, s3, workspace, thread_k, thread_n, sms, max_par):
        calls["gemm"] = {
            "A_shape": tuple(A.shape),
            "A_dtype": A.dtype,
            "B_shape": tuple(B.shape),
            "D_shape": tuple(D.shape),
            "s1_shape": tuple(s1.shape),
            "s2_shape": tuple(s2.shape),
            "s3_shape": tuple(s3.shape),
            "workspace_shape": tuple(workspace.shape),
            "thread_k": thread_k,
            "thread_n": thread_n,
            "sms": sms,
            "max_par": max_par,
        }
        D.copy_(torch.full_like(D, 3.0))

    monkeypatch.setattr(qqq_module, "qqq_gemm", fake_gemm)

    x = torch.randn((2, module.in_features), dtype=torch.float32)
    out = module(x)

    assert calls["gemm"] == {
        "A_shape": (2, module.in_features),
        "A_dtype": torch.int8,
        "B_shape": tuple(module.B.shape),
        "D_shape": (2, module.out_features),
        "s1_shape": (2, 1),
        "s2_shape": tuple(module.s_channel.shape),
        "s3_shape": tuple(module.s_group.shape),
        "workspace_shape": tuple(module.workspace.shape),
        "thread_k": -1,
        "thread_n": -1,
        "sms": -1,
        "max_par": module.max_par,
    }
    assert out.shape == (2, module.out_features)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.full_like(out, 3.0))


def test_qqq_forward_raises_runtime_error_when_jit_ops_missing(monkeypatch):
    module = _build_module()

    monkeypatch.setattr(qqq_module, "qqq_runtime_available", lambda: False)
    monkeypatch.setattr(qqq_module, "qqq_runtime_error", lambda: "missing qqq jit ops")

    with pytest.raises(ModuleNotFoundError, match="missing qqq jit ops"):
        module(torch.randn((1, module.in_features), dtype=torch.float16))
