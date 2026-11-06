# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

import gptqmodel.nn_modules.qlinear.exllamav2_awq as exllamav2_awq_module


def _build_module(monkeypatch) -> exllamav2_awq_module.AwqExllamaV2Linear:
    # Native calls are mocked below, so do not require a CUDA build to construct
    # the module used by these dispatch-only tests.
    monkeypatch.setattr(
        exllamav2_awq_module.AwqExllamaV2Linear, "cached_validate_once", classmethod(lambda cls: (True, None))
    )
    return exllamav2_awq_module.AwqExllamaV2Linear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=True,
    )


def test_exllamav2_awq_make_q_matrix_uses_jit_op(monkeypatch):
    module = _build_module(monkeypatch)
    calls = {}

    def fake_make_q_matrix(qweight, q_perm, q_invperm, q_scale, q_scale_max, q_groups, qzeros, scales, g_idx):
        calls["make_q_matrix_awq"] = {
            "qweight_shape": tuple(qweight.shape),
            "qzeros_shape": tuple(qzeros.shape),
            "scales_dtype": scales.dtype,
            "g_idx_is_none": g_idx is None,
        }
        return 123

    monkeypatch.setattr(exllamav2_awq_module, "exllamav2_awq_make_q_matrix", fake_make_q_matrix)

    handle = module.ext_make_q_matrix_awq(
        module.qweight,
        module.qzeros,
        module.scales.to(dtype=torch.float32),
    )

    assert handle == 123
    assert calls["make_q_matrix_awq"] == {
        "qweight_shape": tuple(module.qweight.shape),
        "qzeros_shape": tuple(module.qzeros.shape),
        "scales_dtype": torch.float16,
        "g_idx_is_none": True,
    }


def test_exllamav2_awq_forward_uses_jit_gemm(monkeypatch):
    module = _build_module(monkeypatch)
    module.q_handle = 77
    calls = {}

    def fake_gemm(x, q_handle, output, force_cuda):
        calls["gemm_half_q_half_awq"] = {
            "x_shape": tuple(x.shape),
            "x_dtype": x.dtype,
            "q_handle": q_handle,
            "output_shape": tuple(output.shape),
            "force_cuda": force_cuda,
        }
        output.copy_(torch.full_like(output, 7.0))

    monkeypatch.setattr(exllamav2_awq_module, "exllamav2_awq_gemm_half_q_half", fake_gemm)
    monkeypatch.setattr(exllamav2_awq_module, "exllamav2_awq_runtime_available", lambda: True)

    x = torch.randn((2, module.in_features), dtype=torch.float32)
    out = module(x)

    assert calls["gemm_half_q_half_awq"] == {
        "x_shape": (2, module.in_features),
        "x_dtype": torch.float32,
        "q_handle": 77,
        "output_shape": (2, module.out_features),
        "force_cuda": False,
    }
    assert out.shape == (2, module.out_features)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.full_like(out, 7.0))


def test_exllamav2_awq_validate_once_surfaces_jit_error(monkeypatch):
    # Pin the host family so this exercises the non-ROCm JIT-error path even
    # when the suite runs on a HIP device.
    monkeypatch.setattr(exllamav2_awq_module, "IS_ROCM", False)
    monkeypatch.setattr(exllamav2_awq_module, "exllamav2_awq_runtime_available", lambda: False)
    monkeypatch.setattr(exllamav2_awq_module, "exllamav2_awq_runtime_error", lambda: "missing exllamav2 awq jit ops")

    ok, err = exllamav2_awq_module.AwqExllamaV2Linear.validate_once()

    assert ok is False
    assert isinstance(err, ImportError)
    assert "missing exllamav2 awq jit ops" in str(err)
