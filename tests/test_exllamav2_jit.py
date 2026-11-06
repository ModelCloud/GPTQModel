# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

import gptqmodel.nn_modules.qlinear.exllamav2 as exllamav2_module
import gptqmodel.utils.exllamav2 as exllamav2_utils
from gptqmodel.utils import cpp as cpp_module


def _build_module(monkeypatch) -> exllamav2_module.ExllamaV2Linear:
    # These tests stub the native ops; construction must not compile CUDA on
    # CPU/ROCm hosts before those stubs can be installed.
    monkeypatch.setattr(
        exllamav2_module.ExllamaV2Linear, "cached_validate_once", classmethod(lambda cls: (True, None))
    )
    return exllamav2_module.ExllamaV2Linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=True,
    )


def test_exllamav2_make_q_matrix_uses_jit_op(monkeypatch):
    module = _build_module(monkeypatch)
    calls = {}

    def fake_make_q_matrix(qweight, q_perm, q_invperm, q_scale, q_scale_max, q_groups, qzeros, scales, g_idx):
        calls["make_q_matrix"] = {
            "qweight_shape": tuple(qweight.shape),
            "scales_dtype": scales.dtype,
            "g_idx_is_none": g_idx is None,
        }
        return 123

    monkeypatch.setattr(exllamav2_module, "exllamav2_make_q_matrix", fake_make_q_matrix)

    weights = {
        "qweight": module.qweight,
        "qzeros": module.qzeros,
        "scales": module.scales.to(dtype=torch.float32),
        "g_idx": torch.zeros_like(module.g_idx),
    }

    handle = module.ext_make_q_matrix(weights)

    assert handle == 123
    assert calls["make_q_matrix"] == {
        "qweight_shape": tuple(module.qweight.shape),
        "scales_dtype": torch.float16,
        "g_idx_is_none": True,
    }


def test_exllamav2_forward_uses_jit_gemm(monkeypatch):
    module = _build_module(monkeypatch)
    module.q_handle = 77
    calls = {}

    def fake_gemm(x, q_handle, output, force_cuda):
        calls["gemm"] = {
            "x_shape": tuple(x.shape),
            "x_dtype": x.dtype,
            "q_handle": q_handle,
            "output_shape": tuple(output.shape),
            "force_cuda": force_cuda,
        }
        output.copy_(torch.full_like(output, 5.0))

    monkeypatch.setattr(exllamav2_module, "exllamav2_gemm_half_q_half", fake_gemm)

    x = torch.randn((2, module.in_features), dtype=torch.float32)
    out = module(x)

    assert calls["gemm"] == {
        "x_shape": (2, module.in_features),
        "x_dtype": torch.float32,
        "q_handle": 77,
        "output_shape": (2, module.out_features),
        "force_cuda": False,
    }
    assert out.shape == (2, module.out_features)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.full_like(out, 5.0))


def test_exllamav2_validate_once_surfaces_jit_error(monkeypatch):
    # Pin the host family so this exercises the non-ROCm JIT-error path even
    # when the suite runs on a HIP device.
    monkeypatch.setattr(exllamav2_module, "IS_ROCM", False)
    monkeypatch.setattr(exllamav2_module, "exllamav2_gptq_runtime_available", lambda: False)
    monkeypatch.setattr(exllamav2_module, "exllamav2_gptq_runtime_error", lambda: "missing exllamav2 jit ops")

    ok, err = exllamav2_module.ExllamaV2Linear.validate_once()

    assert ok is False
    assert isinstance(err, ImportError)
    assert "missing exllamav2 jit ops" in str(err)


def test_exllamav2_include_paths_use_wheel_headers_when_local_cuda_is_incomplete(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp_module, "IS_ROCM", False)
    root = tmp_path / "exllamav2"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (wheel_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(exllamav2_utils, "_exllamav2_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = exllamav2_utils._exllamav2_include_paths()

    assert include_paths[0] == str(root)
    assert str(wheel_cuda_include) in include_paths


def test_exllamav2_include_paths_skip_wheel_headers_when_local_cuda_has_required_headers(monkeypatch, tmp_path):
    root = tmp_path / "exllamav2"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (local_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(exllamav2_utils, "_exllamav2_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = exllamav2_utils._exllamav2_include_paths()

    assert include_paths == [str(root)]
