# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gptqmodel import extension as extension_api
import gptqmodel.nn_modules.qlinear.marlin as marlin_qlinear_module
import gptqmodel.utils.marlin as marlin_utils
from gptqmodel.utils import cpp as cpp_module
from gptqmodel.utils.marlin_scalar_type import scalar_types


class _FakeLoader:
    def __init__(self, *, should_load: bool = True, last_error: str = ""):
        self.should_load = should_load
        self._last_error = last_error
        self.ops: dict[str, object] = {}
        self.load_calls = 0
        self.op_calls: list[str] = []

    def load(self) -> bool:
        self.load_calls += 1
        return self.should_load

    def op(self, op_name: str):
        self.op_calls.append(op_name)
        return self.ops[op_name]

    def last_error_message(self) -> str:
        return self._last_error

    def clear_cache(self) -> None:
        return None


class _FakeExtensionApi:
    def __init__(self, *, available: bool = False, error_text: str = ""):
        self.available = available
        self.error_text = error_text
        self.is_available_calls: list[str] = []
        self.error_calls: list[str] = []

    def is_available(self, extension_name: str) -> bool:
        self.is_available_calls.append(extension_name)
        return self.available

    def error(self, extension_name: str) -> str:
        self.error_calls.append(extension_name)
        return self.error_text


def _jit_scratch_root(tmp_path: Path, suffix: str) -> Path:
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else tmp_path
    root = base / "gptqmodel-jit-tests" / suffix
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_gptq_marlin_gemm_dispatches_fp16_to_torch_ops(monkeypatch):
    fp16_loader = _FakeLoader()
    bf16_loader = _FakeLoader()
    captured = {}

    def fake_gemm(*args):
        captured["dtype"] = args[0].dtype
        captured["shape"] = (args[11], args[12])
        return torch.full((args[11], args[12]), 3.0, dtype=args[0].dtype)

    fp16_loader.ops["gptq_marlin_gemm_fp16"] = fake_gemm

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    out = marlin_utils.gptq_marlin_gemm(
        a=torch.ones((2, 128), dtype=torch.float16),
        c=None,
        b_q_weight=torch.zeros((32, 64), dtype=torch.int32),
        b_bias=None,
        b_scales=torch.ones((1, 64), dtype=torch.float16),
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=torch.zeros(1, dtype=torch.int32),
        b_q_type=scalar_types.uint4b8,
        size_m=2,
        size_n=64,
        size_k=128,
    )

    assert fp16_loader.op_calls == ["gptq_marlin_gemm_fp16"]
    assert bf16_loader.op_calls == []
    assert captured == {"dtype": torch.float16, "shape": (2, 64)}
    assert out.shape == (2, 64)
    assert out.dtype == torch.float16


def test_gptq_marlin_gemm_dispatches_bf16_to_torch_ops(monkeypatch):
    fp16_loader = _FakeLoader()
    bf16_loader = _FakeLoader()
    captured = {}

    def fake_gemm(*args):
        captured["dtype"] = args[0].dtype
        return torch.full((args[11], args[12]), 5.0, dtype=args[0].dtype)

    bf16_loader.ops["gptq_marlin_gemm_bf16"] = fake_gemm

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    out = marlin_utils.gptq_marlin_gemm(
        a=torch.ones((1, 64), dtype=torch.bfloat16),
        c=None,
        b_q_weight=torch.zeros((16, 64), dtype=torch.int32),
        b_bias=None,
        b_scales=torch.ones((1, 64), dtype=torch.bfloat16),
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=torch.zeros(1, dtype=torch.int32),
        b_q_type=scalar_types.uint8b128,
        size_m=1,
        size_n=64,
        size_k=64,
    )

    assert bf16_loader.op_calls == ["gptq_marlin_gemm_bf16"]
    assert fp16_loader.op_calls == []
    assert captured == {"dtype": torch.bfloat16}
    assert out.shape == (1, 64)
    assert out.dtype == torch.bfloat16


def test_gptq_marlin_repack_prefers_requested_dtype_extension(monkeypatch):
    fp16_loader = _FakeLoader()
    bf16_loader = _FakeLoader()
    captured = {}

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
        captured["dtype"] = torch.bfloat16
        captured["shape"] = tuple(b_q_weight.shape)
        return b_q_weight + 1

    bf16_loader.ops["gptq_marlin_repack"] = fake_repack

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    out = marlin_utils.gptq_marlin_repack(
        torch.zeros((32, 64), dtype=torch.int32),
        torch.arange(32, dtype=torch.int32),
        128,
        64,
        4,
        dtype=torch.bfloat16,
    )

    assert bf16_loader.op_calls == ["gptq_marlin_repack"]
    assert fp16_loader.op_calls == []
    assert captured == {"dtype": torch.bfloat16, "shape": (32, 64)}
    assert torch.equal(out, torch.ones((32, 64), dtype=torch.int32))


def test_awq_marlin_repack_raises_when_requested_jit_extension_is_unavailable(monkeypatch):
    fp16_loader = _FakeLoader(should_load=False, last_error="fp16 unavailable")
    bf16_loader = _FakeLoader(should_load=False, last_error="bf16 unavailable")

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    with pytest.raises(RuntimeError, match="bf16 unavailable"):
        marlin_utils.awq_marlin_repack(
            torch.zeros((64, 16), dtype=torch.int32),
            64,
            128,
            4,
            dtype=torch.bfloat16,
        )

    assert fp16_loader.op_calls == []
    assert bf16_loader.op_calls == []


def test_marlin_quant_linear_post_init_uses_compute_dtype_for_repack(monkeypatch):
    captured = {}

    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_error", lambda dtype: "")
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_make_workspace_new",
        lambda device: torch.zeros(1, dtype=torch.int32, device=device),
    )
    monkeypatch.setattr(
        marlin_qlinear_module,
        "gptq_marlin_repack",
        lambda b_q_weight, perm, size_k, size_n, num_bits, dtype=None: (
            captured.update({"dtype": dtype, "shape": tuple(b_q_weight.shape)}) or b_q_weight
        ),
    )
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_permute_scales",
        lambda scales, size_k, size_n, group_size: scales,
    )
    monkeypatch.setattr(marlin_qlinear_module, "marlin_permute_bias", lambda bias: bias)

    module = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        bias=False,
        dtype=torch.bfloat16,
    )
    module.post_init()

    assert captured == {"dtype": torch.bfloat16, "shape": tuple(module.qweight.shape)}


def test_marlin_runtime_error_appends_cuda_extra_install_hint_for_missing_headers(monkeypatch):
    fake_extension_api = _FakeExtensionApi(
        error_text=(
            "Marlin fp16: failed to build torch.ops JIT extension: "
            "fatal error: cusparse.h: No such file or directory"
        ),
    )

    monkeypatch.setattr(marlin_utils, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_utils, "_extension_api", lambda: fake_extension_api)
    monkeypatch.setattr(marlin_utils, "detected_cuda_wheel_include_paths", lambda: [])
    monkeypatch.setattr(marlin_utils, "which", lambda name: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising=False)

    error_text = marlin_utils.marlin_runtime_error(torch.float16)

    assert fake_extension_api.is_available_calls == ["marlin_fp16"]
    assert fake_extension_api.error_calls == ["marlin_fp16"]
    assert "cusparse.h" in error_text
    assert 'pip install "gptqmodel[marlin-cuda]"' in error_text
    assert "A local `nvcc` on PATH is still required for Marlin JIT." in error_text


def test_marlin_runtime_error_skips_install_hint_when_cuda_wheel_headers_are_detected(monkeypatch):
    fake_extension_api = _FakeExtensionApi(
        error_text=(
            "Marlin bf16: failed to build torch.ops JIT extension: "
            "fatal error: cublas_v2.h: No such file or directory"
        ),
    )

    monkeypatch.setattr(marlin_utils, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_utils, "_extension_api", lambda: fake_extension_api)
    monkeypatch.setattr(marlin_utils, "detected_cuda_wheel_include_paths", lambda: ["/tmp/nvidia/cu13/include"])
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising=False)

    error_text = marlin_utils.marlin_runtime_error(torch.bfloat16)

    assert fake_extension_api.is_available_calls == ["marlin_bf16"]
    assert fake_extension_api.error_calls == ["marlin_bf16"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_marlin_cuda_smoke_build_and_forward(monkeypatch, tmp_path):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Marlin requires compute capability >= 8.0")

    scratch_root = _jit_scratch_root(tmp_path, "marlin")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("GPTQMODEL_MARLIN_FP16_BUILD_ROOT", str(scratch_root / "marlin_fp16"))
    monkeypatch.setenv("GPTQMODEL_MARLIN_BF16_BUILD_ROOT", str(scratch_root / "marlin_bf16"))
    monkeypatch.setenv("GPTQMODEL_MARLIN_FORCE_REBUILD", "1")

    assert extension_api.load(name="marlin", use_cache=False) == {
        "marlin_fp16": True,
        "marlin_bf16": True,
    }

    device = torch.device("cuda:0")
    for dtype in (torch.float16, torch.bfloat16):
        module = marlin_qlinear_module.MarlinLinear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=128,
            out_features=64,
            bias=False,
            dtype=dtype,
        ).to(device)
        with torch.no_grad():
            module.qweight.copy_(torch.randint(0, 16, module.qweight.shape, device=device, dtype=torch.int32))
            module.g_idx.copy_(torch.arange(module.in_features, device=device, dtype=torch.int32))
            module.scales.copy_(torch.ones_like(module.scales, device=device))
            module.qzeros.copy_(torch.zeros_like(module.qzeros, device=device))
        module.post_init()

        out = module(torch.randn(4, 128, device=device, dtype=dtype))
        torch.cuda.synchronize(device)

        assert out.shape == (4, 64)
        assert out.dtype == dtype


def test_marlin_include_paths_use_wheel_headers_when_local_cuda_is_incomplete(monkeypatch, tmp_path):
    root = tmp_path / "marlin"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    for header_name in marlin_utils._MARLIN_REQUIRED_CUDA_HEADERS:
        (wheel_cuda_include / header_name).write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(marlin_utils, "_marlin_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = marlin_utils._marlin_include_paths()

    assert include_paths[0] == str(root)
    assert str(wheel_cuda_include) in include_paths


def test_marlin_include_paths_skip_wheel_headers_when_local_cuda_has_required_headers(monkeypatch, tmp_path):
    root = tmp_path / "marlin"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    for header_name in marlin_utils._MARLIN_REQUIRED_CUDA_HEADERS:
        (local_cuda_include / header_name).write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(marlin_utils, "_marlin_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = marlin_utils._marlin_include_paths()

    assert include_paths == [str(root)]
