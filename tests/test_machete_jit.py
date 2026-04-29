# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest
import torch

import gptqmodel.nn_modules.qlinear.machete as machete_linear_module
import gptqmodel.utils.machete as machete_utils
from gptqmodel import extension as extension_api
from gptqmodel.utils.marlin_scalar_type import scalar_types


class _FakeExtensionApi:
    def __init__(self):
        self.available = True
        self.error_message = ""
        self.load_requests: list[str] = []
        self.op_requests: list[tuple[str, str]] = []
        self.prepack_result = object()
        self.schedule_result = ["sch_a", "sch_b"]
        self.mm_result = torch.ones((1, 1))

    def is_available(self, name: str) -> bool:
        assert name == "machete"
        return self.available

    def error(self, name: str) -> str:
        assert name == "machete"
        return self.error_message

    def load(self, *, name: str) -> dict[str, bool]:
        self.load_requests.append(name)
        return {"machete": True}

    def op(self, name: str, op_name: str):
        assert name == "machete"
        self.op_requests.append((name, op_name))
        if op_name == "machete_prepack_B":
            return lambda *args: self.prepack_result
        if op_name == "machete_supported_schedules":
            return lambda *args: self.schedule_result
        if op_name == "machete_mm":
            return lambda *args: self.mm_result
        raise AssertionError(f"unexpected op {op_name}")


def _write_fake_cutlass_checkout(destination: Path, *, version: str) -> None:
    major, minor, patch = version.split(".")
    (destination / "include" / "cutlass").mkdir(parents=True, exist_ok=True)
    (destination / "tools" / "library" / "include").mkdir(parents=True, exist_ok=True)
    (destination / "tools" / "util" / "include").mkdir(parents=True, exist_ok=True)
    (destination / "python").mkdir(parents=True, exist_ok=True)
    (destination / "include" / "cutlass" / "cutlass.h").write_text("// cutlass\n", encoding="utf-8")
    (destination / "include" / "cutlass" / "version.h").write_text(
        (
            "#pragma once\n"
            f"#define CUTLASS_MAJOR {major}\n"
            f"#define CUTLASS_MINOR {minor}\n"
            f"#define CUTLASS_PATCH {patch}\n"
        ),
        encoding="utf-8",
    )
    (destination / "python" / "cutlass_library.py").write_text("# cutlass python\n", encoding="utf-8")


def _write_fake_cutlass_archive(destination: Path) -> None:
    staging_root = destination.parent / f"cutlass-{machete_utils._CUTLASS_VERSION}"
    _write_fake_cutlass_checkout(staging_root, version=machete_utils._CUTLASS_VERSION)

    with tarfile.open(destination, "w:gz") as archive:
        archive.add(staging_root, arcname=f"cutlass-{machete_utils._CUTLASS_VERSION}")

    shutil.rmtree(staging_root, ignore_errors=True)


def test_machete_runtime_routes_through_extension_api(monkeypatch):
    fake_api = _FakeExtensionApi()
    monkeypatch.setattr(machete_utils, "_extension_api", lambda: fake_api)
    monkeypatch.setattr(machete_utils, "_machete_static_runtime_error", lambda: "")

    prepacked = machete_utils.machete_prepack_B(
        torch.ones((1, 1), dtype=torch.int32),
        torch.float16,
        scalar_types.uint4b8,
        torch.float16,
    )
    schedules = machete_utils.machete_supported_schedules(torch.float16, scalar_types.uint4b8)
    output = machete_utils.machete_mm(
        a=torch.ones((1, 1), dtype=torch.float16),
        b_q=torch.ones((1, 1), dtype=torch.int32),
        b_type=scalar_types.uint4b8,
    )

    assert prepacked is fake_api.prepack_result
    assert schedules == ["sch_a", "sch_b"]
    assert output is fake_api.mm_result
    assert machete_utils.prewarm_machete_extension() is True
    assert fake_api.load_requests == ["machete"]
    assert fake_api.op_requests == [
        ("machete", "machete_prepack_B"),
        ("machete", "machete_supported_schedules"),
        ("machete", "machete_mm"),
    ]


def test_machete_static_runtime_error_requires_hopper_sm90(monkeypatch):
    class _Props:
        name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        shared_memory_per_block = 49152
        shared_memory_per_block_optin = 101376

    monkeypatch.setattr(machete_utils, "IS_ROCM", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (12, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_args, **_kwargs: _Props())

    error = machete_utils._machete_static_runtime_error()

    assert "Hopper-class SM90 GPUs only" in error
    assert "12.0" in error


def test_machete_static_runtime_error_checks_optin_shared_memory(monkeypatch):
    class _Props:
        name = "NVIDIA H100 80GB HBM3"
        shared_memory_per_block = 49152
        shared_memory_per_block_optin = 98304

    monkeypatch.setattr(machete_utils, "IS_ROCM", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_args, **_kwargs: _Props())

    error = machete_utils._machete_static_runtime_error()

    assert str(machete_utils._MACHETE_MIN_SHARED_MEMORY_PER_BLOCK_OPTIN) in error
    assert "98304" in error


def test_machete_registers_checkpoint_compatible_qzeros_shape_for_symmetric_gptq():
    module = machete_linear_module.MacheteLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=8192,
        out_features=3072,
        bias=False,
        dtype=torch.float16,
    )

    assert module.qzeros.shape == (64, 384)
    assert module.qzeros.dtype == torch.int32


def test_machete_load_state_dict_accepts_checkpoint_qzeros_shape():
    module = machete_linear_module.MacheteLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=8192,
        out_features=3072,
        bias=False,
        dtype=torch.float16,
    )
    state_dict = module.state_dict()
    state_dict["qzeros"] = torch.zeros((64, 384), dtype=torch.int32)

    module.load_state_dict(state_dict)

    assert module.qzeros.shape == (64, 384)


def test_machete_post_init_discards_loaded_qzeros_for_symmetric_gptq(monkeypatch):
    module = machete_linear_module.MacheteLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        bias=False,
        dtype=torch.float16,
    )
    with torch.no_grad():
        module.qweight.copy_(torch.randint(0, 16, module.qweight.shape, dtype=torch.int32))
        module.g_idx.copy_(torch.arange(module.in_features, dtype=torch.int32))
        module.scales.copy_(torch.ones_like(module.scales))
        module.qzeros.copy_(torch.randint(0, 16, module.qzeros.shape, dtype=torch.int32))

    monkeypatch.setattr(
        machete_linear_module,
        "machete_prepack_B",
        lambda weight, **_kwargs: weight.contiguous(),
    )

    module.post_init()

    assert module.qzeros.numel() == 0
    assert module.qzeros.dtype == torch.int32
    assert module.has_zero_points is False


def test_machete_validate_accepts_asymmetric_gptq(monkeypatch):
    monkeypatch.setattr(
        machete_linear_module.MacheteLinear,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    monkeypatch.setattr(
        machete_linear_module,
        "_validate_machete_device_support",
        lambda: True,
    )

    ok, err = machete_linear_module.MacheteLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        in_features=128,
        out_features=128,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert ok is True
    assert err is None


def test_machete_post_init_transforms_loaded_qzeros_for_asymmetric_gptq(monkeypatch):
    module = machete_linear_module.MacheteLinear(
        bits=4,
        group_size=64,
        desc_act=False,
        sym=False,
        in_features=128,
        out_features=128,
        bias=False,
        dtype=torch.float16,
    )
    with torch.no_grad():
        module.qweight.copy_(torch.randint(0, 16, module.qweight.shape, dtype=torch.int32))
        module.g_idx.copy_(torch.tensor([0] * 64 + [1] * 64, dtype=torch.int32))
        module.scales.copy_(torch.rand_like(module.scales) + 0.25)
        module.qzeros.copy_(torch.randint(0, 16, module.qzeros.shape, dtype=torch.int32))

    expected_qzeros = (
        -1.0
        * module.scales.detach().clone()
        * machete_utils.unpack_quantized_values_into_int32(
            module.qzeros.detach().clone(),
            module.weight_type,
            packed_dim=1,
        ).to(module.scales.dtype)
    ).contiguous()

    monkeypatch.setattr(
        machete_linear_module,
        "machete_prepack_B",
        lambda weight, **_kwargs: weight.contiguous(),
    )

    module.post_init()

    torch.testing.assert_close(module.qzeros, expected_qzeros)
    assert module.qzeros.dtype == torch.float16
    assert module.has_zero_points is True


def test_machete_forward_rejects_missing_qzeros_for_asymmetric_gptq():
    module = machete_linear_module.MacheteLinear(
        bits=4,
        group_size=64,
        desc_act=False,
        sym=False,
        in_features=128,
        out_features=128,
        bias=False,
        dtype=torch.float16,
    )
    module.has_zero_points = True
    module.qzeros = torch.nn.Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False)

    with pytest.raises(AssertionError, match="requires non-empty qzeros"):
        module(torch.randn(2, 128, dtype=torch.float16))


def test_machete_hopper_arch_cuda_cflags_add_sm90a_when_torch_only_targets_sm90(monkeypatch):
    class _Props:
        name = "NVIDIA H200"
        shared_memory_per_block = 49152
        shared_memory_per_block_optin = 232448

    monkeypatch.setattr(machete_utils, "IS_ROCM", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_args, **_kwargs: _Props())
    monkeypatch.setattr(machete_utils, "resolved_cuda_arch_flags", lambda: ["-gencode=arch=compute_90,code=sm_90"])

    flags = machete_utils._machete_hopper_arch_cuda_cflags()

    assert flags == list(machete_utils._MACHETE_SM90A_ARCH_FLAGS)


def test_machete_hopper_arch_cuda_cflags_skip_duplicate_sm90a(monkeypatch):
    class _Props:
        name = "NVIDIA H200"
        shared_memory_per_block = 49152
        shared_memory_per_block_optin = 232448

    monkeypatch.setattr(machete_utils, "IS_ROCM", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_args, **_kwargs: _Props())
    monkeypatch.setattr(
        machete_utils,
        "resolved_cuda_arch_flags",
        lambda: ["-gencode=arch=compute_90a,code=sm_90a"],
    )

    assert machete_utils._machete_hopper_arch_cuda_cflags() == []


def test_machete_extra_cuda_cflags_keep_only_required_torch_undefines(monkeypatch):
    monkeypatch.setattr(machete_utils, "is_nvcc_compatible", lambda: False)
    monkeypatch.setattr(machete_utils, "_machete_hopper_arch_cuda_cflags", lambda: [])

    flags = machete_utils._machete_extra_cuda_cflags()

    assert flags[:3] == list(machete_utils._MACHETE_REQUIRED_TORCH_NVCC_UNDEFINES)
    assert "--threads" in flags
    assert flags[flags.index("--threads") + 1] == machete_utils._MACHETE_JIT_NVCC_THREADS
    assert "-U__CUDA_NO_BFLOAT16_OPERATORS__" not in flags
    assert "-U__CUDA_NO_BFLOAT162_OPERATORS__" not in flags
    assert "-U__CUDA_NO_BFLOAT162_CONVERSIONS__" not in flags
    assert "-U__CUDA_NO_HALF2_OPERATORS__" not in flags


def test_machete_extra_cuda_cflags_enable_static_global_template_stub_when_nvcc_is_compatible(monkeypatch):
    monkeypatch.setattr(machete_utils, "is_nvcc_compatible", lambda: True)
    monkeypatch.setattr(machete_utils, "_machete_hopper_arch_cuda_cflags", lambda: [])

    flags = machete_utils._machete_extra_cuda_cflags()

    assert flags[0] == "-static-global-template-stub=false"
    assert flags[1:4] == list(machete_utils._MACHETE_REQUIRED_TORCH_NVCC_UNDEFINES)


def test_machete_extra_cuda_cflags_skip_static_global_template_stub_when_nvcc_is_incompatible(monkeypatch):
    monkeypatch.setattr(machete_utils, "is_nvcc_compatible", lambda: False)
    monkeypatch.setattr(machete_utils, "_machete_hopper_arch_cuda_cflags", lambda: [])

    flags = machete_utils._machete_extra_cuda_cflags()

    assert "-static-global-template-stub=false" not in flags
    assert "-static-global-template-stub=true" not in flags
    assert flags[:3] == list(machete_utils._MACHETE_REQUIRED_TORCH_NVCC_UNDEFINES)


def test_ensure_cutlass_source_bootstraps_repo_local_checkout(monkeypatch, tmp_path):
    archive_path = tmp_path / f"cutlass-v{machete_utils._CUTLASS_VERSION}.tar.gz"
    _write_fake_cutlass_archive(archive_path)

    monkeypatch.setattr(machete_utils, "_machete_project_root", lambda: tmp_path)
    monkeypatch.delenv("GPTQMODEL_CUTLASS_DIR", raising=False)
    monkeypatch.setattr(
        machete_utils,
        "_download_cutlass_archive",
        lambda _url, destination: shutil.copyfile(archive_path, destination),
    )

    cutlass_root = machete_utils._ensure_cutlass_source()
    monkeypatch.setenv("GPTQMODEL_CUTLASS_DIR", os.environ["GPTQMODEL_CUTLASS_DIR"])

    assert cutlass_root == (tmp_path / "cutlass").resolve()
    assert (cutlass_root / "include" / "cutlass" / "cutlass.h").is_file()
    assert (cutlass_root / "python" / "cutlass_library.py").is_file()
    assert (cutlass_root / machete_utils._CUTLASS_VERSION_MARKER).read_text(encoding="utf-8").strip() == machete_utils._CUTLASS_VERSION
    assert str(cutlass_root) == str((tmp_path / "cutlass").resolve())
    assert str(cutlass_root) == os.environ["GPTQMODEL_CUTLASS_DIR"]


def test_cutlass_checkout_complete_accepts_tools_util_layout(tmp_path):
    cutlass_root = tmp_path / "cutlass"
    _write_fake_cutlass_checkout(cutlass_root, version=machete_utils._CUTLASS_VERSION)
    (cutlass_root / "python" / "cutlass_library").mkdir(parents=True, exist_ok=True)
    (cutlass_root / "python" / "cutlass_library.py").unlink()
    (cutlass_root / "python" / "cutlass_library" / "__init__.py").write_text("# bindings\n", encoding="utf-8")

    assert machete_utils._cutlass_checkout_complete(cutlass_root)


def test_cutlass_checkout_version_reads_header_macros(tmp_path):
    cutlass_root = tmp_path / "cutlass"
    _write_fake_cutlass_checkout(cutlass_root, version="3.5.0")

    assert machete_utils._cutlass_checkout_version(cutlass_root) == "3.5.0"


def test_ensure_cutlass_source_rejects_mismatched_configured_checkout(monkeypatch, tmp_path):
    wrong_cutlass = tmp_path / "wrong-cutlass"
    _write_fake_cutlass_checkout(wrong_cutlass, version="3.5.0")
    monkeypatch.setattr(machete_utils, "_machete_project_root", lambda: tmp_path / "project")
    monkeypatch.setenv("GPTQMODEL_CUTLASS_DIR", str(wrong_cutlass))

    with pytest.raises(RuntimeError, match=r"CUTLASS v3\.5\.0.*requires v4\.4\.2"):
        machete_utils._ensure_cutlass_source()


def test_ensure_cutlass_source_marks_matching_repo_local_checkout_without_redownload(monkeypatch, tmp_path):
    repo_root = tmp_path
    repo_cutlass = repo_root / "cutlass"
    _write_fake_cutlass_checkout(repo_cutlass, version=machete_utils._CUTLASS_VERSION)
    download_calls: list[Path] = []

    monkeypatch.setattr(machete_utils, "_machete_project_root", lambda: repo_root)
    monkeypatch.delenv("GPTQMODEL_CUTLASS_DIR", raising=False)
    monkeypatch.setattr(
        machete_utils,
        "_download_cutlass_archive",
        lambda _url, destination: download_calls.append(destination),
    )

    cutlass_root = machete_utils._ensure_cutlass_source()

    assert cutlass_root == repo_cutlass.resolve()
    assert (repo_cutlass / machete_utils._CUTLASS_VERSION_MARKER).read_text(encoding="utf-8").strip() == machete_utils._CUTLASS_VERSION
    assert download_calls == []


def test_ensure_cutlass_source_refreshes_repo_local_checkout_when_version_mismatches(monkeypatch, tmp_path):
    archive_path = tmp_path / f"cutlass-v{machete_utils._CUTLASS_VERSION}.tar.gz"
    _write_fake_cutlass_archive(archive_path)
    repo_cutlass = tmp_path / "cutlass"
    _write_fake_cutlass_checkout(repo_cutlass, version="3.5.0")

    monkeypatch.setattr(machete_utils, "_machete_project_root", lambda: tmp_path)
    monkeypatch.delenv("GPTQMODEL_CUTLASS_DIR", raising=False)
    monkeypatch.setattr(
        machete_utils,
        "_download_cutlass_archive",
        lambda _url, destination: shutil.copyfile(archive_path, destination),
    )

    cutlass_root = machete_utils._ensure_cutlass_source()

    assert cutlass_root == repo_cutlass.resolve()
    assert machete_utils._cutlass_checkout_version(cutlass_root) == machete_utils._CUTLASS_VERSION
    assert (cutlass_root / machete_utils._CUTLASS_VERSION_MARKER).read_text(encoding="utf-8").strip() == machete_utils._CUTLASS_VERSION


def test_scaled_mm_epilogues_c3x_matches_cutlass_442_broadcast_signatures():
    header = (
        Path(__file__).resolve().parents[1]
        / "gptqmodel_ext"
        / "cutlass_extensions"
        / "epilogue"
        / "scaled_mm_epilogues_c3x.hpp"
    ).read_text(encoding="utf-8")

    assert "Sm90ColBroadcast<\n      0 /*Stages*/, TileShape, T, Stride<Int<1>, Int<0>, Int<0>>" not in header
    assert "Sm90RowBroadcast<\n      0 /*Stages*/, TileShape, T, Stride<Int<0>, Int<1>, Int<0>>" not in header
    assert "Sm90ColBroadcast<\n      0 /*Stages*/, TileShape, T, T, Stride<Int<1>, Int<0>, Int<0>>" in header
    assert "Sm90RowBroadcast<\n      0 /*Stages*/, TileShape, T, T, Stride<Int<0>, Int<1>, Int<0>>" in header


def test_machete_mm_kernel_plain_store_uses_trivial_epilogue():
    kernel_header = (
        Path(__file__).resolve().parents[1]
        / "gptqmodel_ext"
        / "machete"
        / "machete_mm_kernel.cuh"
    ).read_text(encoding="utf-8")

    assert "TrivialEpilogue<ElementAccumulator, ElementD," in kernel_header
    assert "Sm90EVT<\n      cutlass::epilogue::fusion::Sm90AccFetch>" not in kernel_header


def test_machete_sources_generate_once_when_missing(monkeypatch, tmp_path):
    machete_root = tmp_path / "gptqmodel_ext" / "machete"
    cutlass_ext_root = tmp_path / "gptqmodel_ext" / "cutlass_extensions"
    machete_root.mkdir(parents=True, exist_ok=True)
    cutlass_ext_root.mkdir(parents=True, exist_ok=True)
    (machete_root / "generate.py").write_text("# generator\n", encoding="utf-8")
    (machete_root / "machete_pytorch.cu").write_text("// pytorch\n", encoding="utf-8")
    (cutlass_ext_root / "vllm_cutlass_library_extension.py").write_text("# helper\n", encoding="utf-8")

    fake_cutlass = tmp_path / "cutlass"
    fake_cutlass.mkdir(parents=True, exist_ok=True)
    run_calls: list[list[str]] = []

    def fake_run(args, cwd, env, check, capture_output, text):
        del cwd, env, check, capture_output, text
        run_calls.append(list(args))
        generated_dir = machete_root / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        (generated_dir / "machete_dispatch.cu").write_text("// generated\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(machete_utils, "_machete_project_root", lambda: tmp_path)
    monkeypatch.setattr(machete_utils, "_ensure_cutlass_source", lambda: fake_cutlass)
    monkeypatch.setattr(subprocess, "run", fake_run)

    sources_first = machete_utils._machete_sources()
    sources_second = machete_utils._machete_sources()

    assert run_calls == [[sys.executable, str(machete_root / "generate.py")]]
    assert sources_first == sources_second
    assert sources_first[0] == str(machete_root / "machete_pytorch.cu")
    assert sources_first[1] == str(machete_root / "generated" / "machete_dispatch.cu")


def test_machete_sources_regenerate_when_cutlass_root_changes(monkeypatch, tmp_path):
    machete_root = tmp_path / "gptqmodel_ext" / "machete"
    cutlass_ext_root = tmp_path / "gptqmodel_ext" / "cutlass_extensions"
    machete_root.mkdir(parents=True, exist_ok=True)
    cutlass_ext_root.mkdir(parents=True, exist_ok=True)
    (machete_root / "generate.py").write_text("# generator\n", encoding="utf-8")
    (machete_root / "machete_pytorch.cu").write_text("// pytorch\n", encoding="utf-8")
    (cutlass_ext_root / "vllm_cutlass_library_extension.py").write_text("# helper\n", encoding="utf-8")

    cutlass_a = tmp_path / "cutlass_a"
    cutlass_b = tmp_path / "cutlass_b"
    for cutlass_root in (cutlass_a, cutlass_b):
        (cutlass_root / "python").mkdir(parents=True, exist_ok=True)
        (cutlass_root / "python" / "cutlass_library.py").write_text("# bindings\n", encoding="utf-8")

    run_calls: list[list[str]] = []
    current_cutlass_root = cutlass_a

    def fake_run(args, cwd, env, check, capture_output, text):
        del cwd, check, capture_output, text
        run_calls.append(list(args))
        generated_dir = machete_root / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        (generated_dir / "machete_dispatch.cu").write_text(
            f"// generated for {env['GPTQMODEL_CUTLASS_DIR']}\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(machete_utils, "_machete_project_root", lambda: tmp_path)
    monkeypatch.setattr(machete_utils, "_ensure_cutlass_source", lambda: current_cutlass_root)
    monkeypatch.setattr(subprocess, "run", fake_run)

    machete_utils._machete_sources()
    machete_utils._machete_sources()
    current_cutlass_root = cutlass_b
    machete_utils._machete_sources()

    assert run_calls == [
        [sys.executable, str(machete_root / "generate.py")],
        [sys.executable, str(machete_root / "generate.py")],
    ]


def test_machete_ldflags_link_cuda_driver():
    assert "-lcuda" in machete_utils._machete_extra_ldflags()


def test_vllm_cutlass_library_extension_imports_cleanly_in_subprocess(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cutlass_ext_dir = root / "gptqmodel_ext" / "cutlass_extensions"
    cutlass_python_dir = tmp_path / "cutlass_python"
    cutlass_python_dir.mkdir(parents=True, exist_ok=True)
    (cutlass_python_dir / "cutlass_library.py").write_text(
        textwrap.dedent(
            """
            import enum

            class DataType(enum.Enum):
                u4 = enum.auto()
                u8 = enum.auto()
                s4 = enum.auto()
                s8 = enum.auto()
                f16 = enum.auto()
                bf16 = enum.auto()
                e4m3 = enum.auto()
                s32 = enum.auto()
                f32 = enum.auto()

            class KernelScheduleType(enum.Enum):
                Default = enum.auto()

            DataTypeNames = {}
            DataTypeTag = {}
            DataTypeSize = {}
            KernelScheduleTag = {}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(cutlass_ext_dir)!r}); "
                f"sys.path.insert(1, {str(cutlass_python_dir)!r}); "
                "import vllm_cutlass_library_extension as ext; "
                "print(ext.VLLMDataType.u4b8.name)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "u4b8"


def _jit_scratch_root(tmp_path: Path, suffix: str) -> Path:
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else tmp_path
    root = base / "gptqmodel-jit-tests" / suffix
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dense_asymmetric_gptq_reference(
    *,
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    weight_type,
) -> torch.Tensor:
    int_weight = machete_utils.unpack_quantized_values_into_int32(
        qweight,
        weight_type,
        packed_dim=0,
    ).to(dtype=scales.dtype)
    int_zeros = machete_utils.unpack_quantized_values_into_int32(
        qzeros,
        weight_type,
        packed_dim=1,
    ).to(dtype=scales.dtype)
    dense_weight = scales[g_idx.long()] * (int_weight - int_zeros[g_idx.long()])
    return x @ dense_weight


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_machete_cuda_smoke_build_only_with_forced_arch_override(tmp_path):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc required")

    scratch_root = _jit_scratch_root(tmp_path, "machete-build-only")
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    env["GPTQMODEL_MACHETE_BUILD_ROOT"] = str(scratch_root / "machete")
    env["GPTQMODEL_MACHETE_FORCE_REBUILD"] = "1"
    env.pop("GPTQMODEL_CUTLASS_DIR", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import torch
                from gptqmodel.utils import machete as machete_utils

                ok = machete_utils._MACHETE_TORCH_OPS_EXTENSION.load()
                if not ok:
                    raise RuntimeError(machete_utils._MACHETE_TORCH_OPS_EXTENSION.last_error_message())

                namespace = getattr(torch.ops, "gptqmodel_machete", None)
                assert namespace is not None
                assert hasattr(namespace, "machete_prepack_B")
                assert hasattr(namespace, "machete_mm")
                assert hasattr(namespace, "machete_supported_schedules")
                print("compiled")
                """
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "compiled" in result.stdout


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_machete_cuda_smoke_build_and_forward(monkeypatch, tmp_path):
    if not machete_utils._validate_machete_device_support():
        pytest.skip(machete_utils.machete_runtime_error())

    scratch_root = _jit_scratch_root(tmp_path, "machete")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("GPTQMODEL_CUTLASS_DIR", raising=False)
    monkeypatch.setenv("GPTQMODEL_MACHETE_BUILD_ROOT", str(scratch_root / "machete"))
    monkeypatch.setenv("GPTQMODEL_MACHETE_FORCE_REBUILD", "1")

    assert extension_api.load(name="machete", use_cache=False) == {"machete": True}

    device = torch.device("cuda:0")
    module = machete_linear_module.MacheteLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        bias=False,
        dtype=torch.float16,
    ).to(device)
    with torch.no_grad():
        module.qweight.copy_(torch.randint(0, 16, module.qweight.shape, device=device, dtype=torch.int32))
        module.g_idx.copy_(torch.arange(module.in_features, device=device, dtype=torch.int32))
        module.scales.copy_(torch.ones_like(module.scales, device=device))
    module.post_init()

    out = module(torch.randn(4, 128, device=device, dtype=torch.float16))
    torch.cuda.synchronize(device)

    assert out.shape == (4, 128)
    assert out.dtype == torch.float16


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_machete_cuda_asymmetric_gptq_matches_dense_reference(monkeypatch, tmp_path):
    if not machete_utils._validate_machete_device_support():
        pytest.skip(machete_utils.machete_runtime_error())

    scratch_root = _jit_scratch_root(tmp_path, "machete-asym")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("GPTQMODEL_CUTLASS_DIR", raising=False)
    monkeypatch.setenv("GPTQMODEL_MACHETE_BUILD_ROOT", str(scratch_root / "machete"))

    assert extension_api.load(name="machete", use_cache=True) == {"machete": True}

    device = torch.device("cuda:0")
    module = machete_linear_module.MacheteLinear(
        bits=4,
        group_size=64,
        desc_act=False,
        sym=False,
        in_features=128,
        out_features=128,
        bias=False,
        dtype=torch.float16,
    ).to(device)

    qweight = torch.randint(0, 16, module.qweight.shape, device=device, dtype=torch.int32)
    qzeros = torch.randint(0, 16, module.qzeros.shape, device=device, dtype=torch.int32)
    scales = (torch.rand(module.scales.shape, device=device, dtype=torch.float16) + 0.25).contiguous()
    g_idx = torch.tensor([0] * 64 + [1] * 64, device=device, dtype=torch.int32)
    x = torch.randn(8, 128, device=device, dtype=torch.float16)

    with torch.no_grad():
        module.qweight.copy_(qweight)
        module.qzeros.copy_(qzeros)
        module.scales.copy_(scales)
        module.g_idx.copy_(g_idx)

    expected = _dense_asymmetric_gptq_reference(
        x=x,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        g_idx=g_idx,
        weight_type=module.weight_type,
    )

    module.post_init()
    actual = module(x)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
