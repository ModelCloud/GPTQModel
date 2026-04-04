# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch

import gptqmodel.utils.machete as machete_utils
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


def _write_fake_cutlass_archive(destination: Path) -> None:
    staging_root = destination.parent / "cutlass-3.5.0"
    (staging_root / "include" / "cutlass").mkdir(parents=True, exist_ok=True)
    (staging_root / "examples" / "common" / "include").mkdir(parents=True, exist_ok=True)
    (staging_root / "tools" / "library" / "include").mkdir(parents=True, exist_ok=True)
    (staging_root / "python").mkdir(parents=True, exist_ok=True)
    (staging_root / "include" / "cutlass" / "cutlass.h").write_text("// cutlass\n", encoding="utf-8")
    (staging_root / "python" / "cutlass_library.py").write_text("# cutlass python\n", encoding="utf-8")

    with tarfile.open(destination, "w:gz") as archive:
        archive.add(staging_root, arcname="cutlass-3.5.0")

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


def test_ensure_cutlass_source_bootstraps_repo_local_checkout(monkeypatch, tmp_path):
    archive_path = tmp_path / "cutlass-v3.5.0.tar.gz"
    _write_fake_cutlass_archive(archive_path)

    monkeypatch.setattr(machete_utils, "_machete_project_root", lambda: tmp_path)
    monkeypatch.delenv("GPTQMODEL_CUTLASS_DIR", raising=False)
    monkeypatch.setattr(
        machete_utils,
        "_download_cutlass_archive",
        lambda _url, destination: shutil.copyfile(archive_path, destination),
    )

    cutlass_root = machete_utils._ensure_cutlass_source()

    assert cutlass_root == (tmp_path / "cutlass").resolve()
    assert (cutlass_root / "include" / "cutlass" / "cutlass.h").is_file()
    assert (cutlass_root / "python" / "cutlass_library.py").is_file()
    assert str(cutlass_root) == str((tmp_path / "cutlass").resolve())
    assert str(cutlass_root) == os.environ["GPTQMODEL_CUTLASS_DIR"]


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


def test_machete_ldflags_link_cuda_driver():
    assert "-lcuda" in machete_utils._machete_extra_ldflags()
