# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""HIP JIT builds must not hipify into the shipped source tree (#4072).

torch's ROCm loader writes ``<name>_hip.{hip,cpp}`` next to each source it
processes. The generated names are gitignored, so a source-tree mutation stays
invisible to ``git status``; a read-only site-packages install fails outright.
These tests pin the staging contract that keeps every generated file under the
writable build cache.
"""

from pathlib import Path

import pytest

from gptqmodel.utils import cpp
from gptqmodel.utils.cpp import TorchOpsJitExtension


def _extension(tmp_path: Path, *, stage: bool) -> TorchOpsJitExtension:
    source_root = tmp_path / "gptqmodel_ext" / "demo"
    source_root.mkdir(parents=True)
    (source_root / "demo.cu").write_text("#include \"demo.cuh\"\n", encoding="utf-8")
    (source_root / "demo.cuh").write_text("// header\n", encoding="utf-8")
    return TorchOpsJitExtension(
        name="demo_ops",
        namespace="demo",
        required_ops=("run",),
        sources=lambda: [str(source_root / "demo.cu")],
        build_root_env=None,
        default_build_root=tmp_path / "build",
        display_name="Demo",
        extra_include_paths=lambda: [str(source_root)],
        requires_cuda=True,
        stage_hip_sources=stage,
    )


def test_hip_sources_are_staged_under_the_build_root(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp, "IS_ROCM", True)
    extension = _extension(tmp_path, stage=True)
    source_root = tmp_path / "gptqmodel_ext" / "demo"

    sources, includes = extension._resolved_sources_and_include_paths()

    staged_root = tmp_path / "build" / "_src"
    assert sources == [str(staged_root / "demo.cu")]
    assert includes[0] == str(staged_root)
    assert (staged_root / "demo.cu").read_text(encoding="utf-8") == "#include \"demo.cuh\"\n"
    assert (staged_root / "demo.cuh").read_text(encoding="utf-8") == "// header\n"
    # The shipped tree is copied, never generated into.
    assert sorted(path.name for path in source_root.iterdir()) == ["demo.cu", "demo.cuh"]


def test_staging_is_disabled_off_rocm(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp, "IS_ROCM", False)
    extension = _extension(tmp_path, stage=True)
    source_root = tmp_path / "gptqmodel_ext" / "demo"

    sources, includes = extension._resolved_sources_and_include_paths()

    assert sources == [str(source_root / "demo.cu")]
    assert includes[0] == str(source_root)
    assert not (tmp_path / "build" / "_src").exists()


@pytest.mark.parametrize(
    "module_name",
    [
        "gptqmodel.quantization.gptq_cuda",
        "gptqmodel.utils.allspark_hip",
        "gptqmodel.utils.hadamard",
        "gptqmodel.utils.paroquant",
        "gptqmodel.exllamav3.ext",
    ],
)
def test_every_hip_loader_stages_its_sources(module_name):
    import importlib

    module = importlib.import_module(module_name)
    extensions = [
        value
        for value in vars(module).values()
        if isinstance(value, TorchOpsJitExtension)
    ]

    assert extensions, f"{module_name} has no TorchOpsJitExtension"
    assert all(extension.stage_hip_sources for extension in extensions)
