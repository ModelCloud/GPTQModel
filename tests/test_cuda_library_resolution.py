# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Link-time resolution of CUDA math libraries for torch.ops JIT extensions.

CUDA 13 ships cuBLAS in a wheel (``nvidia/cu13/lib/libcublas.so.13``) with no
unversioned ``libcublas.so`` development symlink, so ``-lcublas`` fails to link
even though the library is installed. The AllSpark W8A16 extension needs
cuBLAS, so it links the resolved path instead of the bare flag.
"""

from __future__ import annotations

import gptqmodel.utils.allspark as allspark_utils
import gptqmodel.utils.cpp as cpp_utils


def _toolkit(tmp_path, *names):
    lib64 = tmp_path / "lib64"
    lib64.mkdir()
    for name in names:
        (lib64 / name).write_bytes(b"")
    return lib64


def test_resolve_prefers_the_toolkit_development_symlink(tmp_path, monkeypatch):
    _toolkit(tmp_path, "libcublas.so", "libcublas.so.13")
    monkeypatch.setattr(cpp_utils, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(cpp_utils, "detected_cuda_wheel_library_paths", lambda: [])

    assert cpp_utils.resolve_cuda_library("cublas") == str(tmp_path / "lib64" / "libcublas.so")


def test_resolve_falls_back_to_the_versioned_wheel_library(tmp_path, monkeypatch):
    toolkit = tmp_path / "toolkit"
    toolkit.mkdir()
    wheel_lib = tmp_path / "nvidia" / "cu13" / "lib"
    wheel_lib.mkdir(parents=True)
    (wheel_lib / "libcublas.so.13").write_bytes(b"")

    monkeypatch.setattr(cpp_utils, "CUDA_HOME", str(toolkit))
    monkeypatch.setattr(
        cpp_utils, "detected_cuda_wheel_library_paths", lambda: [str(wheel_lib)]
    )

    assert cpp_utils.resolve_cuda_library("cublas") == str(wheel_lib / "libcublas.so.13")


def test_resolve_returns_none_when_the_library_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(cpp_utils, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(cpp_utils, "detected_cuda_wheel_library_paths", lambda: [])

    assert cpp_utils.resolve_cuda_library("cublas") is None


def test_allspark_links_the_resolved_cublas_path(tmp_path, monkeypatch):
    wheel_lib = tmp_path / "lib"
    wheel_lib.mkdir()
    (wheel_lib / "libcublas.so.13").write_bytes(b"")

    monkeypatch.setattr(allspark_utils, "resolve_cuda_library", lambda name: str(wheel_lib / "libcublas.so.13"))
    assert allspark_utils._allspark_ldflags() == [str(wheel_lib / "libcublas.so.13")]

    monkeypatch.setattr(allspark_utils, "resolve_cuda_library", lambda name: None)
    assert allspark_utils._allspark_ldflags() == ["-lcublas"]
