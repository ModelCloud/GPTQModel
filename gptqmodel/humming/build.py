# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

"""Build native artifacts shipped in Humming wheels."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths

from .utils.cubin import build_cubin_patcher
from .utils.jit import get_native_arch
from .utils.nvrtc import build_nvrtc_compile_binary

ROOT = Path(__file__).resolve().parents[1]
TARGET_TORCH_VERSION = "0x020B000000000000"


def _architecture() -> str:
    arch = get_native_arch()
    if arch is None:
        raise RuntimeError("Unsupported architecture")
    return arch


def _find_cuda_include() -> Path:
    candidates: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if value := os.environ.get(variable):
            candidates.append(Path(value) / "include")
    candidates.extend(Path(entry) / "nvidia" / "cuda_runtime" / "include" for entry in sys.path if entry)
    candidates.extend(Path(entry) / "nvidia" / "cu13" / "include" for entry in sys.path if entry)
    for candidate in candidates:
        if (candidate / "cuda.h").is_file():
            return candidate.resolve()
    raise RuntimeError("CUDA headers were not found; install nvidia-cuda-runtime-cu12 or set CUDA_HOME")


def _torch_library(name: str) -> Path:
    library = Path(torch.__file__).resolve().parent / "lib" / name
    if not library.is_file():
        raise RuntimeError(f"{name} is required; build with a CUDA-enabled torch wheel")
    return library


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build_launcher(output: Path, compiler: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cuda_include = _find_cuda_include()
    torch_cpu = _torch_library("libtorch_cpu.so")
    torch_cuda = _torch_library("libtorch_cuda.so")

    with tempfile.TemporaryDirectory(prefix="humming-link-") as temporary_dir:
        temporary = Path(temporary_dir)
        empty_source = temporary / "empty.c"
        cuda_stub = temporary / "libcuda.so"
        empty_source.write_text("void humming_cuda_stub(void) {}\n")
        _run(["cc", "-shared", str(empty_source), "-Wl,-soname,libcuda.so.1", "-o", str(cuda_stub)])

        command = [
            compiler,
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            "-DUSE_TORCH_STABLE_API=1",
            f"-DTORCH_TARGET_VERSION={TARGET_TORCH_VERSION}",
            str(ROOT / "humming/csrc/launcher/launcher.cpp"),
            "-o",
            str(output),
        ]
        command.extend(f"-I{path}" for path in include_paths())
        command.extend((f"-I{cuda_include}", f"-L{temporary}"))
        command.extend(
            (
                "-Wl,--no-as-needed",
                "-lcuda",
                str(torch_cpu),
                str(torch_cuda),
                "-Wl,-rpath,$ORIGIN/../../../torch/lib",
                "-Wl,--as-needed",
            )
        )
        _run(command)

    print(f"Built {output} for {_architecture()} with torch {torch.__version__}")


def build_native(output_dir: str | Path | None = None) -> Path:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("A C++ compiler is required")
    native_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else ROOT / "humming" / "_native" / _architecture()
    )
    build_launcher(native_dir / "libhumming_launcher.so", compiler)
    build_cubin_patcher(native_dir / "libcubinpatch.so", compiler=compiler)
    build_nvrtc_compile_binary(native_dir / "nvrtc_compile", compiler=compiler)
    return native_dir
