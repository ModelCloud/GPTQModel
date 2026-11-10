# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Build GPTQModel's vendored NF12 libraries in a source-keyed cache."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import torch


_LIBRARIES = ("libnf12_ad.so", "libnf12_ad_torch.so")
_BUILD_ENV = (
    "CUDACXX", "CUDA_HOME", "CUDA_PATH", "CUDAHOSTCXX", "CXX", "CXXFLAGS",
    "CUDAFLAGS", "LDFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
)
_lock = threading.Lock()
_directory: Path | None = None


def _source_root() -> Path:
    package = Path(__file__).resolve().parent
    # In an editable checkout and in a wheel the vendored sources live next
    # to this top-level Python package.
    for root in (package / "_src", package.parent / "gptqmodel_ext" / "nf12"):
        if ((root / "CMakeLists.txt").is_file() and (root / "csrc").is_dir()
                and (root / "include").is_dir()):
            return root
    raise ImportError(
        "gptqmodel_nf12: native binaries and CUDA sources are missing; reinstall GPTQModel "
        "or use a source checkout"
    )


def _architectures() -> list[int]:
    if torch.version.hip is not None or torch.version.cuda is None:
        raise RuntimeError("nf12: native NF12 kernels require an NVIDIA CUDA PyTorch build")
    archs = sorted({major * 10 + minor for major, minor in (
        torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
    )})
    if not archs:
        raise RuntimeError("nf12: JIT compilation requires a visible CUDA GPU")
    if archs[0] < 80:
        raise RuntimeError(
            f"nf12: JIT compilation requires SM80 or newer, found SM{archs[0]}; "
            "select supported GPUs with CUDA_VISIBLE_DEVICES"
        )
    return archs


def _cache_key(source: Path, archs: list[int]) -> str:
    digest = hashlib.sha256()
    identity = {
        "archs": archs,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_path": str(Path(torch.__file__).resolve().parent),
        "python": sys.executable,
        "platform": (sys.platform, platform.machine()),
        "environment": {name: os.environ.get(name) for name in _BUILD_ENV},
        "source_path": str(source.resolve()),
    }
    digest.update(json.dumps(identity, sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    files = [source / "CMakeLists.txt"]
    for directory in ("csrc", "include"):
        files.extend(path for path in (source / directory).rglob("*") if path.is_file())
    for path in sorted(files):
        digest.update(str(path.relative_to(source)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _ready(directory: Path) -> bool:
    return (directory / "complete").is_file() and all(
        (directory / "build" / name).is_file() for name in _LIBRARIES
    )


def _build(source: Path, directory: Path, archs: list[int]) -> None:
    # Import CUDA toolkit discovery only on a cache miss.
    from torch.utils.cpp_extension import CUDA_HOME

    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    nvcc = os.environ.get("CUDACXX")
    if not nvcc:
        nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME else "nvcc"
    nvcc = shutil.which(nvcc)
    if not cmake or not ninja or not nvcc:
        raise RuntimeError(
            "nf12: JIT compilation requires CUDA toolkit 13.0+ (nvcc), CMake 3.26+, "
            "Ninja, and a C++ compiler; set CUDA_HOME or CUDACXX to locate CUDA"
        )

    build = directory / "build"
    log = directory / "build.log"
    env = os.environ.copy()
    # This is a runtime build for visible devices, not the wheel's target list.
    env.pop("NF12_CUDA_ARCHS", None)
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    commands = (
        [cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DNF12_BUILD_NATIVE=ON", "-DNF12_BUILD_TORCH=ON",
         f"-DCMAKE_MAKE_PROGRAM={ninja}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
         f"-DPython_EXECUTABLE={sys.executable}",
         "-DNF12_CUDA_ARCHS=" + ";".join(map(str, archs))],
        [cmake, "--build", str(build), "--target", "nf12", "nf12_torch"],
    )
    logging.getLogger(__name__).info("Building NF12 for %s; log: %s", archs, log)
    try:
        with log.open("w") as output:
            for command in commands:
                subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"nf12: JIT compilation failed; see {log}") from error
    if not all((build / name).is_file() for name in _LIBRARIES):
        raise RuntimeError(f"nf12: JIT build did not produce both native libraries; see {log}")
    (directory / "complete").touch()


def library_directory() -> Path:
    global _directory
    with _lock:
        if _directory is not None:
            return _directory
        source = _source_root()
        archs = _architectures()
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        directory = cache.expanduser().resolve() / "gptqmodel" / "nf12_ad" / _cache_key(source, archs)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Keep the lock file: unlinking it lets waiters lock a different inode.
        with (directory / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not _ready(directory):
                (directory / "complete").unlink(missing_ok=True)
                _build(source, directory, archs)
        _directory = directory / "build"
        return _directory
