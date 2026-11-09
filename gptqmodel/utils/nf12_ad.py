# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the nf12 reference implementation
# (https://github.com/dphnAI/nf12) by AlpinDale and the nf12 contributors.
"""Lazy loader for the vendored NF12-AD CUDA extension.

NF12 (NyoomFloat12) and its CUDA kernels come from
https://github.com/dphnAI/nf12 by AlpinDale, based on the Together AI paper
"NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit Weight
Compression" (Liberman, Fang, Zhang, Dao, Fu, 2026). The sources are vendored
under ``gptqmodel_ext/nf12`` with their Apache-2.0 license and attribution.

Loading order:

1. An installed upstream ``nf12`` package is reused when present.
2. Otherwise the vendored CMake project is built into a source-hashed cache
   directory and ``torch.ops.nf12`` is loaded from it.

Every entry point is best effort. Callers fall back to
``gptqmodel.utils.nf12_codec`` on CPU or when a CUDA toolchain is missing.
"""

from __future__ import annotations

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


log = logging.getLogger(__name__)

# JIT-built artifact names. The op namespace stays ``nf12`` (upstream); only
# the shared objects are GPT-QModel-specific. Nothing is shipped precompiled:
# both libraries are always built from the vendored sources.
_CODEC_LIB = "libnf12_ad.so"
_TORCH_LIB = "libnf12_ad_torch.so"

_LOCK = threading.Lock()
_LOADED = False
_ERROR: str | None = None
_BUILD_DIR: Path | None = None


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "nf12"


def _cache_root() -> Path:
    configured = os.environ.get("GPTQMODEL_NF12_AD_BUILD_ROOT")
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base.expanduser().resolve() / "gptqmodel" / "nf12_ad"


def _visible_archs() -> list[int]:
    archs = sorted(
        {
            major * 10 + minor
            for major, minor in (
                torch.cuda.get_device_capability(index)
                for index in range(torch.cuda.device_count())
            )
        }
    )
    if not archs:
        raise RuntimeError("NF12-AD JIT compilation requires a visible CUDA GPU")
    if archs[0] < 80:
        raise RuntimeError(
            f"NF12-AD requires SM80 or newer, found SM{archs[0]}"
        )
    return archs


def _source_digest(source: Path, archs: list[int]) -> str:
    digest = hashlib.sha256()
    identity = {
        "archs": archs,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.executable,
        "platform": (sys.platform, platform.machine()),
        "source": str(source.resolve()),
    }
    digest.update(json.dumps(identity, sort_keys=True).encode())
    files = [source / "CMakeLists.txt"]
    for directory in ("csrc", "include"):
        files.extend(path for path in (source / directory).rglob("*") if path.is_file())
    for path in sorted(files):
        digest.update(str(path.relative_to(source)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _ready(directory: Path) -> bool:
    return (directory / "complete").is_file() and all(
        (directory / "build" / name).is_file()
        for name in (_CODEC_LIB, _TORCH_LIB)
    )


def _build(source: Path, directory: Path, archs: list[int]) -> None:
    from torch.utils.cpp_extension import CUDA_HOME

    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    nvcc = os.environ.get("CUDACXX")
    if not nvcc and CUDA_HOME:
        nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc")
    nvcc = shutil.which(nvcc or "nvcc")
    if not cmake or not ninja or not nvcc:
        raise RuntimeError(
            "NF12-AD JIT compilation requires CUDA toolkit 13.0+ (nvcc), "
            "CMake 3.26+, Ninja, and a C++ compiler; set CUDA_HOME or CUDACXX"
        )

    build = directory / "build"
    build_log = directory / "build.log"
    env = os.environ.copy()
    env.pop("NF12_CUDA_ARCHS", None)
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    commands = (
        [
            cmake,
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DNF12_BUILD_NATIVE=ON",
            "-DNF12_BUILD_TORCH=ON",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DCMAKE_CUDA_COMPILER={nvcc}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DNF12_CUDA_ARCHS=" + ";".join(str(arch) for arch in archs),
        ],
        [cmake, "--build", str(build), "--target", "nf12", "nf12_torch"],
    )
    log.info("Building NF12-AD for %s; log: %s", archs, build_log)
    try:
        with build_log.open("w", encoding="utf-8") as output:
            for command in commands:
                subprocess.run(
                    command,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"NF12-AD JIT compilation failed; see {build_log}") from exc
    if not all(
        (build / name).is_file()
        for name in (_CODEC_LIB, _TORCH_LIB)
    ):
        raise RuntimeError(f"NF12-AD JIT build did not produce both libraries; see {build_log}")
    (directory / "complete").touch()


def _load_installed() -> bool:
    try:
        import nf12  # type: ignore[import-not-found]
    except ImportError:
        return False
    if Path(nf12.__file__).resolve().is_relative_to(_source_root()):
        return False
    nf12.load()
    # PyTorch creates an operator namespace on attribute access, so checking
    # `hasattr(torch.ops, "nf12")` succeeds even when it contains no operators.
    namespace = torch.ops.nf12
    return all(callable(getattr(namespace, name, None)) for name in ("encode", "decode", "linear"))


def _build_and_load() -> None:
    global _BUILD_DIR
    source = _source_root()
    if not (source / "CMakeLists.txt").is_file():
        raise RuntimeError(f"NF12-AD sources are missing at {source}")
    archs = _visible_archs()
    directory = _cache_root() / _source_digest(source, archs)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        fcntl = None
    lock_path = directory / "build.lock"
    with lock_path.open("a") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        if not _ready(directory):
            (directory / "complete").unlink(missing_ok=True)
            _build(source, directory, archs)
    build = directory / "build"
    # libnf12_ad_torch.so links libnf12_ad.so; load the codec first.
    torch.ops.load_library(str(build / _CODEC_LIB))
    torch.ops.load_library(str(build / _TORCH_LIB))
    _BUILD_DIR = build


def _register_fakes() -> None:
    """Register meta kernels for ``torch.compile`` when torch supports it."""

    try:
        register_fake = torch.library.register_fake
    except AttributeError:  # pragma: no cover - old torch
        return

    def _fake(name, fn):
        try:
            register_fake(f"nf12::{name}")(fn)
        except (RuntimeError, ValueError):
            # The upstream package may already have registered it.
            pass

    def _encode(w):
        ctx = torch.library.get_ctx()
        n, k = w.shape
        return (
            w.new_empty((n, k * 3 // 2), dtype=torch.uint8),
            w.new_empty((ctx.new_dynamic_size(), 8), dtype=torch.int16),
            ctx.new_dynamic_size(min=0, max=120),
        )

    def _decode(packed, escapes, exp_base):
        n, row_bytes = packed.shape
        return packed.new_empty((n, row_bytes * 2 // 3), dtype=torch.bfloat16)

    def _linear(x, packed, escapes, exp_base, bias=None):
        return x.new_empty((*x.shape[:-1], packed.shape[0]))

    _fake("encode", _encode)
    _fake("decode", _decode)
    _fake("linear", _linear)


def load() -> bool:
    """Load the NF12-AD native extension, returning whether it is available."""

    global _LOADED, _ERROR
    if _LOADED:
        return True
    with _LOCK:
        if _LOADED:
            return True
        if not torch.cuda.is_available():
            _ERROR = "NF12-AD native kernels require an available CUDA device."
            return False
        try:
            loaded_installed = False
            try:
                loaded_installed = _load_installed()
            except Exception as exc:  # noqa: BLE001 - an installed package may be stale
                log.debug("Installed nf12 package could not be loaded: %s", exc)
            if not loaded_installed:
                _build_and_load()
            _register_fakes()
        except Exception as exc:  # noqa: BLE001 - availability probe
            _ERROR = f"{type(exc).__name__}: {exc}"
            log.debug("NF12-AD native extension unavailable: %s", _ERROR)
            return False
        _ERROR = None
        _LOADED = True
        return True


def available() -> bool:
    """Return whether the fused CUDA kernel can run on this host."""

    return load()


def last_error() -> str | None:
    """Return the reason the native extension is unavailable, if any."""

    return _ERROR


def build_directory() -> Path | None:
    """Return the directory containing the loaded libraries, if JIT-built."""

    return _BUILD_DIR


def encode(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode a BF16 CUDA weight with the native kernel."""

    if not load():
        raise RuntimeError(_ERROR or "NF12-AD native extension is unavailable")
    packed, escapes, exp_base = torch.ops.nf12.encode(weight)
    return packed, escapes, int(exp_base)


def decode(
    packed: torch.Tensor,
    escapes: torch.Tensor,
    exp_base: int,
) -> torch.Tensor:
    """Decode NF12-AD bytes with the native kernel."""

    if not load():
        raise RuntimeError(_ERROR or "NF12-AD native extension is unavailable")
    return torch.ops.nf12.decode(packed, escapes, int(exp_base))


def linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    escapes: torch.Tensor,
    exp_base: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fused NF12-AD linear kernel."""

    if not load():
        raise RuntimeError(_ERROR or "NF12-AD native extension is unavailable")
    return torch.ops.nf12.linear(x, packed, escapes, int(exp_base), bias)


__all__ = [
    "available",
    "build_directory",
    "decode",
    "encode",
    "last_error",
    "linear",
    "load",
]
