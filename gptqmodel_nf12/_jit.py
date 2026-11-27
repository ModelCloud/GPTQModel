# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Build GPTQModel's vendored NF12 libraries in a source-keyed cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import torch


try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None


_LIBRARIES = ("libnf12.so", "libnf12_torch.so")
_BUILD_ENV = (
    "CUDACXX", "CUDA_HOME", "CUDA_PATH", "CUDAHOSTCXX", "CXX", "CXXFLAGS",
    "CUDAFLAGS", "LDFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
)


class _MissingBuildTools(RuntimeError):
    """A source build is required but its compiler tools are unavailable."""


# All process-level state below is guarded by this lock. The repository tests
# and ships a free-threaded lane (``PYTHON_GIL=0``), where check-then-set on a
# module global is not implicitly atomic, and a shared exception instance must
# not be re-raised from several threads at once. ``RLock`` because
# ``library_directory`` holds it while resolving the target and probing tools.
_lock = threading.RLock()
_directory: Path | None = None
# A *build* that raised is remembered for the process: without this, a
# toolchain that cannot build (an old CMake, a broken compiler) is retried on
# every NF12 call - once per module and subset during quantization. Only the
# ``_build`` call is cached: a transient ``mkdir``/lock-file/``_ready`` failure
# (a temporarily unwritable cache, say) must be retried on the next call.
# Stored as ``(type, args)`` so each caller raises a fresh instance: ``raise``
# mutates the instance's traceback, which is a data race when threads share
# one. ``reset_runtime_cache`` forgets it after the environment is repaired.
_directory_failure: tuple[type[BaseException], tuple] | None = None
_target: tuple[Path, list[int], Path] | None = None
_tools: tuple[str | None, str | None, str | None] | None = None
_UNSET = object()
_toolchain: object = _UNSET

#: The vendored CMakeLists requires this; ``_build_tools`` alone cannot tell.
CMAKE_MIN_VERSION = (3, 26)


def _lock_file(handle) -> None:
    """Take an exclusive inter-process lock on an open build lock file.

    ``fcntl`` does not exist on Windows, where this module still has to import
    so the portable NF12 path and its capability probe keep working (#5382).
    Windows locks a byte range instead; either lock is released when the
    caller's handle is closed.
    """
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return
    import msvcrt  # noqa: PLC0415 - Windows-only fallback

    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)


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


def _build_tools() -> tuple[str | None, str | None, str | None]:
    """Locate ``(cmake, ninja, nvcc)`` once; call ``reset_runtime_cache`` to re-probe."""

    global _tools
    with _lock:
        if _tools is None:
            # Import CUDA toolkit discovery only when the tools are actually needed.
            from torch.utils.cpp_extension import CUDA_HOME

            cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
            nvcc = os.environ.get("CUDACXX")
            if not nvcc:
                nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME else "nvcc"
            _tools = (cmake, ninja, shutil.which(nvcc))
        return _tools


def _tool_version(tool: str) -> tuple[int, ...] | None:
    """Best-effort ``(major, minor, ...)`` for a build tool.

    ``None`` means "could not tell": a probe that fails for any other reason
    must not turn into a "missing tools" skip, because the build attempt below
    still reports the real error (a compiler failure is not a missing tool).
    """

    try:
        completed = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # ``subprocess.run`` always returns a CompletedProcess, but a test double may
    # return something else; an unknown version must fall through to the build.
    if getattr(completed, "returncode", 0) != 0:
        return None
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", getattr(completed, "stdout", None) or "")
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def _probe_toolchain_problem() -> str | None:
    """Why a source build cannot succeed here, or ``None``."""

    cmake, ninja, nvcc = _build_tools()
    missing = [name for name, path in (("cmake", cmake), ("ninja", ninja), ("nvcc", nvcc)) if not path]
    if missing:
        return "missing build tools: " + ", ".join(missing)
    version = _tool_version(cmake)
    if version is not None and version < CMAKE_MIN_VERSION:
        found = ".".join(str(part) for part in version)
        required = ".".join(str(part) for part in CMAKE_MIN_VERSION)
        return f"cmake {found} is older than {required}"
    return None


def _toolchain_problem() -> str | None:
    """Cached :func:`_probe_toolchain_problem`; one ``--version`` probe per process.

    ``source_build_available`` promises callers a portable fallback instead of
    a doomed build; existence alone is not enough, because the vendored
    CMakeLists requires 3.26 and an older CMake fails at configure time. The
    probe spawns a subprocess, so concurrent callers share one result.
    """

    global _toolchain
    with _lock:
        if _toolchain is _UNSET:
            _toolchain = _probe_toolchain_problem()
        return _toolchain


def source_build_available() -> bool:
    """Return whether a source build can be produced for the visible GPUs.

    Callers use this to choose a portable fallback instead of starting a build
    that is guaranteed to fail; :func:`_build` remains the authority and still
    reports the concrete missing tool when a build is attempted.
    """
    if sys.platform == "win32":
        # The builder emits ELF shared objects that the loader opens by that
        # name, so there is no supported native NF12 build on Windows (#5382).
        return False
    try:
        archs = _architectures()
    except (ImportError, RuntimeError):
        return False
    if not archs or min(archs) < 80:
        return False
    return _toolchain_problem() is None


def _build(source: Path, directory: Path, archs: list[int]) -> None:
    cmake, ninja, nvcc = _build_tools()
    problem = _toolchain_problem()
    if problem is not None:
        raise _MissingBuildTools(
            "nf12: JIT compilation requires CUDA toolkit 13.0+ (nvcc), CMake 3.26+, "
            "Ninja, and a C++ compiler; set CUDA_HOME or CUDACXX to locate CUDA "
            f"({problem})"
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


def _cache_target() -> tuple[Path, list[int], Path]:
    """Resolve the source, visible architectures and cache directory once.

    ``_cache_key`` hashes every vendored source file; recomputing it per module
    was paid on every retry of a failing build.
    """

    global _target
    with _lock:
        if _target is None:
            source = _source_root()
            archs = _architectures()
            cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
            directory = cache.expanduser().resolve() / "gptqmodel" / "nf12" / _cache_key(source, archs)
            _target = (source, archs, directory)
        return _target


def cache_directory() -> Path:
    """Return the source-keyed build directory without creating or building it."""
    return _cache_target()[2]


def cache_ready() -> bool:
    """Return whether a completed build already exists for the visible GPUs."""
    try:
        return _ready(cache_directory())
    except (ImportError, OSError, RuntimeError):
        return False


def library_directory() -> Path:
    global _directory, _directory_failure
    with _lock:
        if _directory is not None:
            return _directory
        if _directory_failure is not None:
            exc_type, args = _directory_failure
            raise exc_type(*args)
        source, archs, directory = _cache_target()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Keep the lock file: unlinking it lets waiters lock a different inode.
        with (directory / "build.lock").open("a") as lock:
            _lock_file(lock)
            if not _ready(directory):
                (directory / "complete").unlink(missing_ok=True)
                try:
                    _build(source, directory, archs)
                except Exception as exc:
                    # A broken toolchain is a property of the process, not of
                    # one call - unlike a filesystem error above, which is not
                    # cached and is retried by the next caller.
                    _directory_failure = (type(exc), exc.args)
                    raise
        _directory = directory / "build"
        return _directory


def reset_runtime_cache() -> None:
    """Forget the resolved cache directory, a failed build and tool probes.

    Needed by tests, and by a caller that repairs the toolchain and wants the
    next NF12 use to try again. Successful loads stay memoized elsewhere.
    """

    global _directory, _directory_failure, _target, _tools, _toolchain
    with _lock:
        _directory = None
        _directory_failure = None
        _target = None
        _tools = None
        _toolchain = _UNSET
